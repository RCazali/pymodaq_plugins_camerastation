"""Camera Station application: wires the dashboard cameras, the engine, the publication, the recording,
the trigger files and the user interface.

Threads: the GUI thread receives the frames (``grab_done_signal`` of the 2D viewers), runs the timers
(publication, status, triggers) and never computes an analysis; the engine has one worker thread; the
HDF5 writer has one thread. Every slot catches its exceptions: a camera or a disk problem is shown
and logged, it never stops the station nor the dashboard.
"""
from __future__ import annotations

import collections
import datetime as _dt
import inspect
import logging
import logging.handlers
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from qtpy import QtCore, QtWidgets

from pymodaq_gui import utils as gutils
from pymodaq_gui.parameter import Parameter, ParameterTree

from ..core.analysis import clip_roi
from ..core.engine import AnalysisEngine, STATE_TEXT, ST_NO_FRAMES, ST_OFF, ST_OK, ST_WAITING
from ..core.publisher import Publisher, flatten
from ..core.recorder import H5StationWriter, IMG_EVENT, IMG_MANUAL, IMG_PERIODIC, reduce_image
from ..core.scripts import ScriptLibrary, ScriptRunner
from ..core.settings import CameraConfig, ProfileStore, SettingsStore, StationConfig, clean_name
from ..core.triggers import TriggerWatcher
from ..relay import bus
from ..utils import config_get, machine_name, resolve_paths
from .camera_panel import CameraPanel
from .params import (NO_DETECTOR, SCRIPT_PREFIX, camera_scripts_children, camera_spec, script_param_spec,
                     script_spec, station_spec, trend_plot_spec)
from .preview import PreviewWidget
from .trends import TrendsWidget, camera_specs, script_specs

RELAY_PLUGIN = 'CameraStation'
EXAMPLE_SCRIPTS = Path(__file__).resolve().parents[1] / 'resources' / 'scripts'
STATION_KEYS = {'publish_hz', 'stale_s', 'trigger_folder', 'image_period_s', 'image_binning', 'image_bits',
                'image_full_frame', 'burst_frames', 'preview', 'script_timeout_s', 'display_hz', 'trends',
                'trend_history_s'}
SCRIPT_COLUMNS = ('Script', 'Camera', 'State', 'ms', 'Runs', 'Values')
CAMERA_KEYS = {'name', 'detector', 'enabled', 'kind', 'rate_hz', 'binning', 'background', 'threshold',
               'min_signal', 'saturation', 'mask_cx', 'mask_cy', 'mask_radius', 'record_images'}
VALUE_NAMES = {'sx': 'σx', 'sy': 'σy', 'sat': 'sat%'}


def plugin_name(det) -> str:
    """Instrument plugin of a viewer. ``DAQ_Viewer.detector`` is the plugin name up to PyMoDAQ 5.2 and a
    ``SelectedModule`` (daq_type, module_name; str = 'DAQ0D/CameraStation') since 5.3."""
    plugin = getattr(det, 'detector', '')
    return str(getattr(plugin, 'module_name', plugin) or '')


def extract_2d(dte) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """First 2D array of a DataToExport, without copying, and its acquisition time (wall clock, set by
    the plugin when it created the data; None when missing or implausible)."""
    if dte is None:
        return None, None
    for dwa in dte:
        dim = getattr(getattr(dwa, 'dim', None), 'name', str(getattr(dwa, 'dim', '')))
        if dim == 'Data2D':
            data = getattr(dwa, 'data', None)
            if data:
                try:
                    ts = float(getattr(dwa, 'timestamp', math.nan))
                except (TypeError, ValueError):
                    ts = math.nan
                plausible = math.isfinite(ts) and -60.0 < time.time() - ts < 60.0
                return data[0], ts if plausible else None
    return None, None


def _fmt(v, nd=1) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '—'
    if not math.isfinite(v):
        return '—'
    if abs(v) >= 1e5:
        return f'{v:.3g}'
    return f'{v:.{nd}f}'


class StationLog:
    """Messages for the UI (bounded) and a rotating file; identical messages within 60 s are folded."""

    def __init__(self, folder: Path):
        self.lines: 'collections.deque[str]' = collections.deque(maxlen=500)
        self._new: 'collections.deque[str]' = collections.deque(maxlen=500)   # thread-safe appends
        self._last: Dict[str, float] = {}
        self.logger = logging.getLogger('camera_station')
        self.logger.setLevel(logging.INFO)
        if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in self.logger.handlers):
            try:
                h = logging.handlers.RotatingFileHandler(Path(folder) / 'camera_station.log', maxBytes=1_000_000,
                                                         backupCount=5, encoding='utf-8')
                h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
                self.logger.addHandler(h)
            except OSError:
                pass

    def add(self, msg: str, level: int = logging.INFO):
        now = time.monotonic()
        if now - self._last.get(msg, -1e9) < 60.0:
            return
        self._last[msg] = now
        if len(self._last) > 1000:
            self._last.clear()
        stamp = time.strftime('%H:%M:%S')
        mark = {logging.WARNING: '⚠ ', logging.ERROR: '✖ '}.get(level, '')
        self._new.append(f'{stamp}  {mark}{msg}')
        try:
            self.logger.log(level, msg)
        except Exception:
            pass

    def take_new(self) -> List[str]:
        out = []
        while self._new:
            line = self._new.popleft()
            self.lines.append(line)
            out.append(line)
        return out


class CameraStationApp(QtCore.QObject):
    def __init__(self, modules_manager, config, window=None, data_folder: Optional[Path] = None):
        super().__init__()
        self.modules_manager = modules_manager
        self.config = config
        self.window = window
        self.machine = machine_name(config)
        self.paths = resolve_paths(config, data_folder)
        self.log = StationLog(self.paths['logs'])
        self.store = SettingsStore(self.paths['data'] / f'station_{self.machine}.json')
        self.station: StationConfig = self.store.load()
        if self.store.load_error:
            self.log.add(self.store.load_error, logging.ERROR)
        self.engine = AnalysisEngine()
        self.engine.on_error = lambda cid, msg: self.log.add(f'{cid}: analysis error: {msg}', logging.ERROR)
        self.publisher = Publisher()
        self._relay_wanted = False          # the relay must grab (and forward to LECO): see _keep_relay
        self.writer: Optional[H5StationWriter] = None
        self.watcher = TriggerWatcher(self.station.trigger_folder)
        self._links: Dict[str, tuple] = {}               # cam id -> (viewer, slot)
        self._bursts: Dict[str, list] = {}               # cam id -> [remaining, event id]
        self._last_periodic = time.monotonic()
        self._cpu = (time.monotonic(), time.process_time(), 0.0)
        self._cpu_text = ''
        self._closing = False
        self._building = False
        self.params: Optional[Parameter] = None
        self.tree: Optional[ParameterTree] = None
        self.preview: Optional[PreviewWidget] = None
        self.trends: Optional[TrendsWidget] = None
        self.docks: Dict[str, gutils.Dock] = {}
        self.profiles = ProfileStore(self.paths['profiles'])
        self.panels: Dict[str, CameraPanel] = {}
        self._panels_open: set = set()                   # cam ids whose settings are shown
        self._panels_box: Optional[QtWidgets.QVBoxLayout] = None
        # results for the trends: (source, t_wall, values) appended by the worker threads
        self._trend_in: 'collections.deque[tuple]' = collections.deque(maxlen=20000)
        self._script_runs: Dict[str, int] = {}
        self._trend_keys: Optional[list] = None
        self._sized = False

        self.engine.set_cameras(self.station.cameras)
        for cam in self.station.cameras:
            self._load_images(cam)

        # user scripts: their own thread, fed after the built-in analysis of each frame
        self._install_example_scripts()
        self.library = ScriptLibrary(self.paths['scripts'])
        self.runner = ScriptRunner(self.library)
        for _, _, msg in self.library.scan():
            self.log.add(msg)
        for name in self.library.names():
            self.runner.params[name] = dict(self.station.script_setting(name)['params'])
        self.engine.after_analysis = self._after_analysis
        self.runner.start()
        self.engine.start()

        self.publish_timer = QtCore.QTimer(self)
        self.publish_timer.timeout.connect(self._publish)
        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self._refresh_ui)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.timeout.connect(self._refresh_preview)
        self.trigger_timer = QtCore.QTimer(self)
        self.trigger_timer.timeout.connect(self._poll_triggers)
        self.save_timer = QtCore.QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._save_settings)
        self.scripts_timer = QtCore.QTimer(self)
        self.scripts_timer.timeout.connect(self._scan_scripts)

    def _install_example_scripts(self):
        """First start: the template and the examples are copied into the (empty) scripts folder."""
        folder = self.paths['scripts']
        try:
            if any(folder.glob('*.py')):
                return
            for src in EXAMPLE_SCRIPTS.glob('*.py'):
                (folder / src.name).write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
            self.log.add(f'Example scripts copied into {folder}')
        except OSError as e:
            self.log.add(f'example scripts not copied: {e}', logging.WARNING)

    # ================================================================== UI construction
    def build(self, dockarea):
        self.tree = ParameterTree()
        self.tree.setMinimumWidth(320)
        self.docks['settings'] = gutils.Dock('Settings')
        self.docks['settings'].addWidget(self.tree)
        dockarea.addDock(self.docks['settings'], 'left')

        # cameras: the station status, then one compact panel per camera, like the modules of the dashboard
        cams = QtWidgets.QWidget()
        self._panels_box = QtWidgets.QVBoxLayout(cams)
        self._panels_box.setContentsMargins(2, 2, 2, 2)
        self._panels_box.setSpacing(4)
        self.status_label = QtWidgets.QLabel()      # one line per item, no word wrap: a stable height
        sf = self.status_label.font()
        sf.setBold(True)
        self.status_label.setFont(sf)
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.status_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._panels_box.addWidget(self.status_label)
        add = QtWidgets.QPushButton('＋ Add camera')
        add.setToolTip('Add a camera of this dashboard to the station')
        add.clicked.connect(lambda: self.add_camera())
        self._panels_box.addWidget(add)
        self._panels_box.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(cams)
        scroll.setMinimumWidth(420)
        self.docks['cameras'] = gutils.Dock('Cameras')
        self.docks['cameras'].addWidget(scroll)
        dockarea.addDock(self.docks['cameras'], 'right', self.docks['settings'])

        self.trends = TrendsWidget()
        self.docks['trends'] = gutils.Dock('Trends')
        self.docks['trends'].addWidget(self.trends)
        dockarea.addDock(self.docks['trends'], 'right', self.docks['cameras'])

        self.preview = PreviewWidget(self._roi_from_preview)
        self.docks['preview'] = gutils.Dock('Preview')
        self.docks['preview'].addWidget(self.preview)
        dockarea.addDock(self.docks['preview'], 'above', self.docks['trends'])   # tabbed with the trends

        self.scripts_table = QtWidgets.QTableWidget(0, len(SCRIPT_COLUMNS))
        self.scripts_table.setHorizontalHeaderLabels(SCRIPT_COLUMNS)
        self.scripts_table.verticalHeader().setVisible(False)
        self.scripts_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.scripts_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.scripts_table.horizontalHeader().setStretchLastSection(True)
        self.docks['scripts'] = gutils.Dock('Scripts')
        self.docks['scripts'].addWidget(self.scripts_table)
        dockarea.addDock(self.docks['scripts'], 'above', self.docks['trends'])

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.docks['log'] = gutils.Dock('Log')
        self.docks['log'].addWidget(self.log_view)
        dockarea.addDock(self.docks['log'], 'above', self.docks['trends'])
        self._raise('trends' if self.station.trends else 'log')

        self._create_params()
        self.connect_cameras()
        self._apply_timers()
        self.trigger_timer.start(1500)
        self.scripts_timer.start(2000)
        self.log.add(f'Camera Station started on {self.machine}; settings {self.store.path}')
        if config_get(self.config, 'recording', 'autostart', default=False):
            self.start_recording()
        self._refresh_ui()

    def _create_params(self):
        """(Re)creates the whole settings tree from ``self.station`` (start and profile loading)."""
        self.params = Parameter.create(name='settings', type='group',
                                       children=station_spec(self.station, self.profiles.names()))
        self._rebuild_cameras_group()
        self.params.child('scripts', 'scripts_folder').setValue(str(self.paths['scripts']))
        self._sync_scripts_tree()
        self._trend_keys = None
        self._sync_trends()
        self.params.sigTreeStateChanged.connect(self._tree_changed)
        self.params.child('scripts', 'reload_scripts').sigActivated.connect(lambda: self._scan_scripts(force=True))
        self.params.child('scripts', 'open_scripts_folder').sigActivated.connect(self._open_scripts_folder)
        self.params.child('profiles', 'load_profile').sigActivated.connect(lambda: self.load_profile())
        self.params.child('profiles', 'save_profile').sigActivated.connect(lambda: self.save_profile())
        self.params.child('profiles', 'delete_profile').sigActivated.connect(lambda: self.delete_profile())
        if self.tree is not None:
            self.tree.setParameters(self.params, showTop=False)

    def _initial_sizes(self):
        """Once, when the window is shown: settings narrow, cameras medium, the plots get the rest (the
        dock area splits evenly)."""
        if self._sized or not self.docks['settings'].isVisible():
            return
        self._sized = True
        try:
            split = self.docks['settings'].container()
            if isinstance(split, QtWidgets.QSplitter) and split.count() == 3:
                total = sum(split.sizes())
                split.setSizes([340, 540, max(total - 880, 400)])
        except Exception:
            pass

    def _raise(self, dock: str):
        if dock in self.docks:
            try:
                self.docks[dock].raiseDock()
            except Exception:
                pass

    def _apply_timers(self):
        self.publish_timer.start(max(int(1000 / self.station.publish_hz), 50))
        period = max(int(1000 / self.station.display_hz), 50)
        self.ui_timer.start(period)
        if self.trends is not None:
            self.trends.set_enabled(self.station.trends)
        if self.station.preview:
            self.preview_timer.start(period)
        else:
            self.preview_timer.stop()
            if self.preview is not None:
                self.preview.clear('Preview off (Settings > Station > Preview)')

    # ================================================================== cameras of the dashboard
    def available_detectors(self) -> List[str]:
        """Titles of the viewers that can provide 2D frames (the relay and LECO directors excluded)."""
        out = []
        for det in list(getattr(self.modules_manager, 'detectors_all', None) or
                        getattr(self.modules_manager, 'detectors', []) or []):
            try:
                daq_type = getattr(getattr(det, 'daq_type', None), 'name', '')
                if plugin_name(det) == RELAY_PLUGIN or (daq_type and daq_type not in ('DAQ2D', 'DAQND')):
                    continue
                out.append(str(det.title))
            except Exception:
                continue
        return out

    def _viewer(self, title: str):
        for det in list(getattr(self.modules_manager, 'detectors_all', None) or
                        getattr(self.modules_manager, 'detectors', []) or []):
            try:
                if str(det.title) == title:
                    return det
            except Exception:
                continue
        return None

    def relay_viewers(self) -> List:
        out = []
        for det in list(getattr(self.modules_manager, 'detectors_all', None) or
                        getattr(self.modules_manager, 'detectors', []) or []):
            try:
                if plugin_name(det) == RELAY_PLUGIN:
                    out.append(det)
            except Exception:
                continue
        return out

    def disconnect_cameras(self):
        for cam_id, (det, slot) in list(self._links.items()):
            try:
                det.grab_done_signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._links.clear()

    def connect_cameras(self):
        """(Re)connects every camera to its viewer; called at start, on changes and by 'Reload'."""
        self.disconnect_cameras()
        for cam in self.station.cameras:
            if not cam.detector:
                continue
            det = self._viewer(cam.detector)
            if det is None:
                self.log.add(f'{cam.name}: viewer "{cam.detector}" not in this dashboard', logging.WARNING)
                continue
            slot = self._make_slot(cam.id)
            det.grab_done_signal.connect(slot)
            self._links[cam.id] = (det, slot)
        if not self.relay_viewers():
            self.log.add('No "CameraStation" 0D viewer in the dashboard: the values are computed and '
                         'recorded but nothing is sent to computer A', logging.WARNING)

    def _make_slot(self, cam_id: str):
        def slot(dte):
            self._on_frame(cam_id, dte)
        return slot

    def refresh_hardware(self):
        self._rebuild_cameras_group()
        self.connect_cameras()
        self.log.add('Dashboard modules reloaded')

    def _start_viewer(self, det):
        try:
            if bool(getattr(det, 'grab_state', False)) or not getattr(det, 'initialized_state', True):
                return                                   # DAQ_Viewer.grab() toggles: never stop a running one
            if getattr(det, 'ui', None) is not None:
                det.grab()                               # keeps the viewer's Grab button in sync
            else:
                det.grab_data(True)
        except Exception as e:
            self.log.add(f'{getattr(det, "title", "?")}: grab command failed: {e}', logging.WARNING)

    @staticmethod
    def _leco_keyword(det) -> str:
        """Name of the 'forward the data to LECO' argument of DAQ_Viewer.grab: send_to_leco (5.3),
        send_to_tcpip (5.1); '' if none"""
        try:
            params = inspect.signature(det.grab_data).parameters
        except (TypeError, ValueError):
            return ''
        return next((k for k in ('send_to_leco', 'send_to_tcpip') if k in params), '')

    def _relay_forwards(self, det) -> bool:
        key = self._leco_keyword(det)
        return not key or bool(getattr(det, f'_{key}', True))

    def _start_relay(self, det):
        """Grab of the relay with its data forwarded to LECO: computer A receives the values whoever started
        the grab. A plain DAQ_Viewer grab (started from this station or the dashboard Grab button) never
        sends anything to LECO, and a Grab sent by computer A toggles a running grab off: so the relay is
        (re)started here with forwarding on, and kept running by the UI timer (_keep_relay)."""
        self._relay_wanted = True
        try:
            if not getattr(det, 'initialized_state', True):
                return
            grabbing = bool(getattr(det, 'grab_state', False))
            if grabbing and self._relay_forwards(det):
                return
            if grabbing:
                det.stop_grab()                          # running without forwarding: restarted with it
            key = self._leco_keyword(det)
            kwargs = {key: True} if key else {}
            if getattr(det, 'ui', None) is not None:
                det.grab(**kwargs)                       # keeps the viewer's Grab button in sync
            else:
                det.grab_data(True, **kwargs)
        except Exception as e:
            self.log.add(f'{getattr(det, "title", "?")}: relay grab failed: {e}', logging.WARNING)

    def _keep_relay(self):
        """The relay stopped while the station publishes (e.g. a Grab of computer A toggled it off, or it was
        started without LECO forwarding, e.g. by the Grab button of the dashboard): restart it"""
        for det in self.relay_viewers():
            if not getattr(det, 'initialized_state', True):
                continue
            grabbing = bool(getattr(det, 'grab_state', False))
            if (grabbing and not self._relay_forwards(det)) or (not grabbing and self._relay_wanted):
                self.log.add(f'{getattr(det, "title", "relay")}: relay (re)started with LECO forwarding')
                self._start_relay(det)

    def _stop_viewer(self, det):
        try:
            if bool(getattr(det, 'grab_state', False)):
                det.stop_grab()
        except Exception as e:
            self.log.add(f'{getattr(det, "title", "?")}: stop command failed: {e}', logging.WARNING)

    def _camera_viewer(self, cam_id: str):
        link = self._links.get(cam_id)
        if link is None:
            cam = self.station.camera(cam_id)
            self.log.add(f'{cam.name if cam else cam_id}: no camera viewer connected (Settings of the camera '
                         f'> Camera)', logging.WARNING)
            return None
        return link[0]

    def grab_all(self, start: bool = True):
        """Starts (or stops) the grab of the camera viewers used by the station and of the relay."""
        if not start:
            self._relay_wanted = False
        for det in [self._links[c][0] for c in self._links]:
            if start:
                self._start_viewer(det)
            else:
                self._stop_viewer(det)
        for det in self.relay_viewers():
            if start:
                self._start_relay(det)
            else:
                self._stop_viewer(det)

    def grab_camera(self, cam_id: str, start: bool = True):
        """Continuous grab of one camera. Starting a camera also starts the relay (the bridge to A);
        stopping one leaves the relay running (the other cameras, the heartbeat)."""
        det = self._camera_viewer(cam_id)
        if det is None:
            return
        if start:
            self._start_viewer(det)
            for relay in self.relay_viewers():
                self._start_relay(relay)
        else:
            self._stop_viewer(det)

    def snap_camera(self, cam_id: str):
        """One frame of this camera, analysed whatever the rate gate (a grabbing camera: its next frame)."""
        det = self._camera_viewer(cam_id)
        if det is None:
            return
        self.engine.force_next(cam_id)
        if not bool(getattr(det, 'grab_state', False)):
            try:
                det.snap()
            except Exception as e:
                self.log.add(f'{getattr(det, "title", "?")}: snap failed: {e}', logging.WARNING)
        for relay in self.relay_viewers():
            self._start_relay(relay)

    def preview_camera(self, cam_id: str):
        if self.params is None:
            return
        self.params.child('station', 'preview_group', 'preview_camera').setValue(cam_id)
        if not self.station.preview:
            self.params.child('station', 'preview_group', 'preview').setValue(True)
        self._raise('preview')
        self._refresh_preview()

    # ================================================================== frames (GUI thread)
    def _on_frame(self, cam_id: str, dte):
        if self._closing:
            return
        try:
            img, t_acq = extract_2d(dte)
            if img is None:
                self.log.add(f'{cam_id}: the viewer sends no 2D data', logging.WARNING)
                return
            accepted = self.engine.submit(cam_id, img, t_wall=t_acq, acquired=t_acq is not None)
            burst = self._bursts.get(cam_id)
            if burst and accepted:
                cam = self.station.camera(cam_id)
                if cam is not None:
                    self._save_image(cam, IMG_EVENT, burst[1])
                burst[0] -= 1
                if burst[0] <= 0:
                    del self._bursts[cam_id]
        except Exception as e:
            self.log.add(f'{cam_id}: frame not processed: {e}', logging.ERROR)

    def _after_analysis(self, rt, item, res):
        """Worker thread, after each built-in analysis: every result goes to the trends (deque: thread
        safe, bounded), then the user scripts of the camera run on the same frame."""
        self._trend_in.append((rt.cfg.name, item[2], res))
        self.runner.submit(rt.cfg, rt.analyzer, item[0], item[1], item[2], res)

    # ================================================================== publication (timer)
    def _publish(self):
        if self._closing:
            return
        try:
            cams = [(c.name, c.labels(), self.engine.snapshot(c.id)) for c in self.station.cameras if c.enabled]
            scripts = self._script_entries()
            t = time.time()
            payload = self.publisher.build(cams, t, self.station.stale_s, recording=self.recording,
                                           extra=[(n, lab, snap) for n, lab, snap, _ in scripts])
            hidden = {n for n, _, _, pub in scripts if not pub}       # script results under test: not sent to A
            bus.publish({k: v for k, v in payload.items() if k not in hidden} if hidden else payload)
            if self.writer is not None:                               # ... but always recorded
                self.writer.write_row(t, self.publisher.counter, flatten(payload))
                period = self.station.image_period_s
                now = time.monotonic()
                if period > 0 and now - self._last_periodic >= period:
                    self._last_periodic = now
                    for cam in self.station.cameras:
                        if cam.enabled and cam.record_images and self.engine.state(cam.id, self.station.stale_s) \
                                not in (ST_OFF, ST_NO_FRAMES, ST_WAITING):     # never a frozen frame as "periodic"
                            self._save_image(cam, IMG_PERIODIC)
        except Exception as e:
            self.log.add(f'publication failed: {e}', logging.ERROR)

    # ================================================================== recording
    @property
    def recording(self) -> bool:
        return self.writer is not None

    def start_recording(self) -> bool:
        if self.writer is not None:
            return True
        try:
            now = _dt.datetime.now()
            folder = self.paths['recordings'] / now.strftime('%Y-%m-%d')
            base = f'station_{self.machine}_{now.strftime("%Y%m%d_%H%M%S")}'
            from .. import __version__
            meta = {'machine': self.machine, 'station_json': self.station.to_dict(), 'code_version': __version__,
                    'publish_hz': self.station.publish_hz,
                    'sync': 'join with Laser Control on <viewer>/Stats/frame_counter (= scalars/frame_counter)'}
            c = self.config
            self.writer = H5StationWriter(
                folder, base, meta,
                max_mb=float(config_get(c, 'recording', 'max_file_mb', default=1000)),
                max_hours=float(config_get(c, 'recording', 'max_file_hours', default=12)),
                max_queue_mb=float(config_get(c, 'recording', 'max_queue_mb', default=200)),
                compression=int(config_get(c, 'recording', 'compression', default=4)))
            self.writer.start()
            self.writer.write_event(time.time(), 'recording', 'recording started')
            self._last_periodic = -1e9          # first images right away
            self.log.add(f'Recording started: {folder / base}.h5')
            return True
        except Exception as e:
            self.writer = None
            self.log.add(f'recording not started: {e}', logging.ERROR)
            return False

    def stop_recording(self):
        w, self.writer = self.writer, None
        if w is None:
            return
        try:
            w.write_event(time.time(), 'recording', 'recording stopped')
            w.close()
            self.log.add(f'Recording stopped: {w.n_rows} rows, {w.n_images} images'
                         + (f', {w.n_dropped_images} images dropped' if w.n_dropped_images else ''))
        except Exception as e:
            self.log.add(f'recording stop: {e}', logging.ERROR)

    def toggle_recording(self, on: bool):
        if on:
            self.start_recording()
        else:
            self.stop_recording()

    def _save_image(self, cam: CameraConfig, kind: int, event_id: int = -1) -> bool:
        if self.writer is None:
            return False
        rt = self.engine.runtime(cam.id)
        if rt is None or rt.latest is None:
            return False
        img, frame_no, t_wall = rt.latest
        try:
            a = np.asarray(img)
            roi = (0, 0, a.shape[1], a.shape[0]) if self.station.image_full_frame else clip_roi(cam.roi, a.shape[:2])
            q, off, scale = reduce_image(a, None if self.station.image_full_frame else roi,
                                         self.station.image_binning, self.station.image_bits)
            return self.writer.write_image(cam.id, cam.name, q, off, scale, t_wall, frame_no, self.publisher.counter,
                                           kind, event_id, roi, self.station.image_binning)
        except Exception as e:
            self.log.add(f'{cam.name}: image not recorded: {e}', logging.ERROR)
            return False

    def add_note(self, text: str):
        if not text:
            return
        if self.writer is not None:
            self.writer.write_event(time.time(), 'note', text, source='operator')
        self.log.add(f'Note: {text}' + ('' if self.writer is not None else ' (not recording: log only)'))

    def snapshot(self):
        if self.writer is None:
            self.log.add('Snapshot: start the recording first', logging.WARNING)
            return
        n = sum(self._save_image(c, IMG_MANUAL) for c in self.station.cameras if c.enabled)
        self.writer.write_event(time.time(), 'snapshot', f'{n} images', source='operator')
        self.log.add(f'Snapshot: {n} images recorded')

    # ================================================================== triggers of computer A (O8)
    def _poll_triggers(self):
        if self._closing:
            return
        try:
            events = self.watcher.poll()
            for ev in events:
                eid = int(ev.get('id', -1)) if str(ev.get('id', '')).lstrip('-').isdigit() else -1
                msg = f"{ev.get('kind', '?')}: {ev.get('message', '')}".strip()
                self.log.add(f'Event from Laser Control: {msg}')
                if self.writer is not None:
                    self.writer.write_event(time.time(), 'trigger', msg, source='LaserControl', event_id=eid, data=ev)
                    for cam in self.station.cameras:
                        if cam.enabled and cam.record_images:
                            self._save_image(cam, IMG_EVENT, eid)
                            if self.station.burst_frames > 1:
                                self._bursts[cam.id] = [self.station.burst_frames - 1, eid]
        except Exception as e:
            self.log.add(f'trigger folder: {e}', logging.ERROR)

    # ================================================================== user scripts
    def _script_status(self, name: str) -> str:
        s = self.library.scripts.get(name)
        if s is None:
            return ''
        if name in self.runner.disabled:
            return f'■ {self.runner.disabled[name]}'
        if not s.ok:
            return f'✖ {s.error}'
        if s.error:
            return f'⚠ {s.error}'
        return f'✔ version {s.version}: {", ".join(s.labels)}'

    def _scan_scripts(self, force: bool = False):
        if self._closing:
            return
        try:
            msg = self.runner.watchdog(self.station.script_timeout_s)
            if msg:
                self.log.add(msg, logging.ERROR)
            changed = self.library.scan(force=force)
            if self.library.folder_error:
                self.log.add(self.library.folder_error, logging.WARNING)
            for level, name, text in changed:
                self.log.add(text, {'error': logging.ERROR, 'warning': logging.WARNING}.get(level, logging.INFO))
                if level == 'loaded':
                    self.runner.script_reloaded(name)
                    if self.writer is not None:
                        self.writer.write_event(time.time(), 'script', text, data={'script': name})
            for name in self.library.names():
                self.runner.params.setdefault(name, dict(self.station.script_setting(name)['params']))
            if changed or force or msg:
                self._sync_scripts_tree()
        except Exception as e:
            self.log.add(f'scripts: {e}', logging.ERROR)

    def _open_scripts_folder(self):
        try:
            import os
            os.startfile(str(self.paths['scripts']))         # Windows explorer
        except Exception as e:
            self.log.add(f'folder not opened ({e}): {self.paths["scripts"]}', logging.WARNING)

    def _sync_scripts_tree(self):
        """Adds / removes the script groups and the camera check boxes, without rebuilding the rest of the
        tree (an edit of a script must not collapse what the user has opened)."""
        if self.params is None:
            return
        self._building = True
        try:
            names = self.library.names()
            group = self.params.child('scripts')
            for child in list(group.children()):
                if child.name().startswith(SCRIPT_PREFIX) and child.name()[len(SCRIPT_PREFIX):] not in names:
                    group.removeChild(child)
            for name in names:
                script = self.library.scripts[name]
                setting = self.station.script_setting(name)
                pname = SCRIPT_PREFIX + name
                existing = [c for c in group.children() if c.name() == pname]
                if existing:
                    g = existing[0]
                    g.child('status').setValue(self._script_status(name))
                    pg = g.child('params')
                    if [c.name() for c in pg.children()] != list(script.defaults):   # PARAMS changed in the file
                        pg.clearChildren()
                        current = self.runner.params.get(name, {})
                        for k, v in script.defaults.items():
                            val = current.get(k, v)
                            pg.addChild(script_param_spec(k, val if type(val) is type(v) else v))
                    g.setOpts(tip=script.doc)
                else:
                    group.addChild(script_spec(name, script, setting, self._script_status(name)))
            for cam in self.station.cameras:
                try:
                    sg = self.params.child('cameras', cam.id, 'scripts')
                except Exception:
                    continue
                if [c.name() for c in sg.children()] != names:
                    sg.clearChildren()
                    for spec in camera_scripts_children(cam, names):
                        sg.addChild(spec)
        finally:
            self._building = False

    def _script_entries(self):
        """(publication name, labels, snapshot, published) of every script enabled on an enabled camera."""
        out = []
        for cam in self.station.cameras:
            if not cam.enabled:
                continue
            for name in cam.scripts:
                script = self.library.scripts.get(name)
                if script is None or not script.ok:
                    continue
                pub = clean_name(f'{cam.name}_{name}')
                out.append((pub, tuple(script.labels) + ('valid',), self.runner.snapshot(cam.id, name),
                            bool(self.station.script_setting(name)['publish'])))
        return out

    # ================================================================== settings tree
    def _rebuild_cameras_group(self):
        if self.params is None:
            return
        self._building = True
        try:
            group = self.params.child('cameras')
            group.clearChildren()
            dets = self.available_detectors()
            for cam in self.station.cameras:
                p = group.addChild(camera_spec(cam, dets, self.library.names()))
                self._wire_camera_actions(p, cam.id)
            pc = self.params.child('station', 'preview_group', 'preview_camera')
            ids = [c.id for c in self.station.cameras] or ['']
            cur = pc.value()
            pc.setLimits(ids)
            pc.setValue(cur if cur in ids else ids[0])
        finally:
            self._building = False
        self._rebuild_panels()

    def _rebuild_panels(self):
        """One panel per camera, on the camera group of the settings (rebuilt with it)."""
        if self._panels_box is None:
            return
        for panel in self.panels.values():
            self._panels_box.removeWidget(panel)
            panel.setParent(None)
            panel.deleteLater()
        self.panels.clear()
        for i, cam in enumerate(self.station.cameras):
            panel = CameraPanel(self.params.child('cameras', cam.id), settings_open=cam.id in self._panels_open)
            cid = cam.id
            panel.snap_clicked.connect(lambda cid=cid: self.snap_camera(cid))
            panel.grab_toggled.connect(lambda on, cid=cid: self.grab_camera(cid, on))
            panel.stop_clicked.connect(lambda cid=cid: self.grab_camera(cid, False))
            panel.preview_clicked.connect(lambda cid=cid: self.preview_camera(cid))
            panel.settings_toggled.connect(lambda on, cid=cid: self._panel_settings_shown(cid, on))
            self._panels_box.insertWidget(1 + i, panel)          # after the station status line
            self.panels[cam.id] = panel
        self._refresh_panels()

    def _panel_settings_shown(self, cam_id: str, shown: bool):
        if shown:
            self._panels_open.add(cam_id)
        else:
            self._panels_open.discard(cam_id)

    # ================================================================== trends
    def _trend_specs(self):
        specs = []
        for cam in self.station.cameras:
            if cam.enabled:
                specs += camera_specs(cam.name, cam.labels())
        for pub, labels, _, _ in self._script_entries():
            specs += script_specs(pub, labels)
        return specs

    def _sync_trends(self):
        """Plot list of the settings and plots of the Trends dock, when the cameras / scripts changed."""
        if self.params is None:
            return
        specs = self._trend_specs()
        keys = [s.key for s in specs]
        if keys != self._trend_keys:
            self._trend_keys = keys
            self._building = True
            try:
                group = self.params.child('trends_group', 'plots')
                group.clearChildren()
                for s in specs:
                    group.addChild(trend_plot_spec(s.key, s.title, self._trend_shown(s)))
            finally:
                self._building = False
        if self.trends is not None:
            self.trends.set_plots([s for s in specs if self._trend_shown(s)])

    def _trend_shown(self, spec) -> bool:
        return self.station.trend_choices.get(spec.key, spec.default)

    def _refresh_trends(self):
        n = len(self._trend_in)
        for _ in range(n):
            source, t, res = self._trend_in.popleft()
            if self.trends is not None:
                self.trends.add(source, t, res)
        for pub, _, snap, _ in self._script_entries():
            runs = snap.get('runs', 0)
            if snap and runs != self._script_runs.get(pub) and self.trends is not None:
                self._script_runs[pub] = runs
                self.trends.add(pub, snap.get('t_wall', time.time()), snap.get('result', {}))
        self._sync_trends()
        if self.trends is not None and self.station.trends:
            dock = self.docks.get('trends')
            if dock is None or dock.isVisible():
                self.trends.redraw(self.station.trend_history_s)

    def _wire_camera_actions(self, p: Parameter, cam_id: str):
        p.child('remove').sigActivated.connect(lambda: self.remove_camera(cam_id))
        p.child('roi', 'roi_full').sigActivated.connect(lambda: self.set_roi(cam_id, [0, 0, 0, 0]))
        p.child('disk', 'capture_ref').sigActivated.connect(lambda: self.capture(cam_id, 'ref'))
        p.child('disk', 'clear_ref').sigActivated.connect(lambda: self.clear_image(cam_id, 'ref'))
        p.child('dark', 'capture_dark').sigActivated.connect(lambda: self.capture(cam_id, 'dark'))
        p.child('dark', 'clear_dark').sigActivated.connect(lambda: self.clear_image(cam_id, 'dark'))

    def _camera_of(self, param: Parameter) -> Optional[str]:
        group = self.params.child('cameras')
        p = param
        while p is not None:
            if p.parent() is group:
                return p.name()
            p = p.parent()
        return None

    def _tree_changed(self, root, changes):
        if self._building:
            return
        for param, change, data in changes:
            if change != 'value':
                continue
            try:
                self._param_changed(param, data)
            except Exception as e:
                self.log.add(f'setting {param.name()} not applied: {e}', logging.ERROR)

    def _script_of(self, param: Parameter) -> Optional[str]:
        p = param
        while p is not None:
            if p.name().startswith(SCRIPT_PREFIX) and p.parent() is not None and p.parent().name() == 'scripts':
                return p.name()[len(SCRIPT_PREFIX):]
            p = p.parent()
        return None

    def _param_changed(self, param: Parameter, value):
        name = param.name()
        cam_id = self._camera_of(param)
        st = self.station
        script = self._script_of(param) if cam_id is None else None
        if script is not None:                       # live parameter / publication of a script
            setting = st.script_setting(script)
            if name == 'publish':
                setting['publish'] = bool(value)
                self.log.add(f'script {script}: {"published to computer A" if value else "not sent to A"}')
            elif param.parent() is not None and param.parent().name() == 'params':
                setting['params'][name] = value
                self.runner.params.setdefault(script, {})[name] = value
            self._settings_changed()
            return
        if cam_id is not None and param.parent() is not None and param.parent().name() == 'scripts':
            cam = st.camera(cam_id)                  # script ticked / unticked on a camera
            if cam is not None:
                if value and name not in cam.scripts:
                    cam.scripts.append(name)
                elif not value and name in cam.scripts:
                    cam.scripts.remove(name)
                self._settings_changed()
            return
        if cam_id is None and param.parent() is not None and param.parent().name() == 'plots':
            st.trend_choices[param.opts.get('trend_key', '')] = bool(value)   # a trend plot ticked / unticked
            self._sync_trends()
            self._settings_changed()
            return
        if cam_id is None:
            if name in STATION_KEYS:
                setattr(st, name, value)
                st.validate()
                if name in ('publish_hz', 'preview', 'display_hz', 'trends'):
                    self._apply_timers()
                if name == 'preview' and value:
                    self._raise('preview')
                if name == 'trends':
                    self._raise('trends' if value else 'log')
                if name == 'trigger_folder':
                    self.watcher = TriggerWatcher(st.trigger_folder)
                    self.log.add(f'Trigger folder: {st.trigger_folder or "off"}')
            elif name in ('preview_camera', 'preview_view'):
                pass
            else:
                return
            self._settings_changed()
            return
        cam = st.camera(cam_id)
        if cam is None:
            return
        if name in ('x0', 'y0', 'w', 'h'):
            i = ('x0', 'y0', 'w', 'h').index(name)
            cam.roi[i] = int(value)
        elif name in CAMERA_KEYS:
            if name == 'detector':
                value = '' if value == NO_DETECTOR else value
            old_name = cam.name
            setattr(cam, name, value)
            if name == 'name':
                st.validate()
                if cam.name != value:
                    self._set_quiet(param, cam.name)
                param.parent().setOpts(title=cam.name)
                self._update_preview_limits()
                self.log.add(f'{old_name} renamed {cam.name}: the keys received by computer A change '
                             f'("…/{cam.name}/x"), update its roles', logging.WARNING)
            if name == 'kind':
                self.log.add(f'{cam.name}: analysis {value}; published labels: {", ".join(cam.labels())}')
        else:
            return
        cam.validate()
        if name == 'detector':
            self.connect_cameras()
        self.engine.set_cameras(st.cameras)
        self._settings_changed()

    def _set_quiet(self, param: Parameter, value):
        self._building = True
        try:
            param.setValue(value)
        finally:
            self._building = False

    def _update_preview_limits(self):
        pc = self.params.child('station', 'preview_group', 'preview_camera')
        ids = [c.id for c in self.station.cameras] or ['']
        cur = pc.value()
        self._building = True
        try:
            pc.setLimits(ids)
            pc.setValue(cur if cur in ids else ids[0])
        finally:
            self._building = False

    def _settings_changed(self):
        self.save_timer.start(800)

    def _save_settings(self):
        try:
            self.store.save(self.station)
        except Exception as e:
            self.log.add(f'settings not saved: {e}', logging.ERROR)
            return
        if self.writer is not None:        # the recording knows every setting change
            self.writer.write_event(time.time(), 'settings', 'settings changed', data=self.station.to_dict())
            self.writer.update_metadata({'station_json': self.station.to_dict()})

    # ================================================================== cameras: add / remove / ROI
    def add_camera(self, detector: str = '') -> CameraConfig:
        dets = self.available_detectors()
        used = {c.detector for c in self.station.cameras}
        if not detector:
            free = [d for d in dets if d not in used]
            detector = free[0] if free else ''
        cam = self.station.new_camera(detector)
        self.station.validate()
        self.engine.set_cameras(self.station.cameras)
        self._rebuild_cameras_group()
        self.connect_cameras()
        self._settings_changed()
        self.log.add(f'Camera {cam.name} added' + (f' on "{detector}"' if detector else ''))
        return cam

    def remove_camera(self, cam_id: str):
        cam = self.station.camera(cam_id)
        if cam is None:
            return
        if self.window is not None and not self._confirm(f'Remove camera {cam.name}? Computer A will stop '
                                                         f'receiving "{cam.name}/…".'):
            return
        self.station.cameras = [c for c in self.station.cameras if c.id != cam_id]
        self._bursts.pop(cam_id, None)
        self.engine.set_cameras(self.station.cameras)
        self.connect_cameras()
        QtCore.QTimer.singleShot(0, self._rebuild_cameras_group)   # not while the tree emits the click
        self._settings_changed()
        self.log.add(f'Camera {cam.name} removed')

    def _confirm(self, text: str) -> bool:
        r = QtWidgets.QMessageBox.question(self.window, 'Camera Station', text,
                                           QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                           QtWidgets.QMessageBox.No)
        return r == QtWidgets.QMessageBox.Yes

    def set_roi(self, cam_id: str, roi: List[int]):
        cam = self.station.camera(cam_id)
        if cam is None:
            return
        cam.roi = [int(v) for v in roi]
        cam.validate()
        group = self.params.child('cameras', cam_id, 'roi')
        self._building = True
        try:
            for i, k in enumerate(('x0', 'y0', 'w', 'h')):
                group.child(k).setValue(cam.roi[i])
        finally:
            self._building = False
        self.engine.set_cameras(self.station.cameras)
        self._settings_changed()

    def _roi_from_preview(self, roi: List[int]):
        cam_id = self.params.child('station', 'preview_group', 'preview_camera').value()
        if self.station.camera(cam_id) is not None:
            self.set_roi(cam_id, roi)

    # ================================================================== settings profiles
    def _selected_profile(self) -> str:
        return str(self.params.child('profiles', 'profile').value() or '') if self.params else ''

    def _update_profile_list(self, select: str = ''):
        p = self.params.child('profiles', 'profile')
        names = self.profiles.names()
        self._building = True
        try:
            p.setLimits(names or [''])
            p.setValue(select if select in names else (names[0] if names else ''))
        finally:
            self._building = False

    def save_profile(self, name: str = '') -> bool:
        """Saves the current settings as a profile (``name`` or the 'New name' field, else the selected one)."""
        if not name and self.params is not None:
            name = str(self.params.child('profiles', 'profile_name').value() or '').strip() or \
                self._selected_profile()
        if not name:
            self.log.add('Profile not saved: type a name in Profiles > New name', logging.WARNING)
            return False
        if self.window is not None and self.profiles.path(name).exists() and \
                not self._confirm(f'Overwrite the profile "{name}"?'):
            return False
        try:
            saved = self.profiles.save(name, self.station)
        except Exception as e:
            self.log.add(f'profile "{name}" not saved: {e}', logging.ERROR)
            return False
        if self.params is not None:
            self._set_quiet(self.params.child('profiles', 'profile_name'), '')
            self._update_profile_list(saved)
        self.log.add(f'Profile "{saved}" saved')
        return True

    def load_profile(self, name: str = '') -> bool:
        """Replaces the current settings by a profile (the cameras, scripts and display are rebuilt)."""
        name = name or self._selected_profile()
        if not name:
            self.log.add('No profile to load: save one first', logging.WARNING)
            return False
        if self.window is not None and not self._confirm(
                f'Load the profile "{name}"? The current settings are replaced (save them as a profile '
                f'first to keep them).'):
            return False
        st, err = self.profiles.load(name)
        if st is None:
            self.log.add(err, logging.ERROR)
            return False
        old_names = {c.name for c in self.station.cameras}
        self._apply_station(st)
        self.log.add(f'Profile "{name}" loaded')
        if {c.name for c in st.cameras} != old_names:
            self.log.add('The camera names changed: the keys received by computer A change, update its roles',
                         logging.WARNING)
        if self.writer is not None:
            self.writer.write_event(time.time(), 'settings', f'profile {name} loaded')
        self._update_profile_list(name)
        return True

    def delete_profile(self, name: str = '') -> bool:
        name = name or self._selected_profile()
        if not name:
            return False
        if self.window is not None and not self._confirm(f'Delete the profile "{name}"?'):
            return False
        ok = self.profiles.delete(name)
        self.log.add(f'Profile "{name}" deleted' if ok else f'Profile "{name}" not deleted',
                     logging.INFO if ok else logging.WARNING)
        self._update_profile_list()
        return ok

    def _apply_station(self, st: StationConfig):
        self.station = st
        self._bursts.clear()
        self.engine.set_cameras(st.cameras)
        for cam in st.cameras:
            self._load_images(cam)
        for name in self.library.names():
            self.runner.params[name] = dict(st.script_setting(name)['params'])
        self.watcher = TriggerWatcher(st.trigger_folder)
        self._create_params()
        self.connect_cameras()
        self._apply_timers()
        self._settings_changed()

    # ================================================================== reference / dark frames
    def _image_path(self, name: str) -> Path:
        return self.paths['references'] / name

    def _load_images(self, cam: CameraConfig):
        """Reference and dark frames of a camera from the references folder (engine runtime must exist)."""
        rt = self.engine.runtime(cam.id)
        if rt is None:
            return
        for what, fname in (('ref', cam.reference_file), ('dark', cam.dark_file)):
            arr = None
            if fname:
                try:
                    arr = np.load(self._image_path(fname), allow_pickle=False)
                except Exception as e:
                    self.log.add(f'{cam.name}: {what} image {fname} not loaded: {e}', logging.WARNING)
            if what == 'ref':
                rt.analyzer.set_reference(arr)
            else:
                rt.analyzer.set_dark(arr)

    def capture(self, cam_id: str, what: str):
        cam = self.station.camera(cam_id)
        if cam is None:
            return
        n = int(self.params.child('cameras', cam_id, 'disk', 'n_capture').value()) if self.params else 10
        if cam_id not in self._links:
            self.log.add(f'{cam.name}: no camera connected, nothing to capture', logging.WARNING)
            return
        label = 'reference' if what == 'ref' else 'dark frame'
        self.log.add(f'{cam.name}: capturing the {label} ({n} frames)…')
        self.engine.start_capture(cam_id, what, n, lambda cid, img: self._captured(cid, what, img))

    def _captured(self, cam_id: str, what: str, img: np.ndarray):
        cam = self.station.camera(cam_id)
        if cam is None:
            return
        stamp = time.strftime('%Y%m%d_%H%M%S')
        fname = f'{cam.id}_{what}_{stamp}.npy'
        try:
            np.save(self._image_path(fname), img)
        except Exception as e:
            self.log.add(f'{cam.name}: {what} not saved: {e}', logging.ERROR)
            return
        rt = self.engine.runtime(cam_id)
        key = 'reference_file' if what == 'ref' else 'dark_file'
        setattr(cam, key, fname)
        if what == 'ref':
            rt.analyzer.set_reference(img)
            pname = ('disk', 'reference_file')
        else:
            rt.analyzer.set_dark(img)
            pname = ('dark', 'dark_file')
        self._set_quiet(self.params.child('cameras', cam_id, *pname), fname)
        self._settings_changed()
        if self.writer is not None:
            self.writer.write_event(time.time(), 'reference', f'{cam.name}: new {what} {fname}')
        self.log.add(f'{cam.name}: {what} saved ({fname}, mean {float(np.mean(img)):.1f} counts)')

    def clear_image(self, cam_id: str, what: str):
        cam = self.station.camera(cam_id)
        rt = self.engine.runtime(cam_id)
        if cam is None or rt is None:
            return
        if what == 'ref':
            cam.reference_file = ''
            rt.analyzer.set_reference(None)
            self._set_quiet(self.params.child('cameras', cam_id, 'disk', 'reference_file'), '')
        else:
            cam.dark_file = ''
            rt.analyzer.set_dark(None)
            self._set_quiet(self.params.child('cameras', cam_id, 'dark', 'dark_file'), '')
        self._settings_changed()
        self.log.add(f'{cam.name}: {what} cleared')

    # ================================================================== periodic UI
    def _cpu_load(self) -> str:
        now, pt = time.monotonic(), time.process_time()
        t0, p0, busy0 = self._cpu
        if now - t0 >= 2.0:
            busy = self.engine.busy_s
            proc = 100.0 * (pt - p0) / (now - t0)
            ana = 100.0 * (busy - busy0) / (now - t0)
            self._cpu = (now, pt, busy)
            self._cpu_text = f'CPU process {proc:.0f} % of one core (analysis {ana:.0f} %)'
        return self._cpu_text

    def _refresh_ui(self):
        if self._closing or self.params is None:
            return
        try:
            self._keep_relay()
            for line in self.log.take_new():
                self.log_view.appendPlainText(line)
            self._initial_sizes()
            self._refresh_panels()
            self._refresh_scripts_table()
            text = self._status_text()
            if self.status_label.text() != text:
                self.status_label.setText(text)
                self.status_label.setToolTip(text)
        except Exception as e:
            self.log.add(f'status refresh: {e}', logging.ERROR)
        try:
            self._refresh_trends()
        except Exception as e:
            self.log.add(f'trends: {e}', logging.ERROR)

    def _refresh_panels(self):
        for cam in self.station.cameras:
            panel = self.panels.get(cam.id)
            if panel is None:
                continue
            snap = self.engine.snapshot(cam.id)
            state = self.engine.state(cam.id, self.station.stale_s)
            res = snap.get('result', {})
            info = snap.get('error') or snap.get('reason') or ''
            det = self._links.get(cam.id, (None,))[0]
            if not cam.detector:
                info = 'no camera viewer selected (⚙ > Camera)'
            elif det is None:
                info = f'viewer "{cam.detector}" not found in the dashboard'
            elif not cam.enabled:
                info = 'analysis disabled (⚙ > Enabled)'
            grabbing = None if det is None else bool(getattr(det, 'grab_state', False))
            if grabbing is False and state != ST_OK and not info:
                info = 'camera stopped: Grab (green triangle) or Snap (one frame)'
            rates = (f'camera {_fmt(snap.get("in_rate"))} Hz → analysed {_fmt(snap.get("done_rate"))} Hz '
                     f'(max {cam.rate_hz:g}) · {_fmt(snap.get("calc_ms"))} ms')
            values = '  '.join(f'{VALUE_NAMES.get(lab, lab)} '
                               f'{_fmt(100 * res.get(lab, math.nan), 2) if lab == "sat" else _fmt(res.get(lab), 1)}'
                               for lab in cam.labels() if lab != 'valid')
            panel.update_view(f'{cam.name}  ({cam.detector or "no camera"})', STATE_TEXT.get(state, state),
                              state == ST_OK, grabbing, rates, values, info)

    def _refresh_scripts_table(self):
        rows = []
        for cam in self.station.cameras:
            if not cam.enabled:
                continue
            for name in cam.scripts:
                s = self.library.scripts.get(name)
                snap = self.runner.snapshot(cam.id, name)
                res = snap.get('result', {})
                if s is None:
                    state = '✖ file missing'
                elif name in self.runner.disabled:
                    state = '■ disabled'
                elif not s.ok:
                    state = '✖ not loaded'
                elif snap.get('error'):
                    state = '✖ ' + snap['error']
                elif not snap:
                    state = '… waiting'
                elif snap.get('age', math.inf) > self.station.stale_s:
                    state = '⚠ no frames'
                else:
                    state = '▶ ok' if res.get('valid') == 1.0 else '⚠ invalid'
                if s is not None and s.ok and self.station.script_setting(name)['publish']:
                    state += ' → A'
                values = '  '.join(f'{k}={_fmt(v, 3)}' for k, v in res.items() if k != 'valid')
                rows.append((name, cam.name, state, _fmt(snap.get('ms')), str(snap.get('runs', 0)), values))
        if self.scripts_table.rowCount() != len(rows):
            self.scripts_table.setRowCount(len(rows))
        for r, vals in enumerate(rows):
            for c, text in enumerate(vals):
                item = self.scripts_table.item(r, c)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    self.scripts_table.setItem(r, c, item)
                if item.text() != text:
                    item.setText(text)
                    item.setToolTip(text)

    def _status_text(self) -> str:
        parts = []
        relays = self.relay_viewers()
        parts.append(f'▶ Publishing {self.station.publish_hz:g} Hz · heartbeat {self.publisher.counter}'
                     if relays else '⚠ No CameraStation 0D viewer: nothing sent to computer A')
        w = self.writer
        if w is not None:
            size = ''
            try:
                if w.current_path is not None and w.current_path.exists():
                    size = f' {w.current_path.stat().st_size / 1e6:.1f} MB'
            except OSError:
                pass
            rec = f'● REC {w.current_path.name if w.current_path else ""}{size} · {w.n_images} img'
            if w.n_dropped_images or w.n_dropped_rows:
                rec += f' · ⚠ {w.n_dropped_images} img / {w.n_dropped_rows} rows lost'
            if w.error:
                rec += f' · ✖ {w.error}'
            parts.append(rec)
        else:
            parts.append('■ not recording')
        if self.watcher.enabled:
            parts.append(f'✖ triggers: {self.watcher.error}' if self.watcher.error
                         else f'triggers ok ({self.watcher.n_events} received)')
        cpu = self._cpu_load()
        if cpu:
            parts.append(cpu)
        return '\n'.join(parts)          # one line per item: no word wrap, the height never depends on the width

    def _refresh_preview(self):
        if self._closing or self.preview is None or not self.station.preview:
            return
        try:
            dock = self.docks.get('preview')
            if dock is not None and not dock.isVisible():
                return
            cam_id = self.params.child('station', 'preview_group', 'preview_camera').value()
            cam = self.station.camera(cam_id)
            rt = self.engine.runtime(cam_id) if cam else None
            if rt is None or rt.latest is None:
                self.preview.clear(f'{cam.name if cam else "?"}: no frame yet')
                return
            snap = self.engine.snapshot(cam_id)
            view = self.params.child('station', 'preview_group', 'preview_view').value()
            if view == 'script output':
                shown = self.runner.shown.get(cam_id)
                if shown is None:
                    self.preview.clear(f'{cam.name}: no script output (call ctx.show(array) in a script)')
                else:
                    self.preview.show_array(shown[1], f'{cam.name} — script {shown[0]}')
                return
            sig = None
            if view == 'analysed signal' and cam.kind == 'disk':
                sig = rt.analyzer.last_depletion
            img = rt.latest[0]
            roi = clip_roi(cam.roi, np.asarray(img).shape[:2])
            self.preview.show_frame(img, f'{cam.name} ({STATE_TEXT.get(self.engine.state(cam_id, self.station.stale_s))})',
                                    cam, snap.get('result', {}), signal=sig, signal_roi=roi,
                                    signal_binning=cam.binning)
        except Exception as e:
            self.log.add(f'preview: {e}', logging.WARNING)

    # ================================================================== shutdown
    def shutdown(self):
        if self._closing:
            return
        self._closing = True
        for t in (self.publish_timer, self.ui_timer, self.preview_timer, self.trigger_timer, self.scripts_timer):
            t.stop()
        self.engine.after_analysis = None
        self.runner.stop(1.0)
        if self.save_timer.isActive():
            self.save_timer.stop()
            self._save_settings()
        self.disconnect_cameras()
        bus.clear()
        self.stop_recording()
        self.engine.stop()
        self.log.add('Camera Station closed')
