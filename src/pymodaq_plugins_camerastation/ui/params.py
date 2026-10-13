"""Parameter-tree specifications of the station and of one camera (PyMoDAQ style)."""
from __future__ import annotations

from typing import List

from ..core.analysis import BACKGROUNDS, KINDS
from ..core.settings import CameraConfig, StationConfig

NO_DETECTOR = '(none)'


def station_spec(st: StationConfig) -> List[dict]:
    return [
        {'title': 'Station', 'name': 'station', 'type': 'group', 'children': [
            {'title': 'Publish rate (Hz):', 'name': 'publish_hz', 'type': 'float', 'value': st.publish_hz,
             'min': 0.1, 'max': 20.0,
             'tip': '0D values sent to computer A (and logged) per second. 1-2 Hz is plenty for drifts.'},
            {'title': 'Stale after (s):', 'name': 'stale_s', 'type': 'float', 'value': st.stale_s, 'min': 0.5,
             'tip': 'A camera without a new result for this long publishes NaN and valid = 0.'},
            {'title': 'Trigger folder (from A):', 'name': 'trigger_folder', 'type': 'str',
             'value': st.trigger_folder,
             'tip': 'Folder shared with computer A where Laser Control writes its event files (O8). '
                    'Empty = off. A burst of images is recorded on each event.'},
            {'title': 'Recording', 'name': 'recording', 'type': 'group', 'children': [
                {'title': 'Image every (s):', 'name': 'image_period_s', 'type': 'float',
                 'value': st.image_period_s, 'min': 0.0,
                 'tip': 'Periodic reduced image of each camera in the recording (0 = none).'},
                {'title': 'Image binning:', 'name': 'image_binning', 'type': 'list', 'limits': [1, 2, 4, 8],
                 'value': st.image_binning},
                {'title': 'Image bits:', 'name': 'image_bits', 'type': 'list', 'limits': [8, 16],
                 'value': st.image_bits,
                 'tip': '8 bits: per-image offset and scale (counts ~ offset + value * scale), ~10x smaller.'},
                {'title': 'Full frame:', 'name': 'image_full_frame', 'type': 'bool', 'value': st.image_full_frame,
                 'tip': 'Off: only the analysis ROI of each camera is saved.'},
                {'title': 'Images per event:', 'name': 'burst_frames', 'type': 'int', 'value': st.burst_frames,
                 'min': 0, 'max': 100, 'tip': 'Images saved per camera after a trigger event of computer A.'},
            ]},
            {'title': 'Preview', 'name': 'preview_group', 'type': 'group', 'children': [
                {'title': 'Show preview:', 'name': 'preview', 'type': 'bool', 'value': st.preview,
                 'tip': 'Small live image with the analysis overlay (1 Hz). Off = no display cost.'},
                {'title': 'Camera:', 'name': 'preview_camera', 'type': 'list',
                 'limits': [c.id for c in st.cameras] or [''], 'value': st.cameras[0].id if st.cameras else ''},
                {'title': 'Show:', 'name': 'preview_view', 'type': 'list',
                 'limits': ['frame', 'analysed signal', 'script output'],
                 'value': 'frame', 'tip': '"analysed signal": what the moments are computed on '
                                          '(depletion for a disk camera). "script output": the last '
                                          'array given to ctx.show() by a script of this camera.'},
            ]},
            {'title': 'Add camera', 'name': 'add_camera', 'type': 'action'},
        ]},
        {'title': 'Scripts (test analyses live)', 'name': 'scripts', 'type': 'group', 'children': [
            {'title': 'Folder:', 'name': 'scripts_folder', 'type': 'str', 'value': '', 'readonly': True,
             'tip': 'A .py file saved here is loaded / reloaded within seconds (see _template.py).'},
            {'title': 'Open folder', 'name': 'open_scripts_folder', 'type': 'action'},
            {'title': 'Reload all scripts', 'name': 'reload_scripts', 'type': 'action'},
            {'title': 'Timeout (s):', 'name': 'script_timeout_s', 'type': 'float', 'value': st.script_timeout_s,
             'min': 1.0, 'tip': 'A script running longer is disabled until its file is saved again.'},
        ]},
        {'title': 'Cameras', 'name': 'cameras', 'type': 'group', 'children': [camera_spec(c, []) for c in st.cameras]},
    ]


SCRIPT_PREFIX = 'script__'
_PTYPE = {bool: 'bool', int: 'int', float: 'float', str: 'str'}


def script_param_spec(key: str, value) -> dict:
    return {'title': f'{key}:', 'name': key, 'type': _PTYPE.get(type(value), 'str'), 'value': value}


def script_spec(name: str, script, setting: dict, status: str) -> dict:
    """Station-level group of one script: status, publication, live parameters."""
    params = dict(getattr(script, 'defaults', {}) or {})
    for k, v in (setting.get('params') or {}).items():
        if k in params and type(v) is type(params[k]):
            params[k] = v
    return {'title': name, 'name': SCRIPT_PREFIX + name, 'type': 'group', 'tip': getattr(script, 'doc', ''),
            'children': [
                {'title': 'Status:', 'name': 'status', 'type': 'str', 'value': status, 'readonly': True},
                {'title': 'Publish to A:', 'name': 'publish', 'type': 'bool', 'value': bool(setting.get('publish')),
                 'tip': 'Send the results to computer A as "<camera>_<script>/<label>". Off: shown and '
                        'recorded only (testing).'},
                {'title': 'Parameters', 'name': 'params', 'type': 'group',
                 'children': [script_param_spec(k, v) for k, v in params.items()]},
            ]}


def camera_scripts_children(c: CameraConfig, script_names: List[str]) -> List[dict]:
    return [{'title': n, 'name': n, 'type': 'bool', 'value': n in c.scripts} for n in script_names]


def camera_spec(c: CameraConfig, detectors: List[str], script_names: List[str] = ()) -> dict:
    dets = [NO_DETECTOR] + sorted(set(detectors) | ({c.detector} if c.detector else set()))
    return {'title': f'{c.name}', 'name': c.id, 'type': 'group', 'children': [
        {'title': 'Name (channel):', 'name': 'name', 'type': 'str', 'value': c.name,
         'tip': 'Published as "<viewer on A>/<name>/<label>". Changing it changes the keys seen by A.'},
        {'title': 'Camera (2D viewer):', 'name': 'detector', 'type': 'list', 'limits': dets,
         'value': c.detector or NO_DETECTOR},
        {'title': 'Enabled:', 'name': 'enabled', 'type': 'bool', 'value': c.enabled},
        {'title': 'Analysis:', 'name': 'kind', 'type': 'list', 'limits': list(KINDS), 'value': c.kind,
         'tip': '\n'.join(f'{k}: {v}' for k, v in KINDS.items())},
        {'title': 'Max rate (Hz):', 'name': 'rate_hz', 'type': 'float', 'value': c.rate_hz, 'min': 0.05,
         'max': 30.0, 'tip': 'Frames above this rate are dropped before any work.'},
        {'title': 'Binning:', 'name': 'binning', 'type': 'list', 'limits': [1, 2, 4, 8], 'value': c.binning,
         'tip': 'Binning before the analysis: 2 = 4x less work; results stay in full-frame pixels.'},
        {'title': 'ROI', 'name': 'roi', 'type': 'group', 'expanded': False, 'children': [
            {'title': 'x0 (px):', 'name': 'x0', 'type': 'int', 'value': c.roi[0], 'min': 0},
            {'title': 'y0 (px):', 'name': 'y0', 'type': 'int', 'value': c.roi[1], 'min': 0},
            {'title': 'Width (px, 0 = to edge):', 'name': 'w', 'type': 'int', 'value': c.roi[2], 'min': 0},
            {'title': 'Height (px, 0 = to edge):', 'name': 'h', 'type': 'int', 'value': c.roi[3], 'min': 0},
            {'title': 'Full frame', 'name': 'roi_full', 'type': 'action'},
        ]},
        {'title': 'Signal', 'name': 'signal', 'type': 'group', 'expanded': False, 'children': [
            {'title': 'Background:', 'name': 'background', 'type': 'list', 'limits': list(BACKGROUNDS),
             'value': c.background, 'tip': 'none / border (median of the ROI border) / dark (captured dark '
                                           'frame). Not used by the disk analysis (the reference cancels it).'},
            {'title': 'Threshold (fraction of max):', 'name': 'threshold', 'type': 'float', 'value': c.threshold,
             'min': 0.0, 'max': 0.95, 'step': 0.01},
            {'title': 'Min signal (counts/px):', 'name': 'min_signal', 'type': 'float', 'value': c.min_signal,
             'min': 0.0, 'tip': 'Below this peak (above background) there is no beam: NaN, valid = 0.'},
            {'title': 'Saturation (counts, 0 = auto):', 'name': 'saturation', 'type': 'float',
             'value': c.saturation, 'min': 0.0, 'tip': 'e.g. 4095 for a 12-bit camera in 16-bit words.'},
        ]},
        {'title': 'Disk', 'name': 'disk', 'type': 'group', 'expanded': c.kind == 'disk', 'children': [
            {'title': 'Mask centre x (px):', 'name': 'mask_cx', 'type': 'float', 'value': c.mask_cx, 'min': 0.0},
            {'title': 'Mask centre y (px):', 'name': 'mask_cy', 'type': 'float', 'value': c.mask_cy, 'min': 0.0},
            {'title': 'Mask radius (px, 0 = none):', 'name': 'mask_radius', 'type': 'float',
             'value': c.mask_radius, 'min': 0.0},
            {'title': 'Reference:', 'name': 'reference_file', 'type': 'str', 'value': c.reference_file,
             'readonly': True},
            {'title': 'Frames averaged:', 'name': 'n_capture', 'type': 'int', 'value': 10, 'min': 1, 'max': 200},
            {'title': 'Capture reference (laser OFF)', 'name': 'capture_ref', 'type': 'action'},
            {'title': 'Clear reference', 'name': 'clear_ref', 'type': 'action'},
        ]},
        {'title': 'Dark frame', 'name': 'dark', 'type': 'group', 'expanded': False, 'children': [
            {'title': 'Dark frame:', 'name': 'dark_file', 'type': 'str', 'value': c.dark_file, 'readonly': True},
            {'title': 'Capture dark (camera covered)', 'name': 'capture_dark', 'type': 'action'},
            {'title': 'Clear dark', 'name': 'clear_dark', 'type': 'action'},
        ]},
        {'title': 'Scripts', 'name': 'scripts', 'type': 'group', 'tip': 'User scripts run on this camera',
         'children': camera_scripts_children(c, list(script_names))},
        {'title': 'Record images:', 'name': 'record_images', 'type': 'bool', 'value': c.record_images},
        {'title': 'Remove this camera', 'name': 'remove', 'type': 'action'},
    ]}
