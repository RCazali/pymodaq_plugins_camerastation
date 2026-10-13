"""Office simulator of the camera computer.

    python -m pymodaq_plugins_camerastation.tools.sim_preset [--force]
    python -m pymodaq_plugins_camerastation.extensions.camera_station camera_sim

Writes the dashboard preset (PyMoDAQ 5.1) or experiment (5.2) ``camera_sim``: two simulated cameras
(``Disk cam``: disk fluorescence depleted by two passes, ``Output cam``: a drifting beam) and the
``CameraStation`` 0D relay. If this machine has no station settings yet, the two cameras are also
declared in the station (Disk / Output), so that the extension works right away.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

DEFAULT_NAME = 'camera_sim'
CAMERAS: Tuple[Tuple[str, str], ...] = (('Disk cam', 'disk'), ('Output cam', 'beam'))
RELAY = 'CameraStation'


def is_pymodaq_52() -> bool:
    try:
        import pymodaq.utils.managers.experiment.experiment_manager  # noqa: F401
        return True
    except ImportError:
        return False


def _find(param, name: str):
    for child in param.children():
        if child.name() == name:
            return child
        found = _find(child, name)
        if found is not None:
            return found
    return None


def _fix_none_values(param):
    for child in param.children():
        if child.type() != 'group' and child.opts.get('value', None) is None:
            if child.type() in ('bool', 'bool_push', 'led', 'led_push'):
                child.setValue(False)
            elif child.type() == 'list':
                limits = child.opts.get('limits') or []
                values = list(limits.values()) if isinstance(limits, dict) else list(limits)
                if values:
                    child.setValue(values[0])
        _fix_none_values(child)


def _set_detector(d, title: str, scene: Optional[str]):
    d.child('name').setValue(title)
    module_name = _find(d, 'module_name')       # none in an experiment: it has no plugin settings
    if module_name is not None:
        module_name.setValue(title)
    if scene is not None:
        p = _find(d, 'scene')
        if p is not None:
            p.setValue(scene)
    saver = _find(d, 'saver_settings')
    if saver is not None:   # applying these values at load time hits a 5.1 viewer quirk (no h5saver yet)
        saver.remove()


def build_preset_51(name: str):
    from pymodaq_gui.parameter import Parameter
    import pymodaq.utils.managers.preset_manager_utils  # noqa: F401  registers groupmove / groupdet
    p = Parameter.create(title='Preset', name='Preset', type='group', children=[
        {'title': 'Filename:', 'name': 'filename', 'type': 'str', 'value': name},
        {'title': 'Model Settings:', 'name': 'model_settings', 'type': 'group', 'visible': False, 'children': []},
        {'title': 'Moves:', 'name': 'Moves', 'type': 'groupmove'},
        {'title': 'Detectors:', 'name': 'Detectors', 'type': 'groupdet'}])
    dets = p.child('Detectors')
    for title, scene in CAMERAS:
        dets.addNew(('DAQ2D', 'CamSim'))
        _set_detector(dets.children()[-1], title, scene)
    dets.addNew(('DAQ0D', RELAY))
    _set_detector(dets.children()[-1], RELAY, None)
    _fix_none_values(p)
    return p


def build_experiment_52(name: str):
    from pymodaq_gui.parameter import Parameter
    from pymodaq.utils.managers.experiment.experiment_manager import ExperimentManager
    from pymodaq.utils.managers.modules.utils import ModuleType
    p = Parameter.create(title='Preset', name='Preset', type='group', children=ExperimentManager.params)
    dets = p.child(ModuleType.Detector.value)
    for title, scene in CAMERAS:
        dets.addNew(('DAQ2D', 'CamSim'))
        _set_detector(dets.children()[-1], title, scene)
    dets.addNew(('DAQ0D', RELAY))
    _set_detector(dets.children()[-1], RELAY, None)
    _fix_none_values(p)
    return p


def target_path(name: str = DEFAULT_NAME) -> Path:
    if is_pymodaq_52():
        from pymodaq.utils.config import get_set_experiment_path
        return Path(get_set_experiment_path()).joinpath(f'{name}.xml')
    from pymodaq.utils.config import get_set_preset_path
    return Path(get_set_preset_path()).joinpath(f'{name}.xml')


def write_sim_config(name: str = DEFAULT_NAME, overwrite: bool = False) -> Path:
    from pymodaq_gui.parameter import ioxml
    path = target_path(name)
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} already exists (use --force to replace it)')
    p = build_experiment_52(name) if is_pymodaq_52() else build_preset_51(name)
    ioxml.parameter_to_xml_file(p, path.with_suffix(''), overwrite=True)
    return path


def write_sim_station(data_folder: Optional[Path] = None) -> Optional[Path]:
    """Station settings for the simulator, only if this machine has none (never overwrites)."""
    from ..core.settings import CameraConfig, SettingsStore, StationConfig
    from ..utils import Config, machine_name, resolve_paths
    cfg = Config()
    paths = resolve_paths(cfg, data_folder)
    store = SettingsStore(paths['data'] / f'station_{machine_name(cfg)}.json')
    if store.path.exists():
        return None
    st = StationConfig(cameras=[
        CameraConfig(id='disk', name='Disk', detector='Disk cam', kind='disk', binning=2, threshold=0.1,
                     mask_cx=320, mask_cy=240, mask_radius=175),
        CameraConfig(id='out', name='Output', detector='Output cam', kind='beam', binning=2),
    ]).validate()
    store.save(st)
    return store.path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', default=DEFAULT_NAME)
    parser.add_argument('--force', action='store_true', help='replace an existing preset / experiment')
    args = parser.parse_args(argv)
    from qtpy import QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])  # noqa: F841 (Parameters)
    try:
        path = write_sim_config(args.name, overwrite=args.force)
        print(f'dashboard configuration written: {path}')
    except FileExistsError as e:
        print(e)
    if is_pymodaq_52():
        print('An experiment does not hold the plugin settings: set "Scene" to "disk" in the settings of '
              'the "Disk cam" viewer (both cameras start with the scene "beam").')
    st = write_sim_station()
    print(f'station settings written: {st}' if st else 'station settings kept (already present)')
    print('Disk camera: capture the reference with "Laser on" unticked in the "Disk cam" viewer settings.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
