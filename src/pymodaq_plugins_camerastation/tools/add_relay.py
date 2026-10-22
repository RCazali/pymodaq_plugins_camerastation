"""Copy a dashboard preset of computer B and add the CameraStation relay connected to the LECO coordinator of A.

The source preset is only read; the copy gets a DAQ0D ``CameraStation`` viewer whose LECO settings (host, port,
name) are filled; the Camera Station extension connects it to the coordinator when it starts it (PyMoDAQ 5.1)::

    python -m pymodaq_plugins_camerastation.tools.add_relay disk disk_leco --host 192.168.0.100

Then open the dashboard with ``disk_leco`` and the Camera Station extension. Computer A receives the relay under
its LECO name (default ``CameraStation``): actor name of its LECODirector viewer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .sim_preset import _find, _fix_none_values, _set_detector, is_pymodaq_52

RELAY_PLUGIN = 'CameraStation'


def add_relay(source: str, target: str, host: str, port: int = 12300, name: str = 'CameraStation',
              title: str = 'CameraStation', overwrite: bool = False) -> Path:
    from pymodaq_gui.parameter import ioxml, Parameter
    import pymodaq.utils.managers.preset_manager_utils  # noqa: F401  registers groupmove / groupdet
    from pymodaq.utils.config import get_set_preset_path
    folder = Path(get_set_preset_path())
    src = folder / f'{source}.xml'
    dst = folder / f'{target}.xml'
    if src == dst:
        raise ValueError('the copy must have another name: the source preset is never modified')
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        raise FileExistsError(f'{dst} already exists (use --force to replace it)')
    p = Parameter.create(title='Preset', name='Preset', type='group', children=ioxml.XML_file_to_parameter(src))
    p.child('filename').setValue(target)
    dets = p.child('Detectors')
    for d in dets.children():
        if _find(d, 'detector') is not None and str(_find(d, 'detector').value()) == RELAY_PLUGIN:
            raise ValueError(f'{source} already has a {RELAY_PLUGIN} relay ({d.child("name").value()})')
    dets.addNew(('DAQ0D', RELAY_PLUGIN))
    d = dets.children()[-1]
    _set_detector(d, title, None)
    leco = _find(d, 'leco')
    # "Connect" stays unticked: PyMoDAQ 5.1 would apply it before host / port (connection to the default address);
    # the Camera Station extension connects a relay that has a LECO name when it starts it
    for key, value in (('host', host), ('port', int(port)), ('leco_name', name), ('connect_leco_server', False)):
        _find(leco, key).setValue(value)
    _fix_none_values(d)
    ioxml.parameter_to_xml_file(p, dst.with_suffix(''), overwrite=True)
    return dst


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('source', help='preset to copy (name without .xml), e.g. disk')
    parser.add_argument('target', help='name of the copy, e.g. disk_leco')
    parser.add_argument('--host', required=True, help='address of the LECO coordinator (computer A)')
    parser.add_argument('--port', type=int, default=12300)
    parser.add_argument('--name', default='CameraStation', help='LECO name of the relay')
    parser.add_argument('--force', action='store_true', help='replace an existing copy')
    a = parser.parse_args(argv)
    if is_pymodaq_52():
        print('PyMoDAQ 5.2+: add the relay in the experiment editor (no dashboard presets)')
        return 2
    from qtpy import QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])  # noqa: F841
    path = add_relay(a.source, a.target, a.host, a.port, a.name, overwrite=a.force)
    print(f'preset written: {path} (relay "{a.name}" -> {a.host}:{a.port})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
