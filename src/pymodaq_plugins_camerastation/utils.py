# -*- coding: utf-8 -*-
"""
Per-machine configuration of the Camera Station plugin (PyMoDAQ plugin-template convention).

The user file lives in the PyMoDAQ local folder (``~/.pymodaq/config_camerastation.toml``), outside
git, one per machine. Missing entries are completed from ``resources/config_template.toml``.
"""
import socket
from pathlib import Path

from pymodaq_utils.config import BaseConfig, USER  # noqa: F401  (USER kept for template parity)


class Config(BaseConfig):
    """Main class to deal with configuration values for this plugin"""
    config_template_path = Path(__file__).parent.joinpath('resources/config_template.toml')
    config_name = f"config_{__package__.split('pymodaq_plugins_')[1]}"


def config_get(config, *path, default=None):
    """Tolerant read of a config value (works with every pymodaq_utils BaseConfig version)."""
    try:
        return config(*path)
    except Exception:
        return default


def machine_name(config) -> str:
    name = str(config_get(config, 'machine', 'name', default='') or '').strip()
    return name or socket.gethostname()


def resolve_paths(config, override: Path = None) -> dict:
    """Folders of the station: data (settings, references), recordings. Created if needed."""
    if override is not None:
        base = Path(override)
    else:
        data = str(config_get(config, 'paths', 'data_folder', default='') or '').strip()
        if data:
            base = Path(data)
        else:
            from pymodaq_utils.config import get_set_local_dir
            base = Path(get_set_local_dir()) / 'camera_station'
    rec = str(config_get(config, 'paths', 'recordings_folder', default='') or '').strip() if override is None else ''
    paths = {
        'data': base,
        'references': base / 'references',
        'scripts': base / 'scripts',
        'recordings': Path(rec) if rec else base / 'recordings',
        'logs': base / 'logs',
    }
    for p in paths.values():
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    return paths
