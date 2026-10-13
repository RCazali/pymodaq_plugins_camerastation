"""Link between the Camera Station extension and its 0D relay plugin (same process, Qt signal).

The extension publishes ``{data name: {label: value}}`` at a fixed rate; the relay plugin
(``DAQ_0DViewer_CameraStation``) re-emits it as a DataToExport. Connected to LECO, that viewer is
what computer A receives (keys ``<viewer on A>/<data name>/<label>``).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from qtpy.QtCore import QObject, Signal

from pymodaq_data.data import DataToExport, DataWithAxes


def payload_to_dte(payload: Dict[str, Dict[str, float]]) -> DataToExport:
    """One Data0D per data name, one channel per label (labels kept, so the keys are stable)."""
    data = []
    for name, values in payload.items():
        labels = list(values)
        if not labels:
            continue
        data.append(DataWithAxes(name=name, source='calculated', dim='Data0D', labels=labels,
                                 data=[np.array([float(values[lab])]) for lab in labels]))
    return DataToExport(name='CameraStation', data=data)


class _StationBus(QObject):
    data_ready = Signal(object)          # DataToExport

    def __init__(self):
        super().__init__()
        self.last: Optional[DataToExport] = None
        self.n_published = 0

    def publish(self, payload: Dict[str, Dict[str, float]]):
        dte = payload_to_dte(payload)
        self.last = dte
        self.n_published += 1
        self.data_ready.emit(dte)

    def clear(self):
        """The extension stops: the relay must not serve old values any more."""
        self.last = None


bus = _StationBus()
