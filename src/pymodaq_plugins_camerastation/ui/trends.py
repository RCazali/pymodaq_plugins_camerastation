"""Trend plots of the results in the extension: one small plot per quantity, each with its own y scale.

Why not the dashboard: the CameraStation 0D viewer is the bridge to computer A (one Data0D per camera,
all labels together, so a ``sum`` of 1e6 flattens a centroid of 320 px). Here every quantity has its
own plot, all plots share the time axis.

Cost control: every result is stored once in a ring buffer (numpy, no Python lists); the curves are
redrawn by the display timer only when the dock is visible and the trends are on.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

RING_SIZE = 8192          # samples per quantity (13 min at 10 Hz); the history setting trims the display
PLOT_MIN_HEIGHT = 95
COLORS = ((31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40))

# camera results: one plot per quantity (x and y in the same plot would share a scale and hide the drift
# of each): (label, title, unit, scale, shown by default).
CAMERA_GROUPS: Tuple[Tuple[str, str, str, float, bool], ...] = (
    ('x', 'x', 'px', 1.0, True),
    ('y', 'y', 'px', 1.0, True),
    ('sx', 'σx', 'px', 1.0, True),
    ('sy', 'σy', 'px', 1.0, True),
    ('sum', 'sum', 'counts', 1.0, True),
    ('mean', 'mean', 'counts', 1.0, True),
    ('peak', 'peak', 'counts', 1.0, False),
    ('kx', 'kurtosis x', '', 1.0, False),
    ('ky', 'kurtosis y', '', 1.0, False),
    ('sat', 'saturation', '%', 100.0, False),
)


@dataclass(frozen=True)
class PlotSpec:
    key: str                       # "<source>/<group>": settings key of the plot
    source: str                    # camera name or "<camera>_<script>" (the published data name)
    title: str
    labels: Tuple[str, ...]
    unit: str = ''
    scale: float = 1.0
    default: bool = True           # shown until the user decides otherwise


def camera_specs(name: str, labels: Sequence[str]) -> List[PlotSpec]:
    have = set(labels)
    return [PlotSpec(f'{name}/{lab}', name, f'{name}: {title}', (lab,), unit, scale, shown)
            for lab, title, unit, scale, shown in CAMERA_GROUPS if lab in have]


def script_specs(pub: str, labels: Sequence[str]) -> List[PlotSpec]:
    return [PlotSpec(f'{pub}/{lab}', pub, f'{pub}: {lab}', (lab,)) for lab in labels if lab != 'valid']


class Ring:
    """Fixed-size (time, value) buffer."""

    def __init__(self, n: int = RING_SIZE):
        self.t = np.full(n, np.nan)
        self.v = np.full(n, np.nan)
        self.i = 0
        self.full = False

    def append(self, t: float, v: float):
        self.t[self.i] = t
        self.v[self.i] = v
        self.i += 1
        if self.i == len(self.t):
            self.i = 0
            self.full = True

    def last(self) -> float:
        if not self.full and self.i == 0:
            return math.nan
        return float(self.v[self.i - 1])

    def since(self, t0: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.full:
            t = np.concatenate((self.t[self.i:], self.t[:self.i]))
            v = np.concatenate((self.v[self.i:], self.v[:self.i]))
        else:
            t, v = self.t[:self.i], self.v[:self.i]
        k = int(np.searchsorted(t, t0))        # times are increasing
        return t[k:], v[k:]


def _value(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


class TrendsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.info = QtWidgets.QLabel('')
        lay.addWidget(self.info)
        self.glw = pg.GraphicsLayoutWidget()
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.glw)
        lay.addWidget(self.scroll, 1)
        self.rings: Dict[Tuple[str, str], Ring] = {}
        self.specs: List[PlotSpec] = []
        self.plots: Dict[str, Tuple[pg.PlotItem, Dict[str, pg.PlotDataItem]]] = {}
        self.enabled = True
        self.set_enabled(True)

    # ------------------------------------------------------------------ data
    def add(self, source: str, t: float, values: Dict[str, float]):
        valid = values.get('valid', 1.0)
        for lab, v in values.items():
            if lab == 'valid':
                continue
            ring = self.rings.get((source, lab))
            if ring is None:
                ring = self.rings[(source, lab)] = Ring()
            ring.append(float(t), _value(v) if valid == 1.0 else math.nan)

    # ------------------------------------------------------------------ layout
    def set_enabled(self, on: bool):
        self.enabled = bool(on)
        self.scroll.setVisible(self.enabled)
        self.info.setText('' if on else 'Trends off (Settings > Trends > Show trends): no display cost.')
        self.info.setVisible(not on or not self.specs)

    def set_plots(self, specs: List[PlotSpec]):
        if [s.key for s in specs] == [s.key for s in self.specs]:
            return
        self.specs = list(specs)
        self.glw.clear()
        self.plots.clear()
        master: Optional[pg.PlotItem] = None
        for row, spec in enumerate(self.specs):
            p = self.glw.addPlot(row=row, col=0, axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
            p.showGrid(x=True, y=True, alpha=0.2)
            p.getAxis('left').setWidth(55)
            p.setMenuEnabled(False)
            if master is None:
                master = p
            else:
                p.setXLink(master)
            if row < len(self.specs) - 1:
                p.getAxis('bottom').setStyle(showValues=False)
            curves = {lab: p.plot([], [], pen=pg.mkPen(COLORS[i % len(COLORS)], width=1.5), connect='finite')
                      for i, lab in enumerate(spec.labels)}
            self.plots[spec.key] = (p, curves)
        self.glw.setMinimumHeight(PLOT_MIN_HEIGHT * max(len(self.specs), 1))
        if not self.specs:
            self.info.setText('No plot selected (Settings > Trends > Plots).' if self.enabled else self.info.text())
        self.info.setVisible(not self.enabled or not self.specs)

    # ------------------------------------------------------------------ drawing
    def redraw(self, history_s: float, now: Optional[float] = None):
        if not self.enabled or not self.specs or not self.isVisible():
            return
        now = time.time() if now is None else now
        t0 = now - history_s
        master = None
        for spec in self.specs:
            p, curves = self.plots[spec.key]
            master = master or p
            last = []
            for lab, curve in curves.items():
                ring = self.rings.get((spec.source, lab))
                if ring is None:
                    curve.setData([], [])
                    continue
                t, v = ring.since(t0)
                curve.setData(t, v * spec.scale, connect='finite')
                lv = ring.last() * spec.scale
                last.append(f'{lab} {_fmt(lv)}' if len(curves) > 1 else _fmt(lv))
            unit = f' {spec.unit}' if spec.unit else ''
            p.setTitle(f'{spec.title}   <b>{"   ".join(last)}</b>{unit}', size='9pt')
        if master is not None:
            master.setXRange(t0, now, padding=0)


def _fmt(v: float) -> str:
    if not math.isfinite(v):
        return '—'
    a = abs(v)
    if a >= 1e5 or (0 < a < 1e-2):
        return f'{v:.3g}'
    return f'{v:.2f}' if a < 100 else f'{v:.1f}'
