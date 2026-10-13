"""Light preview of one camera: decimated image + analysis overlay (pyqtgraph only, no Viewer2D).

Cost control: refreshed by a slow timer (1 Hz) only when the dock is visible, the image is decimated
to <= 400 px by slicing (no copy, no interpolation) and the levels are computed on that small image.
The ROI rectangle can be dragged: the new ROI is sent back to the camera settings.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

MAX_PX = 400


class PreviewWidget(QtWidgets.QWidget):
    def __init__(self, on_roi_changed: Callable[[list], None], parent=None):
        super().__init__(parent)
        self.on_roi_changed = on_roi_changed
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.info = QtWidgets.QLabel('Preview off')
        f = self.info.font()
        f.setPointSize(f.pointSize() + 1)
        f.setBold(True)
        self.info.setFont(f)
        lay.addWidget(self.info)
        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw, 1)
        self.vb = self.glw.addViewBox(lockAspect=True, invertY=True)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        pen_roi = pg.mkPen((255, 255, 0), width=2)
        self.roi = pg.RectROI([0, 0], [10, 10], pen=pen_roi, movable=True, resizable=True, removable=False)
        self.roi.sigRegionChangeFinished.connect(self._roi_moved)
        self.vb.addItem(self.roi)
        self.mark = pg.ScatterPlotItem(size=22, symbol='+', pen=pg.mkPen((255, 255, 255), width=3), brush=None)
        self.vb.addItem(self.mark)
        self.ellipse = QtWidgets.QGraphicsEllipseItem()
        self.ellipse.setPen(pg.mkPen((0, 255, 255), width=2, style=QtCore.Qt.DashLine))
        self.vb.addItem(self.ellipse)
        self.mask = QtWidgets.QGraphicsEllipseItem()
        self.mask.setPen(pg.mkPen((255, 128, 255), width=1, style=QtCore.Qt.DotLine))
        self.vb.addItem(self.mask)
        self._updating = False
        self._shape = None

    def clear(self, text: str):
        self.info.setText(text)
        self.img.clear()
        self.mark.setData([])
        self.ellipse.setRect(QtCore.QRectF())

    def show_frame(self, image, title: str, cfg, result: dict, signal: Optional[np.ndarray] = None,
                   signal_roi=None, signal_binning: int = 1):
        a = np.asarray(image)
        if a.ndim == 3:
            a = a[..., 0]
        H, W = a.shape[:2]
        if signal is not None and signal_roi is not None:
            x0, y0, w, h = signal_roi
            step = max(1, int(math.ceil(max(signal.shape) / MAX_PX)))
            disp = signal[::step, ::step]
            rect = QtCore.QRectF(x0, y0, signal.shape[1] * signal_binning, signal.shape[0] * signal_binning)
        else:
            step = max(1, int(math.ceil(max(H, W) / MAX_PX)))
            disp = a[::step, ::step]
            rect = QtCore.QRectF(0, 0, W, H)
        disp = np.asarray(disp, dtype=np.float32)
        lo, hi = (float(disp.min()), float(disp.max())) if disp.size else (0.0, 1.0)
        self.img.setImage(disp.T, levels=(lo, hi if hi > lo else lo + 1), autoLevels=False)
        self.img.setRect(rect)
        if self._shape != (H, W):
            self._shape = (H, W)
            self.vb.autoRange()
        # ROI rectangle (full-frame pixels)
        x0, y0, w, h = cfg.roi
        w = W - x0 if w <= 0 else w
        h = H - y0 if h <= 0 else h
        self.roi.setVisible(True)
        if not getattr(self.roi, 'isMoving', False):       # never fight the user while dragging
            self._updating = True
            self.roi.setPos([x0, y0], update=False, finish=False)
            self.roi.setSize([w, h], update=True, finish=False)
            self._updating = False
        # centroid and 2-sigma ellipse
        x, y = result.get('x', math.nan), result.get('y', math.nan)
        sx, sy = result.get('sx', math.nan), result.get('sy', math.nan)
        if all(map(math.isfinite, (x, y))):
            self.mark.setData([x], [y])
            if all(map(math.isfinite, (sx, sy))):
                self.ellipse.setRect(QtCore.QRectF(x - 2 * sx, y - 2 * sy, 4 * sx, 4 * sy))
        else:
            self.mark.setData([])
            self.ellipse.setRect(QtCore.QRectF())
        r = cfg.mask_radius if cfg.kind == 'disk' else 0
        self.mask.setRect(QtCore.QRectF(cfg.mask_cx - r, cfg.mask_cy - r, 2 * r, 2 * r) if r > 0 else QtCore.QRectF())
        self.info.setText(f'{title}   {W}x{H} px   display 1/{step}   levels {lo:.0f}…{hi:.0f}')

    def show_array(self, array, title: str):
        """Any 2D array given by a script (its own pixel grid: no ROI / centroid overlay)."""
        a = np.asarray(array, dtype=np.float32)
        if a.ndim != 2 or a.size == 0:
            self.clear(f'{title}: not a 2D array {a.shape}')
            return
        step = max(1, int(math.ceil(max(a.shape) / MAX_PX)))
        disp = a[::step, ::step]
        finite = disp[np.isfinite(disp)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        self.img.setImage(np.nan_to_num(disp).T, levels=(lo, hi if hi > lo else lo + 1), autoLevels=False)
        self.img.setRect(QtCore.QRectF(0, 0, a.shape[1], a.shape[0]))
        if self._shape != ('array', a.shape):
            self._shape = ('array', a.shape)
            self.vb.autoRange()
        for item in (self.ellipse, self.mask):
            item.setRect(QtCore.QRectF())
        self.mark.setData([])
        self.roi.setVisible(False)          # its pixels are not camera pixels: no ROI editing here
        self.info.setText(f'{title}   {a.shape[1]}x{a.shape[0]} (script pixels)   levels {lo:.3g}…{hi:.3g}')

    def _roi_moved(self):
        if self._updating:
            return
        pos, size = self.roi.pos(), self.roi.size()
        roi = [int(round(max(pos.x(), 0))), int(round(max(pos.y(), 0))),
               int(round(max(size.x(), 1))), int(round(max(size.y(), 1)))]
        self.on_roi_changed(roi)
