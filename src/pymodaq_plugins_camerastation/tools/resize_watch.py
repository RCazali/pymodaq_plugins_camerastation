"""Find which widget keeps resizing in a PyMoDAQ window (diagnosis of a "display that keeps moving").

Loads the dashboard (no extension), watches every resize event of the application and prints, once
per second, the widgets resized more than ``--min`` times in that second, with the dock they belong
to. Reproduce the problem (drag the edge between two docks), read the console, Ctrl+C to stop.

    python -m pymodaq_plugins_camerastation.tools.resize_watch <experiment or preset name>

PyMoDAQ 5.2+: experiment name; 5.1: preset name. The printout is what a bug report needs: the class
of the oscillating widget, its dock and its sizes.
"""
from __future__ import annotations

import argparse
import collections
import sys
import time

from qtpy import QtCore, QtWidgets

RESIZE_EVENTS = {QtCore.QEvent.Resize, QtCore.QEvent.GraphicsSceneResize}


def _dock_of(obj) -> str:
    w = obj if isinstance(obj, QtWidgets.QWidget) else None
    if w is None and hasattr(obj, 'scene'):                       # QGraphicsWidget: its view
        try:
            views = obj.scene().views()
            w = views[0] if views else None
        except Exception:
            w = None
    while w is not None:
        name = getattr(w, 'name', None)
        if type(w).__name__ == 'Dock' and callable(name):
            return str(name())
        w = w.parentWidget()
    return '?'


def _size(obj) -> str:
    try:
        s = obj.size()
        return f'{int(s.width())}x{int(s.height())}'
    except Exception:
        return '?'


class ResizeWatch(QtCore.QObject):
    def __init__(self, min_per_s: int = 10, out=sys.stdout):
        super().__init__()
        self.min_per_s = min_per_s
        self.out = out
        self.counts: 'collections.Counter' = collections.Counter()
        self.sizes = {}
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.report)
        self.timer.start(1000)

    def eventFilter(self, obj, event):
        try:
            if event.type() in RESIZE_EVENTS:
                key = (type(obj).__name__, obj.objectName() if hasattr(obj, 'objectName') else '', id(obj))
                self.counts[key] += 1
                self.sizes[key] = obj
        except Exception:
            pass
        return False

    def report(self):
        hot = [(k, n) for k, n in self.counts.items() if n >= self.min_per_s]
        if hot:
            stamp = time.strftime('%H:%M:%S')
            print(f'--- {stamp}: {len(hot)} widget(s) resized >= {self.min_per_s} times in 1 s', file=self.out)
            for (cls, name, _), n in sorted(hot, key=lambda kn: -kn[1])[:15]:
                obj = self.sizes.get((cls, name, _))
                print(f'  {n:5d}x  {cls}{"/" + name if name else ""}  size {_size(obj)}  dock "{_dock_of(obj)}"',
                      file=self.out)
            self.out.flush()
        self.counts.clear()
        self.sizes.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('name', help='experiment (PyMoDAQ 5.2+) or preset (5.1) to load')
    parser.add_argument('--min', type=int, default=10, help='resizes per second to report (default 10)')
    args = parser.parse_args()
    try:
        from pymodaq_gui.qt_utils import mkQApp
    except ImportError:
        from pymodaq_gui.utils.utils import mkQApp
    app = mkQApp('Resize watch')
    watch = ResizeWatch(args.min)
    app.installEventFilter(watch)
    try:
        from pymodaq.dashboard import load_dashboard_with_experiment
        keep = load_dashboard_with_experiment(args.name)          # noqa: F841  (references kept alive)
    except ImportError:
        from pymodaq.utils.gui_utils.loader_utils import load_dashboard_with_preset
        keep = load_dashboard_with_preset(args.name, None)        # noqa: F841
    print('Dashboard loaded: reproduce the problem, widgets resizing continuously are listed here.', flush=True)
    fn = getattr(app, 'exec', None) or app.exec_
    sys.exit(fn())


if __name__ == '__main__':
    main()
