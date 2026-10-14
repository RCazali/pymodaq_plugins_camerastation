"""The resize watcher names the widget that keeps resizing (diagnosis tool, tools/resize_watch.py)."""
import io
import os

import pytest

pytest.importorskip('pymodaq')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from test_station_pymodaq import app  # noqa: E402
from qtpy import QtWidgets  # noqa: E402


def test_oscillating_widget_is_reported():
    a = app()
    from pymodaq_plugins_camerastation.tools.resize_watch import ResizeWatch
    out = io.StringIO()
    watch = ResizeWatch(min_per_s=5, out=out)
    watch.timer.stop()
    a.installEventFilter(watch)
    try:
        w = QtWidgets.QLabel('oscillating')
        w.setObjectName('culprit')
        w.show()
        for i in range(20):
            w.resize(100 + 10 * (i % 2), 50)
            a.processEvents()
        quiet = QtWidgets.QLabel('quiet')
        quiet.show()
        a.processEvents()
        watch.report()
    finally:
        a.removeEventFilter(watch)
    text = out.getvalue()
    assert 'QLabel/culprit' in text and 'quiet' not in text
