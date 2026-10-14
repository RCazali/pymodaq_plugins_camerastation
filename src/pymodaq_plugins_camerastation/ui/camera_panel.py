"""One compact panel per camera, in the spirit of a dashboard DAQ_Viewer: a toolbar (snap, grab, stop,
preview, settings), the state, one line of rates and one line of results; the settings of the camera
open below on demand.

Layout stability: every label is single-line with an ignored horizontal size hint (no word wrap), so
a changing text never changes the size of the panel (no resize feedback loop with the dock).
"""
from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets

from pymodaq_gui.parameter import Parameter, ParameterTree

from .icons import glyph_icon, shape_icon


def _tool(icon: QtGui.QIcon, tip: str, checkable: bool = False) -> QtWidgets.QToolButton:
    b = QtWidgets.QToolButton()
    b.setIcon(icon)
    b.setIconSize(QtCore.QSize(20, 20))
    b.setToolTip(tip)
    b.setCheckable(checkable)
    b.setAutoRaise(True)
    return b


def _line(bold: bool = False, mono: bool = False) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel('')
    lab.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
    lab.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    f = lab.font()
    if mono:
        f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        f.setPointSize(lab.font().pointSize())
    f.setBold(bold)
    lab.setFont(f)
    return lab


class CameraPanel(QtWidgets.QFrame):
    snap_clicked = QtCore.Signal()
    grab_toggled = QtCore.Signal(bool)
    stop_clicked = QtCore.Signal()
    preview_clicked = QtCore.Signal()
    settings_toggled = QtCore.Signal(bool)

    def __init__(self, param: Parameter, settings_open: bool = False, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 4)
        lay.setSpacing(1)

        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(2)
        self.title = _line(bold=True)
        f = self.title.font()
        f.setPointSize(f.pointSize() + 1)
        self.title.setFont(f)
        self.state = QtWidgets.QLabel('')
        f = self.state.font()
        f.setBold(True)
        self.state.setFont(f)
        self.state.setMinimumWidth(self.state.fontMetrics().horizontalAdvance('⚠ no frames') + 8)
        self.state.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.snap_btn = _tool(shape_icon('#3498db', 'snap'), 'Snap: grab and analyse one frame')
        self.grab_btn = _tool(shape_icon('#2ecc71', 'triangle'),
                              'Grab: continuous acquisition of this camera (and of the relay to A)', checkable=True)
        self.stop_btn = _tool(shape_icon('#95a5a6', 'square'), 'Stop the acquisition of this camera')
        self.preview_btn = _tool(shape_icon('#bdc3c7', 'image'), 'Show this camera in the Preview dock')
        self.settings_btn = _tool(glyph_icon('⚙', '#bdc3c7'), 'Settings of this camera', checkable=True)
        bar.addWidget(self.title, 1)
        bar.addWidget(self.state)
        bar.addSpacing(6)
        for b in (self.snap_btn, self.grab_btn, self.stop_btn, self.preview_btn, self.settings_btn):
            bar.addWidget(b)
        lay.addLayout(bar)

        self.rates = _line(mono=True)
        self.values = _line(mono=True)
        self.info = _line()
        f = self.info.font()
        f.setItalic(True)
        self.info.setFont(f)
        lay.addWidget(self.rates)
        lay.addWidget(self.values)
        lay.addWidget(self.info)

        self.tree = ParameterTree()
        self.tree.setParameters(param, showTop=False)
        self.tree.setMinimumHeight(280)
        self.tree.setVisible(settings_open)
        self.settings_btn.setChecked(settings_open)
        lay.addWidget(self.tree)

        self.snap_btn.clicked.connect(self.snap_clicked)
        self.grab_btn.clicked.connect(lambda: self.grab_toggled.emit(self.grab_btn.isChecked()))
        self.stop_btn.clicked.connect(self.stop_clicked)
        self.preview_btn.clicked.connect(self.preview_clicked)
        self.settings_btn.toggled.connect(self._toggle_settings)

    def _toggle_settings(self, on: bool):
        self.tree.setVisible(on)
        self.settings_toggled.emit(on)

    def update_view(self, name: str, state: str, ok: bool, grabbing, rates: str, values: str, info: str):
        """``grabbing``: True / False, or None when the camera has no viewer."""
        _set(self.title, name)
        _set(self.state, state if ok else f'<span style="color:#e67e22">{state}</span>')
        _set(self.rates, rates)
        _set(self.values, values)
        _set(self.info, info)
        has = grabbing is not None
        for b in (self.snap_btn, self.grab_btn, self.stop_btn):
            b.setEnabled(has)
        if has and self.grab_btn.isChecked() != bool(grabbing):
            self.grab_btn.setChecked(bool(grabbing))


def _set(label: QtWidgets.QLabel, text: str):
    if label.text() != text:
        label.setText(text)
        label.setToolTip(text)
