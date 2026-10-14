"""Icons painted at run time (no resource files, no dependence on the icon set of the PyMoDAQ version).

Colours are painted in the pixmap, never set with a style sheet: a per-widget style sheet replaces the
rules of the PyMoDAQ theme for that widget (sizes, paddings) and breaks the alignment of toolbars."""
from qtpy import QtCore, QtGui, QtWidgets


def shape_icon(color: str, shape: str = 'circle') -> QtGui.QIcon:
    pix = QtGui.QPixmap(32, 32)
    pix.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setBrush(QtGui.QColor(color))
    p.setPen(QtGui.QPen(QtGui.QColor('white'), 2))
    if shape == 'square':
        p.drawRoundedRect(4, 4, 24, 24, 4, 4)
    elif shape == 'triangle':
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(8, 4), QtCore.QPointF(28, 16), QtCore.QPointF(8, 28)]))
    elif shape == 'snap':                       # one frame: a ring and a dot
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(color), 3))
        p.drawEllipse(4, 4, 24, 24)
        p.setBrush(QtGui.QColor(color))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(11, 11, 10, 10)
    elif shape == 'image':                      # preview: a frame with a spot
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(color), 3))
        p.drawRoundedRect(3, 6, 26, 20, 3, 3)
        p.setBrush(QtGui.QColor(color))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(12, 12, 8, 8)
    else:
        p.drawEllipse(4, 4, 24, 24)
    p.end()
    return QtGui.QIcon(pix)


def glyph_icon(text: str, color: str) -> QtGui.QIcon:
    pix = QtGui.QPixmap(32, 32)
    pix.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    f = QtGui.QFont()
    f.setPixelSize(28)
    p.setFont(f)
    p.setPen(QtGui.QColor(color))
    p.drawText(QtCore.QRect(0, 0, 32, 32), QtCore.Qt.AlignCenter, text)
    p.end()
    return QtGui.QIcon(pix)


def std_icon(name: str) -> QtGui.QIcon:
    style = QtWidgets.QApplication.style()
    sp = getattr(QtWidgets.QStyle, name, None)
    return style.standardIcon(sp) if sp is not None else QtGui.QIcon()
