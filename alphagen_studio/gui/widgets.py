"""Společné ovládací prvky."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QSizePolicy, QSlider, QVBoxLayout, QWidget


class LabeledSlider(QWidget):
    """Popisek + posuvník + číselné pole (synchronizované)."""

    valueChanged = Signal(float)  # noqa: N815

    def __init__(self, label: str, minimum: float, maximum: float, value: float, step: float = 0.01,
                 decimals: int = 2, suffix: str = "", tooltip: str = "", parent=None) -> None:
        super().__init__(parent)
        self._scale = 1.0 / step
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(1)
        top = QHBoxLayout()
        self.label = QLabel(label)
        top.addWidget(self.label)
        top.addStretch(1)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.spin.setFixedWidth(82)
        self.spin.setValue(value)
        top.addWidget(self.spin)
        lay.addLayout(top)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(round(minimum * self._scale)), int(round(maximum * self._scale)))
        self.slider.setValue(int(round(value * self._scale)))
        lay.addWidget(self.slider)
        if tooltip:
            self.setToolTip(tooltip)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self._default = value

    def _from_slider(self, v: int) -> None:
        val = v / self._scale
        if abs(self.spin.value() - val) > 1e-9:
            self.spin.blockSignals(True)
            self.spin.setValue(val)
            self.spin.blockSignals(False)
        self.valueChanged.emit(val)

    def _from_spin(self, val: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(val * self._scale)))
        self.slider.blockSignals(False)
        self.valueChanged.emit(val)

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, v: float) -> None:  # noqa: N802
        self.spin.setValue(v)

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802 - dvojklik = výchozí hodnota
        self.setValue(self._default)
        super().mouseDoubleClickEvent(e)


class Section(QWidget):
    """Rozbalovací sekce („roletka“): klik na nadpis ji otevře/zavře. Obsah do ``self.body``."""

    def __init__(self, title: str, expanded: bool = False, sub: bool = False) -> None:
        from PySide6.QtWidgets import QFrame, QToolButton

        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2 if sub else 6, 0, 2)
        lay.setSpacing(0)
        self.header = QToolButton()
        self.header.setObjectName("SubSection" if sub else "Section")
        self.header.setText(title.replace("&", "&&"))  # "&" by Qt bral jako zkratku
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.toggled.connect(self._toggle)
        lay.addWidget(self.header)
        self.frame = QFrame()
        self.frame.setObjectName("SubSectionBody" if sub else "SectionBody")
        self.body = QVBoxLayout(self.frame)
        self.body.setContentsMargins(14, 8, 12, 10)
        lay.addWidget(self.frame)
        self.frame.setVisible(expanded)

    def _toggle(self, on: bool) -> None:
        self.header.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.frame.setVisible(on)

    def set_expanded(self, on: bool) -> None:
        self.header.setChecked(on)


def dim_label(text: str, wrap: bool = True, name: str = "Dim") -> QLabel:
    lab = QLabel(text)
    lab.setObjectName(name)
    lab.setWordWrap(wrap)
    lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
    if wrap:  # dlouhý text nesmí roztáhnout panel (jinak se obsah ořízne)
        lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lab.setMinimumWidth(50)
    return lab


def panel_title(text: str) -> QLabel:
    lab = QLabel(text.upper())
    lab.setObjectName("PanelTitle")
    return lab
