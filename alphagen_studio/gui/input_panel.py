"""Levý panel – vstup: drag & drop, výběr souboru/sekvence, informace, IN/OUT."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QSizePolicy,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..video_io import IMAGE_EXTS, VIDEO_EXTS, VideoInfo, format_tc
from .widgets import dim_label, panel_title


class DropArea(QLabel):
    dropped = Signal(str)

    def __init__(self) -> None:
        super().__init__("Drop a video here\nor a PNG sequence folder / frame")
        self.setObjectName("Drop")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(90)

    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setProperty("hover", True)
            self.style().polish(self)

    def dragLeaveEvent(self, e) -> None:  # noqa: N802
        self.setProperty("hover", False)
        self.style().polish(self)

    def dropEvent(self, e) -> None:  # noqa: N802
        self.setProperty("hover", False)
        self.style().polish(self)
        urls = e.mimeData().urls()
        if urls:
            self.dropped.emit(urls[0].toLocalFile())


class InputPanel(QWidget):
    fileChosen = Signal(str)  # noqa: N815
    rangeChanged = Signal(int, int)  # noqa: N815
    setInRequested = Signal()  # noqa: N815
    setOutRequested = Signal()  # noqa: N815

    def __init__(self, settings) -> None:
        super().__init__()
        self.settings = settings
        self.setObjectName("Panel")
        self.info: VideoInfo | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 6, 14, 12)
        lay.addWidget(panel_title("Input"))

        self.drop = DropArea()
        self.drop.dropped.connect(self.fileChosen.emit)
        lay.addWidget(self.drop)
        row = QHBoxLayout()
        b1 = QPushButton("Video…")
        b1.setToolTip("Choose a video file")
        b1.clicked.connect(self._pick_file)
        b2 = QPushButton("Sequence…")
        b2.setToolTip("Pick any frame of the sequence (e.g. shot_0001.png)")
        b2.clicked.connect(self._pick_sequence)
        row.addWidget(b1)
        row.addWidget(b2)
        lay.addLayout(row)

        g = QGroupBox("Clip info")
        gl = QVBoxLayout(g)
        self.name_label = QLabel("–")
        self.name_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.name_label.setMinimumWidth(50)
        self.name_label.setStyleSheet("font-weight:600;")
        gl.addWidget(self.name_label)
        self.info_label = dim_label("No clip loaded.")
        gl.addWidget(self.info_label)
        self.warn_label = dim_label("", name="Warn")
        gl.addWidget(self.warn_label)
        lay.addWidget(g)

        r = QGroupBox("Range (IN / OUT)")
        rl = QGridLayout(r)
        self.in_spin = QSpinBox()
        self.out_spin = QSpinBox()
        for s in (self.in_spin, self.out_spin):
            s.setRange(0, 0)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(self._range_edit)
        rl.addWidget(QLabel("IN"), 0, 0)
        rl.addWidget(self.in_spin, 0, 1)
        bin_ = QPushButton("◂")
        bin_.setToolTip("Set IN to the current viewer frame (I key)")
        bin_.clicked.connect(self.setInRequested.emit)
        bin_.setFixedWidth(34)
        rl.addWidget(bin_, 0, 2)
        rl.addWidget(QLabel("OUT"), 1, 0)
        rl.addWidget(self.out_spin, 1, 1)
        bout = QPushButton("◂")
        bout.setToolTip("Set OUT to the current viewer frame (O key)")
        bout.clicked.connect(self.setOutRequested.emit)
        bout.setFixedWidth(34)
        rl.addWidget(bout, 1, 2)
        breset = QPushButton("Full clip")
        breset.clicked.connect(self.reset_range)
        rl.addWidget(breset, 2, 0, 1, 3)
        self.range_label = dim_label("")
        rl.addWidget(self.range_label, 3, 0, 1, 3)
        lay.addWidget(r)
        lay.addStretch(1)

    # ------------------------------------------------------------------
    def _start_dir(self) -> str:
        return self.settings.last_input_dir or str(Path.home())

    def _pick_file(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))
        path, _ = QFileDialog.getOpenFileName(self, "Choose video", self._start_dir(),
                                              f"Video ({exts});;All files (*.*)")
        if path:
            self.fileChosen.emit(path)

    def _pick_sequence(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        path, _ = QFileDialog.getOpenFileName(self, "Choose sequence frame", self._start_dir(),
                                              f"Images ({exts})")
        if path:
            self.fileChosen.emit(path)

    def set_info(self, info: VideoInfo) -> None:
        self.info = info
        name = Path(info.seq_pattern or info.path).name
        self.name_label.setText(name if len(name) <= 34 else name[:20] + "…" + name[-12:])
        self.name_label.setToolTip(name)
        self.info_label.setText(info.summary_cs())
        self.warn_label.setText("\n".join("⚠ " + w for w in info.warnings))
        for s in (self.in_spin, self.out_spin):
            s.blockSignals(True)
            s.setRange(0, info.nb_frames - 1)
        self.in_spin.setValue(0)
        self.out_spin.setValue(info.nb_frames - 1)
        for s in (self.in_spin, self.out_spin):
            s.blockSignals(False)
        self._update_range_label()

    def reset_range(self) -> None:
        if self.info:
            self.set_range(0, self.info.nb_frames - 1)

    def set_range(self, a: int, b: int) -> None:
        self.in_spin.blockSignals(True)
        self.out_spin.blockSignals(True)
        self.in_spin.setValue(a)
        self.out_spin.setValue(b)
        self.in_spin.blockSignals(False)
        self.out_spin.blockSignals(False)
        self._range_edit()

    def _range_edit(self) -> None:
        a, b = self.in_spin.value(), self.out_spin.value()
        if b < a:
            self.out_spin.blockSignals(True)
            self.out_spin.setValue(a)
            self.out_spin.blockSignals(False)
            b = a
        self._update_range_label()
        self.rangeChanged.emit(a, b)

    def _update_range_label(self) -> None:
        if not self.info:
            return
        a, b = self.in_spin.value(), self.out_spin.value()
        n = b - a + 1
        self.range_label.setText(
            f"{n} frames · {format_tc(self.info.frame_to_time(a))} → {format_tc(self.info.frame_to_time(b + 1))}"
            f" · duration {format_tc(n / self.info.fps_float)}"
        )

    def range(self) -> tuple[int, int]:
        """(IN, OUT) – OUT inkluzivní."""
        return self.in_spin.value(), self.out_spin.value()
