"""Spodní panel – start/cancel, průběh, VRAM/RAM, ETA, log."""

from __future__ import annotations

from PySide6.QtGui import QColor, QTextCharFormat
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import theme

LEVEL_COLORS = {"ERROR": theme.ERR, "CRITICAL": theme.ERR, "WARNING": theme.WARN, "PROGRESS": theme.TEXT_DIM,
                "INFO": theme.TEXT, "OK": theme.OK}


def fmt_eta(sec: float | None) -> str:
    if sec is None:
        return "– (measuring)"
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class BottomPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Panel")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 10)

        left = QVBoxLayout()
        self.start_btn = QPushButton("▶  Start processing")
        self.start_btn.setObjectName("Primary")
        self.start_btn.setToolTip("AlphaGen inference (resumes from finished segments) + export")
        self.quick_btn = QPushButton("⚡  Quick test (2 s)")
        self.quick_btn.setToolTip("Turbo on ~2 seconds around the current frame (~30–60 s). Shows what the model "
                                  "picks as the subject – Before / After in the viewer; only a temporary matte goes to the cache.")
        self.export_btn = QPushButton("Export only")
        self.export_btn.setToolTip("Re-export with the current matte refinement – no new inference")
        self.cancel_btn = QPushButton("■  Cancel")
        self.cancel_btn.setObjectName("Danger")
        self.cancel_btn.setEnabled(False)
        for b in (self.start_btn, self.quick_btn, self.export_btn, self.cancel_btn):
            b.setMinimumWidth(170)
            left.addWidget(b)
        left.addStretch(1)
        lay.addLayout(left)

        mid = QGridLayout()
        mid.setHorizontalSpacing(10)
        self.stage = QLabel("Ready.")
        self.stage.setStyleSheet("font-weight:600;")
        mid.addWidget(self.stage, 0, 0, 1, 4)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        mid.addWidget(self.progress, 1, 0, 1, 4)
        self.detail = QLabel("")
        self.detail.setObjectName("Dim")
        mid.addWidget(self.detail, 2, 0, 1, 2)
        self.eta = QLabel("Remaining: –")
        mid.addWidget(self.eta, 2, 2, 1, 2)
        mid.addWidget(QLabel("VRAM"), 3, 0)
        self.vram = QProgressBar()
        self.vram.setObjectName("Vram")
        self.vram.setRange(0, 1000)
        self.vram.setFormat("–")
        mid.addWidget(self.vram, 3, 1, 1, 3)
        mid.addWidget(QLabel("RAM"), 4, 0)
        self.ram = QProgressBar()
        self.ram.setObjectName("Vram")
        self.ram.setRange(0, 1000)
        self.ram.setFormat("–")
        mid.addWidget(self.ram, 4, 1, 1, 3)
        midw = QWidget()
        midw.setLayout(mid)
        midw.setMinimumWidth(380)
        midw.setMaximumWidth(520)
        lay.addWidget(midw)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(6000)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.log, "Log")
        lay.addWidget(self.tabs, 1)

    def add_queue_tab(self, widget) -> None:
        self.queue_tab = self.tabs.addTab(widget, "Queue")

    def set_queue_title(self, text: str) -> None:
        self.tabs.setTabText(self.queue_tab, text)

    # ------------------------------------------------------------------
    def set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.quick_btn.setEnabled(not running)
        self.export_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    def append_log(self, level: str, msg: str) -> None:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(LEVEL_COLORS.get(level, theme.TEXT)))
        cur = self.log.textCursor()
        cur.movePosition(cur.MoveOperation.End)
        cur.insertText(msg + "\n", fmt)
        self.log.setTextCursor(cur)
        self.log.ensureCursorVisible()

    def set_progress(self, done: int, total: int, eta: float | None, detail: str) -> None:
        self.progress.setValue(int(1000 * done / max(1, total)))
        self.progress.setFormat(f"{done} / {total}  (%p %)")
        self.detail.setText(detail)
        self.eta.setText(f"Remaining: {fmt_eta(eta)}")

    def set_vram(self, used_mb: int, total_mb: int) -> None:
        self.vram.setValue(int(1000 * used_mb / max(1, total_mb)))
        self.vram.setFormat(f"{used_mb / 1024:.1f} / {total_mb / 1024:.1f} GB")

    def set_ram(self, avail_gb: float, total_gb: float) -> None:
        used = total_gb - avail_gb
        self.ram.setValue(int(1000 * used / max(0.1, total_gb)))
        self.ram.setFormat(f"{used:.1f} / {total_gb:.1f} GB")
