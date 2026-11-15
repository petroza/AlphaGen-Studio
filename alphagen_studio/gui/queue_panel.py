"""Záložka Queue – dávková fronta jobů na noc."""

from __future__ import annotations

import subprocess

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..jobqueue import CANCELLED, DONE, FAILED, RUNNING, WAITING, JobQueue
from . import theme

STATUS_TEXT = {WAITING: "Waiting", RUNNING: "Running…", DONE: "Done", FAILED: "Failed", CANCELLED: "Stopped"}
STATUS_COLOR = {WAITING: theme.TEXT_DIM, RUNNING: theme.ACCENT, DONE: theme.OK, FAILED: theme.ERR,
                CANCELLED: theme.WARN}


class QueuePanel(QWidget):
    addCurrentRequested = Signal()  # noqa: N815
    addFilesRequested = Signal()  # noqa: N815
    startRequested = Signal()  # noqa: N815
    stopRequested = Signal()  # noqa: N815

    def __init__(self, queue: JobQueue) -> None:
        super().__init__()
        self.queue = queue
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 6)

        bar = QHBoxLayout()
        self.add_cur = QPushButton("+ Current clip")
        self.add_cur.setToolTip("Add the loaded clip with the current IN/OUT and settings")
        self.add_files = QPushButton("+ Files…")
        self.add_files.setToolTip("Add several videos at once (full length). In Automatic mode each clip gets\n"
                                  "settings matched to its own resolution and your GPU.")
        self.up = QPushButton("▲")
        self.down = QPushButton("▼")
        self.remove = QPushButton("Remove")
        self.retry = QPushButton("Retry failed")
        self.clear = QPushButton("Clear done")
        for b in (self.up, self.down):
            b.setFixedWidth(32)
        for b in (self.add_cur, self.add_files, self.up, self.down, self.remove, self.retry, self.clear):
            bar.addWidget(b)
        bar.addStretch(1)
        self.shutdown = QCheckBox("Shut down PC when the queue finishes")
        self.shutdown.setToolTip("Windows shuts down 2 minutes after the last job (can be cancelled with "
                                 "'shutdown /a' in a command prompt).")
        bar.addWidget(self.shutdown)
        self.start = QPushButton("▶  Run queue")
        self.start.setObjectName("Primary")
        self.stop = QPushButton("■  Stop queue")
        self.stop.setObjectName("Danger")
        self.stop.setEnabled(False)
        bar.addWidget(self.start)
        bar.addWidget(self.stop)
        lay.addLayout(bar)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "Clip", "Range", "Settings", "Status", "Info"])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.table, 1)
        self.summary = QLabel("")
        self.summary.setObjectName("Dim")
        lay.addWidget(self.summary)

        self.add_cur.clicked.connect(self.addCurrentRequested.emit)
        self.add_files.clicked.connect(self.addFilesRequested.emit)
        self.start.clicked.connect(self.startRequested.emit)
        self.stop.clicked.connect(self.stopRequested.emit)
        self.remove.clicked.connect(lambda: (self.queue.remove(self.selected_ids()), self.refresh()))
        self.up.clicked.connect(lambda: self._move(-1))
        self.down.clicked.connect(lambda: self._move(1))
        self.retry.clicked.connect(lambda: (self.queue.retry_failed(), self.refresh()))
        self.clear.clicked.connect(lambda: (self.queue.clear_finished(), self.refresh()))
        self.refresh()

    # ------------------------------------------------------------------
    def selected_ids(self) -> list[str]:
        rows = {i.row() for i in self.table.selectedIndexes()}
        return [self.queue.items[r].id for r in sorted(rows) if r < len(self.queue.items)]

    def _move(self, delta: int) -> None:
        ids = self.selected_ids()
        if not ids:
            return
        self.queue.move(ids[0], delta)
        self.refresh()
        row = next((n for n, i in enumerate(self.queue.items) if i.id == ids[0]), 0)
        self.table.selectRow(row)

    def set_running(self, running: bool) -> None:
        self.start.setEnabled(not running)
        self.stop.setEnabled(running)

    def refresh(self) -> None:
        items = list(self.queue.items)
        self.table.setRowCount(len(items))
        for r, it in enumerate(items):
            js = it.settings
            if it.profile:
                sett = f"Automatic · {it.profile.capitalize()}"
            else:
                sett = (f"{js.get('long_side') or 'full'} px · seg {js.get('segment_frames')} · "
                        f"{', '.join(js.get('export', {}).get('formats', []))}")
            info = it.message
            if it.duration_text():
                info = f"{it.duration_text()}  {info}"
            vals = [str(r + 1), it.name, it.range_text(), sett, STATUS_TEXT.get(it.status, it.status), info]
            for c, v in enumerate(vals):
                cell = QTableWidgetItem(v)
                if c == 1:
                    cell.setToolTip(it.input_path)
                if c == 4:
                    cell.setForeground(QColor(STATUS_COLOR.get(it.status, theme.TEXT)))
                if c == 5 and it.outputs:
                    cell.setToolTip("\n".join(f"{k}: {v}" for k, v in it.outputs.items()))
                cell.setTextAlignment(Qt.AlignVCenter | (Qt.AlignLeft if c in (1, 5) else Qt.AlignCenter))
                self.table.setItem(r, c, cell)
        n = self.queue.counts()
        self.summary.setText(f"{len(items)} job(s): {n[WAITING]} waiting, {n[RUNNING]} running, {n[DONE]} done, "
                             f"{n[FAILED]} failed, {n[CANCELLED]} stopped. The queue is saved – it survives a restart.")

    @staticmethod
    def schedule_shutdown() -> None:
        subprocess.run(["shutdown", "/s", "/t", "120", "/c", "AlphaGen Studio: queue finished"],  # noqa: S603, S607
                       check=False, creationflags=0x08000000)
