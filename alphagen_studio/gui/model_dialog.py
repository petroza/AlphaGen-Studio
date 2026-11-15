"""Správce modelů – stav, stažení z oficiálních zdrojů, ověření úplnosti."""

from __future__ import annotations

import threading
import webbrowser

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..models import ALPHA_REPO, BASE_REPO, HF, ModelError, ModelManager
from .widgets import dim_label


class _Bridge(QObject):
    progress = Signal(str, object, object, float)
    status = Signal(str)
    done = Signal(bool, str)


class ModelDialog(QDialog):
    def __init__(self, manager: ModelManager, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model Manager – LTX-2.5 AlphaGen")
        self.resize(980, 520)
        self.mm = manager
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.bridge = _Bridge()
        self.bridge.progress.connect(self._on_progress)
        self.bridge.status.connect(lambda s: self.status.setText(s))
        self.bridge.done.connect(self._on_done)

        lay = QVBoxLayout(self)
        lay.addWidget(dim_label(
            "The official Lightricks files are required (~67.5 GB in total). The repositories are gated: sign in to "
            "Hugging Face, open both pages below, accept the LTX-2.x Community License and create a token "
            "with read access (Settings → Access Tokens). Once downloaded, inference runs fully locally."))
        links = QHBoxLayout()
        for repo in (BASE_REPO, ALPHA_REPO):
            b = QPushButton(f"Open {repo}")
            b.clicked.connect(lambda _=False, r=repo: webbrowser.open(f"{HF}/{r}"))
            links.addWidget(b)
        b = QPushButton("Create token…")
        b.clicked.connect(lambda: webbrowser.open(f"{HF}/settings/tokens"))
        links.addWidget(b)
        lay.addLayout(links)

        row = QHBoxLayout()
        row.addWidget(QLabel("HF token"))
        self.token = QLineEdit(self.mm.settings.extra.get("hf_token", ""))
        self.token.setEchoMode(QLineEdit.Password)
        self.token.setPlaceholderText("hf_…  (enter it yourself; it is sent only to huggingface.co)")
        row.addWidget(self.token, 1)
        self.remember = QCheckBox("Remember (saved to config/settings.json)")
        self.remember.setChecked(bool(self.mm.settings.extra.get("hf_token")))
        row.addWidget(self.remember)
        lay.addLayout(row)

        drow = QHBoxLayout()
        drow.addWidget(QLabel("Models folder"))
        self.models_dir = QLineEdit(self.mm.settings.models_dir)
        drow.addWidget(self.models_dir, 1)
        b = QPushButton("…")
        b.clicked.connect(self._pick_dir)
        drow.addWidget(b)
        lay.addLayout(drow)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Component", "Size", "Status", "Path"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table, 1)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        lay.addWidget(self.bar)
        self.status = QLabel("")
        lay.addWidget(self.status)

        btns = QHBoxLayout()
        self.dl_btn = QPushButton("Download missing")
        self.dl_btn.setObjectName("Primary")
        self.dl_btn.clicked.connect(self._download)
        self.find_btn = QPushButton("Find existing files…")
        self.find_btn.clicked.connect(self._find)
        self.conv_btn = QPushButton("Download lightweight VAE (1.4 GB)")
        self.conv_btn.setToolTip("Official convolutional video VAE from Lightricks/LTX-2.5 – enables Full HD on 12–24 GB cards.")
        self.conv_btn.clicked.connect(self._download_conv)
        self.sha_btn = QPushButton("Verify SHA256")
        self.sha_btn.clicked.connect(self._verify)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.cancel.set)
        self.stop_btn.setEnabled(False)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        for b in (self.dl_btn, self.conv_btn, self.find_btn, self.sha_btn, self.stop_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(close)
        lay.addLayout(btns)
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        st = self.mm.status()
        self.table.setRowCount(len(st))
        for r, s in enumerate(st):
            if s["size_ok"]:
                state = "✔ OK"
            elif s["partial"]:
                state = f"partial {s['partial'] / max(1, s['expected']) * 100:.0f}%"
            elif s["exists"]:
                state = "✖ wrong size"
            else:
                state = "missing"
            vals = [s["label"], f"{s['expected'] / 1024**3:.2f} GB", state, s["path"]]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 2:
                    it.setForeground(Qt.green if s["size_ok"] else Qt.yellow)
                self.table.setItem(r, c, it)
        self.conv_btn.setText("Lightweight VAE: ✔ downloaded" if self.mm.has_conv_vae() else "Download lightweight VAE (1.4 GB)")
        self.conv_btn.setEnabled(not self.mm.has_conv_vae())
        missing = self.mm.total_missing_bytes()
        self.status.setText("All models are ready." if not missing else
                            f"{missing / 1024**3:.1f} GB left to download.")

    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Models folder", self.models_dir.text())
        if d:
            self.models_dir.setText(d)
            self.mm.settings.models_dir = d
            self.mm.settings.save()
            self.refresh()

    def _save_token(self) -> str:
        tok = self.token.text().strip()
        if self.remember.isChecked() and tok:
            self.mm.settings.extra["hf_token"] = tok
        else:
            self.mm.settings.extra.pop("hf_token", None)
        self.mm.settings.models_dir = self.models_dir.text().strip() or self.mm.settings.models_dir
        self.mm.settings.save()
        return tok

    def _busy(self, on: bool) -> None:
        for b in (self.dl_btn, self.conv_btn, self.find_btn, self.sha_btn):
            b.setEnabled(not on)
        self.stop_btn.setEnabled(on)

    def _download(self) -> None:
        tok = self._save_token()
        if not tok:
            QMessageBox.warning(self, "HF token", "The Lightricks repositories are gated – enter a Hugging Face token.")
            return
        self.cancel.clear()
        self._busy(True)

        def work():
            try:
                self.bridge.status.emit("Fetching the file list from Hugging Face…")
                try:
                    self.mm.refresh_manifest(tok)
                except Exception as exc:  # noqa: BLE001 - manifest je volitelný (SHA)
                    self.bridge.status.emit(f"Could not load the manifest: {exc}")
                todo = list(self.mm.missing())
                if not self.mm.has_conv_vae():  # úsporný VAE: Full HD na 12–24 GB kartách (Automatic ho použije)
                    todo.append(self.mm.optional[0])
                for f in todo:
                    self.bridge.status.emit(f"Downloading {f.filename} …")
                    self.mm.download(f, tok, lambda k, d, t, sp: self.bridge.progress.emit(k, d, t, sp), self.cancel)
                self.bridge.done.emit(True, "Download complete.")
            except ModelError as exc:
                self.bridge.done.emit(False, str(exc))
            except Exception as exc:  # noqa: BLE001
                self.bridge.done.emit(False, f"{type(exc).__name__}: {exc}")

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def _download_conv(self) -> None:
        tok = self._save_token()
        self.cancel.clear()
        self._busy(True)

        def work():
            try:
                f = self.mm.optional[0]
                self.bridge.status.emit(f"Downloading {f.filename} …")
                self.mm.download(f, tok or None, lambda k, d, t, sp: self.bridge.progress.emit(k, d, t, sp), self.cancel)
                self.bridge.done.emit(True, "Lightweight VAE downloaded.")
            except Exception as exc:  # noqa: BLE001
                self.bridge.done.emit(False, str(exc))

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def _verify(self) -> None:
        tok = self._save_token()
        self.cancel.clear()
        self._busy(True)

        def work():
            try:
                self.mm.refresh_manifest(tok or None)
                bad = []
                for f in self.mm.files:
                    if not self.mm.path_for(f).exists():
                        bad.append(f"{f.filename}: missing")
                        continue
                    self.bridge.status.emit(f"Verifying SHA256 {f.filename} …")
                    ok = self.mm.verify_sha256(f, lambda d, t, k=f.key: self.bridge.progress.emit(k, d, t, 0.0),
                                               self.cancel)
                    if not ok:
                        bad.append(f"{f.filename}: checksum MISMATCH")
                self.bridge.done.emit(not bad, "All files match the official SHA256." if not bad
                                      else "\n".join(bad))
            except Exception as exc:  # noqa: BLE001
                self.bridge.done.emit(False, str(exc))

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def _find(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Where are your LTX-2.5 models already downloaded?")
        if not d:
            return
        found = self.mm.find_existing(d)
        QMessageBox.information(self, "Found files",
                                "\n".join(found.values()) if found else "No matching files found.")
        self.refresh()

    def _on_progress(self, key: str, done, total, speed: float) -> None:
        self.bar.setValue(int(1000 * done / max(1, total)))
        sp = f", {speed / 1024**2:.1f} MB/s" if speed else ""
        self.bar.setFormat(f"{key}: {done / 1024**3:.2f} / {total / 1024**3:.2f} GB{sp}")

    def _on_done(self, ok: bool, msg: str) -> None:
        self._busy(False)
        self.refresh()
        (QMessageBox.information if ok else QMessageBox.warning)(self, "Models", msg)
