"""Hlavní okno AlphaGen Studio."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from PySide6.QtCore import QByteArray, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, __version__
from ..backend.client import check_environment
from ..config import LOG_DIR, AppSettings
from ..engine import Engine, JobListener, JobSettings
from ..gpu import gpu_memory, list_gpus, ram_info
from ..logging_setup import LOGGER_NAME, CallbackHandler, get_logger
from ..models import ModelManager
from ..video_io import VideoError, normalize_vfr, probe
from ..jobqueue import JobQueue, QueueItem, QueueRunner
from .bottom_panel import BottomPanel
from .queue_panel import QueuePanel
from .input_panel import InputPanel
from .model_dialog import ModelDialog
from .preview import PreviewPanel
from .settings_panel import SettingsPanel

log = get_logger("gui")


class QtListener(QObject, JobListener):
    sig_log = Signal(str, str)
    sig_stage = Signal(str)
    sig_progress = Signal(int, int, object, str)
    sig_segment = Signal(int, int)
    sig_finished = Signal(dict)
    sig_failed = Signal(str, str, list)

    def log(self, level, msg):
        self.sig_log.emit(level, msg)
        if level != "PROGRESS":  # do souboru logs/alphagen.log
            log.log({"ERROR": 40, "WARNING": 30}.get(level, 20), msg)

    def stage(self, text):
        self.sig_stage.emit(text)
        log.info(text)

    def progress(self, done, total, eta_s, detail):
        self.sig_progress.emit(done, total, eta_s, detail)

    def segment_done(self, index, total):
        self.sig_segment.emit(index, total)

    def finished(self, report):
        self.sig_finished.emit(report)

    def failed(self, message, kind, suggestions):
        self.sig_failed.emit(message, kind, suggestions)


class MainWindow(QMainWindow):
    sig_info_loaded = Signal(object, str)
    sig_info_error = Signal(str)
    sig_mem = Signal(object, float, float)
    sig_env = Signal(dict)
    sig_log = Signal(str, str)
    sig_queue_changed = Signal()
    sig_queue_done = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings.load()
        self.settings.ensure_dirs()
        self.setWindowTitle(f"{APP_NAME} {__version__} – LTX-2.5 AlphaGen")
        self.resize(1600, 980)
        self.engine = Engine(self.settings)
        self.models = self.engine.models
        self.gpus = list_gpus()
        self.engine.gpu_total_gb = self.gpus[0].total_gb if self.gpus else 0.0
        self.ram_total = ram_info()[1]
        self.input_path: str | None = None
        self.info = None
        self.running = False

        self.input_panel = InputPanel(self.settings)
        self.preview = PreviewPanel()
        self.settings_panel = SettingsPanel(self.settings, self.gpus, self.ram_total)
        self.settings_panel.estimator = lambda geo, segs, vae: self.engine.estimate_seconds(
            geo, segs, vae, self.settings_panel.gpu_index())
        self.bottom = BottomPanel()
        self.queue = JobQueue()
        self.queue_panel = QueuePanel(self.queue)
        self.bottom.add_queue_tab(self.queue_panel)
        self.queue_runner: QueueRunner | None = None

        def scroll(w: QWidget, min_w: int) -> QScrollArea:
            sa = QScrollArea()
            sa.setObjectName("Panel")
            sa.setWidget(w)
            sa.setWidgetResizable(True)
            sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sa.setMinimumWidth(min_w)
            return sa

        top = QSplitter(Qt.Horizontal)
        top.addWidget(scroll(self.input_panel, 300))
        top.addWidget(self.preview)
        top.addWidget(scroll(self.settings_panel, 380))
        top.setSizes([320, 900, 400])
        top.setCollapsible(1, False)
        top.setStretchFactor(1, 1)
        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(top)
        vsplit.addWidget(self.bottom)
        vsplit.setSizes([760, 220])
        self.splitters = (top, vsplit)
        c = QWidget()
        lay = QVBoxLayout(c)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(vsplit)
        self.setCentralWidget(c)

        self.listener = QtListener()
        self._build_menu()
        self._connect()
        self._shortcuts()
        self._restore_ui()

        # log z celé aplikace do spodního panelu
        self.sig_log.connect(self.bottom.append_log)
        # varování/chyby z modulů (ne z GUI – ty už v panelu jsou) do spodního logu
        handler = CallbackHandler(lambda lvl, msg: self.sig_log.emit(lvl, msg))
        handler.setLevel(logging.WARNING)
        handler.addFilter(lambda rec: not rec.name.startswith("alphagen.gui"))
        logging.getLogger(LOGGER_NAME).addHandler(handler)

        self.mem_timer = QTimer(self)
        self.mem_timer.timeout.connect(self._poll_memory)
        self.mem_timer.start(2000)
        self._poll_memory()
        QTimer.singleShot(300, self._startup_checks)

    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        a = QAction("Open Video…", self)
        a.setShortcut(QKeySequence.Open)
        a.triggered.connect(self.input_panel._pick_file)
        m.addAction(a)
        a = QAction("Open PNG Sequence…", self)
        a.triggered.connect(self.input_panel._pick_sequence)
        m.addAction(a)
        m.addSeparator()
        a = QAction("Open Output Folder", self)
        a.triggered.connect(lambda: os.startfile(self.settings_panel.output_dir()))  # noqa: S606
        m.addAction(a)
        m.addSeparator()
        a = QAction("Exit", self)
        a.triggered.connect(self.close)
        m.addAction(a)
        m = mb.addMenu("&Models")
        a = QAction("Model Manager…", self)
        a.triggered.connect(self.open_models)
        m.addAction(a)
        a = QAction("Apply GPU Recommendation", self)
        a.triggered.connect(self.settings_panel.apply_recommendation)
        m.addAction(a)
        m = mb.addMenu("&Tools")
        a = QAction("Environment Diagnostics", self)
        a.triggered.connect(self._startup_checks)
        m.addAction(a)
        a = QAction("Open Log Folder", self)
        a.triggered.connect(lambda: os.startfile(str(LOG_DIR)))  # noqa: S606
        m.addAction(a)
        a = QAction("Open Work Folder (Segment Cache)", self)
        a.triggered.connect(lambda: os.startfile(self.settings.work_dir))  # noqa: S606
        m.addAction(a)
        a = QAction("Stop Inference Worker (Free VRAM)", self)
        a.triggered.connect(self._stop_worker)
        m.addAction(a)
        m = mb.addMenu("&Help")
        a = QAction("About", self)
        a.triggered.connect(self._about)
        m.addAction(a)

    def _connect(self) -> None:
        self.input_panel.fileChosen.connect(self.open_input)
        self.input_panel.rangeChanged.connect(self._range_changed)
        self.input_panel.setInRequested.connect(lambda: self._set_in_out(True))
        self.input_panel.setOutRequested.connect(lambda: self._set_in_out(False))
        self.settings_panel.matteChanged.connect(self.preview.set_matte_settings)
        self.settings_panel.planChanged.connect(self._update_cache_link)
        self.bottom.start_btn.clicked.connect(lambda: self.start(False))
        self.bottom.export_btn.clicked.connect(lambda: self.start(True))
        self.bottom.cancel_btn.clicked.connect(self.cancel)
        L = self.listener  # noqa: N806
        L.sig_log.connect(self.bottom.append_log)
        L.sig_stage.connect(self._on_stage)
        L.sig_progress.connect(self.bottom.set_progress)
        L.sig_segment.connect(lambda i, n: self.preview.refresh_ready() or self.preview.refresh())
        L.sig_finished.connect(self._on_finished)
        L.sig_failed.connect(self._on_failed)
        self.sig_info_loaded.connect(self._on_info)
        self.sig_info_error.connect(lambda m: QMessageBox.warning(self, "Invalid video", m))
        self.sig_mem.connect(self._on_mem)
        self.sig_env.connect(self._on_env)
        qp = self.queue_panel
        qp.addCurrentRequested.connect(self._queue_add_current)
        qp.addFilesRequested.connect(self._queue_add_files)
        qp.startRequested.connect(self._queue_start)
        qp.stopRequested.connect(self._queue_stop)
        self.sig_queue_changed.connect(self._queue_changed)
        self.sig_queue_done.connect(self._queue_done)
        self._queue_changed()

    def _shortcuts(self) -> None:
        def sc(key, fn):
            s = QShortcut(QKeySequence(key), self)
            s.setContext(Qt.ApplicationShortcut)
            s.activated.connect(fn)

        sc(Qt.Key_Space, self.preview.toggle_play)
        sc(Qt.Key_Left, lambda: self.preview.step(-1))
        sc(Qt.Key_Right, lambda: self.preview.step(1))
        sc(Qt.SHIFT | Qt.Key_Left, lambda: self.preview.step(-10))
        sc(Qt.SHIFT | Qt.Key_Right, lambda: self.preview.step(10))
        sc(Qt.Key_I, lambda: self._set_in_out(True))
        sc(Qt.Key_O, lambda: self._set_in_out(False))
        for i in range(5):
            sc(Qt.Key_1 + i, lambda i=i: self.preview.set_mode_index(i))

    # ------------------------------------------------------------------
    def _startup_checks(self) -> None:
        ff = self.settings.ffmpeg_path()
        self.bottom.append_log("INFO" if ff else "ERROR", f"FFmpeg: {ff or 'NOT FOUND – run setup.bat'}")
        if not self.gpus:
            self.bottom.append_log("ERROR", "No NVIDIA GPU with CUDA found – AlphaGen inference will not run.")
        for g in self.gpus:
            self.bottom.append_log("INFO", f"GPU {g.label()}, driver {g.driver}")
        avail, total = ram_info()
        self.bottom.append_log("INFO", f"RAM: {total:.0f} GB ({avail:.0f} GB free)")
        threading.Thread(target=lambda: self.sig_env.emit(check_environment(self.settings.ltx_python_path())),
                         daemon=True).start()
        missing = self.models.missing()
        if missing:
            self.bottom.append_log("WARNING", f"{len(missing)} model file(s) missing "
                                              f"({self.models.total_missing_bytes() / 1024**3:.1f} GB).")

    def _on_env(self, env: dict) -> None:
        if env.get("ok"):
            self.bottom.append_log("INFO", f"Inference environment OK: Python {env.get('python')}, torch {env.get('torch')}, "
                                           f"CUDA {env.get('cuda')}, ltx_pipelines ✔, GPU {env.get('gpus')}")
        else:
            self.bottom.append_log("ERROR", f"Inference environment not ready: {env}")
        if self.models.missing() and not getattr(self, "_asked_models", False):
            self._asked_models = True
            r = QMessageBox.question(
                self, "AlphaGen Models",
                "The official LTX-2.5 AlphaGen model files are missing (~67.5 GB from Hugging Face).\n"
                "Open the Model Manager and download them?")
            if r == QMessageBox.Yes:
                self.open_models()

    def open_models(self) -> None:
        dlg = ModelDialog(self.models, self)
        dlg.exec()
        self.engine.models = ModelManager(self.settings)
        self.models = self.engine.models

    def _poll_memory(self) -> None:
        idx = self.settings_panel.gpu_index()

        def work():
            mem = gpu_memory(idx)
            avail, total = ram_info()
            self.sig_mem.emit(mem, avail, total)

        threading.Thread(target=work, daemon=True).start()

    def _on_mem(self, mem, avail: float, total: float) -> None:
        if mem:
            self.bottom.set_vram(*mem)
        self.bottom.set_ram(avail, total)

    # ------------------------------------------------------------------
    def open_input(self, path: str) -> None:
        if self.running:
            QMessageBox.information(self, APP_NAME, "The input can't be changed while processing.")
            return
        self.bottom.stage.setText(f"Loading {Path(path).name} …")
        self.settings.last_input_dir = str(Path(path).parent)
        self.settings.save()

        def work():
            try:
                info = probe(path, self.settings.ffprobe_path())
                src = path
                if info.is_vfr:
                    self.listener.stage("Variable frame rate video – converting to constant FPS (lossless)…")
                    cfr = Path(self.settings.cache_dir) / f"cfr_{abs(hash((path, os.path.getmtime(path))))}.mkv"
                    if not cfr.exists():
                        normalize_vfr(info, self.settings.ffmpeg_path(), cfr)
                    src = str(cfr)
                    info = probe(src, self.settings.ffprobe_path())
                self.sig_info_loaded.emit(info, src)
            except (VideoError, OSError) as exc:
                self.sig_info_error.emit(str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _on_info(self, info, src: str) -> None:
        self.info = info
        self.input_path = src
        self.input_panel.set_info(info)
        self.preview.set_source(info, self.settings.ffmpeg_path())
        self.settings_panel.set_source(info)
        self.settings_panel.set_range(0, info.nb_frames - 1)
        self.preview.set_range(0, info.nb_frames - 1)
        self.bottom.stage.setText(f"Loaded: {Path(info.seq_pattern or info.path).name}")
        for w in info.warnings:
            self.bottom.append_log("WARNING", w)
        self._update_cache_link()

    def _range_changed(self, a: int, b: int) -> None:
        self.preview.set_range(a, b)
        self.settings_panel.set_range(a, b)

    def _set_in_out(self, is_in: bool) -> None:
        if not self.info:
            return
        a, b = self.input_panel.range()
        f = self.preview.frame
        self.input_panel.set_range(f, max(f, b)) if is_in else self.input_panel.set_range(min(a, f), f)

    def job_settings(self) -> JobSettings:
        sp = self.settings_panel
        a, b = self.input_panel.range()
        return JobSettings(
            in_frame=a, out_frame=b + 1, long_side=sp.resolution.currentData() or 0,
            segment_frames=sp.seg_len.value(), overlap=sp.overlap.value(), seed=sp.seed.value(),
            gpu_index=sp.gpu_index(), quantization=sp.quant.currentData(), offload=sp.offload.currentData(),
            vae=sp.vae.currentData(),
            matte=sp.matte_settings(), export=sp.export_settings(), output_dir=sp.output_dir(),
            base_name=sp.base_name.text().strip(),
        )

    def _update_cache_link(self) -> None:
        if not self.info or self.running:
            return
        js = self.job_settings()
        cache = self.engine.find_cache(self.info, js)
        self.preview.set_cache(cache, js.in_frame)
        if cache is not None:
            n = int(cache.ready_mask().sum())
            if n:
                self.bottom.append_log("INFO", f"Found a cached matte for these settings: {n}/{cache.total} "
                                               "frames done (processing will resume).")

    # ------------------------------------------------------------------
    def start(self, export_only: bool) -> None:
        if self._queue_running():
            QMessageBox.information(self, APP_NAME, "The queue is running – stop it first.")
            return
        if not self.info or not self.input_path:
            QMessageBox.information(self, APP_NAME, "Load an input clip first.")
            return
        js = self.job_settings()
        if not js.export.formats:
            QMessageBox.warning(self, APP_NAME, "Select at least one export format.")
            return
        if not export_only and self.models.missing():
            r = QMessageBox.question(self, APP_NAME, "Model files are missing. Open the Model Manager?")
            if r == QMessageBox.Yes:
                self.open_models()
            return
        self.preview.stop()
        self.running = True
        self.bottom.set_running(True)
        self.bottom.set_progress(0, 1, None, "")
        try:
            geo, segs, job_dir, total, _ = self.engine.plan_job(self.info, js)
            job_dir.mkdir(parents=True, exist_ok=True)
            from ..engine import MatteCache

            self.preview.set_cache(MatteCache(job_dir, segs, geo, total), js.in_frame)
        except (VideoError, ValueError) as exc:
            self._on_failed(str(exc), "other", [])
            return
        self._save_ui()
        g = next((x for x in self.gpus if x.index == js.gpu_index), None)
        self.engine.start(self.input_path, js, self.listener, export_only=export_only,
                          gpu_total_gb=g.total_gb if g else 0.0)

    def cancel(self) -> None:
        if self.running:
            self.bottom.stage.setText("Cancelling…")
            self.engine.request_cancel()

    def _on_stage(self, text: str) -> None:
        self.bottom.stage.setText(text)
        self.bottom.append_log("INFO", f"▸ {text}")

    def _on_finished(self, report: dict) -> None:
        self.running = False
        self.bottom.set_running(False)
        self.bottom.stage.setText("Done ✔")
        checks = report.get("checks", {})
        bad = [k for k, v in checks.items() if not v.get("ok")]
        lines = [f"{k}: {v}" for k, v in report.get("outputs", {}).items()]
        self.bottom.append_log("OK" if not bad else "ERROR", "Outputs:\n  " + "\n  ".join(lines))
        self.preview.refresh_ready()
        self.preview.refresh()
        msg = "Export complete.\n\n" + "\n".join(lines)
        if "prores4444" in checks:
            c = checks["prores4444"]
            msg += (f"\n\nProRes 4444: {c.get('pix_fmt')}, {c.get('frames')} frames, alpha verified by decoding "
                    f"(max. error {c.get('max_alpha_error')}) – {'OK' if c.get('ok') else 'ERROR'}")
        if bad:
            msg += "\n\n⚠ Some checks failed: " + ", ".join(bad)
        QMessageBox.information(self, APP_NAME, msg)

    def _on_failed(self, message: str, kind: str, suggestions: list) -> None:
        self.running = False
        self.bottom.set_running(False)
        self.bottom.stage.setText("Cancelled" if kind == "cancelled" else "Error")
        self.bottom.append_log("WARNING" if kind == "cancelled" else "ERROR", message)
        self.preview.refresh_ready()
        if kind == "cancelled":
            return
        if kind == "oom":
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Out of VRAM")
            box.setText("The GPU ran out of memory (CUDA out of memory).\n\n" + "\n".join("• " + s for s in suggestions))
            b_res = box.addButton("Lower resolution", QMessageBox.AcceptRole)
            b_seg = box.addButton("Shorten segment", QMessageBox.AcceptRole)
            box.addButton("Close", QMessageBox.RejectRole)
            box.exec()
            sp = self.settings_panel
            if sp.mode == "auto":
                sp.set_mode("expert")  # ruční úprava = režim Expert (s vysvětlivkami)
            if box.clickedButton() is b_res and sp.resolution.currentIndex() < sp.resolution.count() - 1:
                sp.resolution.setCurrentIndex(sp.resolution.currentIndex() + 1)
            elif box.clickedButton() is b_seg:
                sp.seg_len.setValue(max(17, sp.seg_len.value() - 16))
            return
        QMessageBox.warning(self, "Processing error", message + ("\n\n" + "\n".join("• " + s for s in suggestions)
                                                                 if suggestions else ""))
        if kind == "missing_model":
            self.open_models()

    # ------------------------------------------------------------------ dávková fronta
    def _queue_running(self) -> bool:
        return bool(self.queue_runner and self.queue_runner.running())

    def _queue_item_settings(self, full_clip: bool) -> tuple[dict, str | None]:
        js = self.job_settings()
        if full_clip:
            js.in_frame, js.out_frame = 0, None
        sp = self.settings_panel
        profile = sp.profile if sp.mode == "auto" else None
        return js.to_dict(), profile

    def _queue_add_current(self) -> None:
        if not self.info or not self.input_path:
            QMessageBox.information(self, APP_NAME, "Load a clip first, or use “+ Files…”.")
            return
        settings, profile = self._queue_item_settings(full_clip=False)
        self.queue.add(QueueItem(self.input_path, settings, profile))
        self._queue_changed()
        self.bottom.tabs.setCurrentIndex(self.bottom.queue_tab)

    def _queue_add_files(self) -> None:
        exts = "*.mp4 *.mov *.m4v *.mkv *.mxf *.avi *.webm *.mts *.m2ts"
        paths, _ = QFileDialog.getOpenFileNames(self, "Add videos to the queue", self.settings.last_input_dir or "",
                                                f"Video ({exts});;All files (*.*)")
        if not paths:
            return
        settings, profile = self._queue_item_settings(full_clip=True)
        for p in paths:
            self.queue.add(QueueItem(p, dict(settings), profile))
        self.settings.last_input_dir = str(Path(paths[0]).parent)
        self._queue_changed()
        self.bottom.tabs.setCurrentIndex(self.bottom.queue_tab)

    def _queue_start(self) -> None:
        if self.running:
            QMessageBox.information(self, APP_NAME, "A single job is running – wait for it or cancel it first.")
            return
        if not self.queue.next_waiting():
            QMessageBox.information(self, APP_NAME, "No waiting jobs in the queue.")
            return
        if self.models.missing():
            QMessageBox.warning(self, APP_NAME, "Model files are missing – open the Model Manager first.")
            return
        self.preview.stop()
        self._save_ui()
        self.queue_runner = QueueRunner(self.engine, self.queue, self.listener,
                                        on_change=self.sig_queue_changed.emit, on_finished=self.sig_queue_done.emit)
        self.bottom.set_running(True)
        self.bottom.start_btn.setEnabled(False)
        self.bottom.export_btn.setEnabled(False)
        self.bottom.cancel_btn.setEnabled(False)  # frontu zastavuje „Stop queue“
        self.queue_panel.set_running(True)
        self.queue_runner.start()

    def _queue_stop(self) -> None:
        if self._queue_running():
            self.bottom.stage.setText("Stopping the queue…")
            self.queue_runner.stop()

    def _queue_changed(self) -> None:
        self.queue_panel.refresh()
        n = self.queue.counts()
        pending = n["waiting"] + n["running"]
        self.bottom.set_queue_title(f"Queue ({pending})" if pending else "Queue")

    def _queue_done(self, counts: dict) -> None:
        self.queue_panel.set_running(False)
        self.bottom.set_running(False)
        self._queue_changed()
        stopped = self.queue_runner.stop_requested.is_set() if self.queue_runner else False
        self.bottom.stage.setText("Queue stopped" if stopped else "Queue finished ✔")
        self.bottom.append_log("OK", f"Queue: {counts.get('done', 0)} done, {counts.get('failed', 0)} failed, "
                                     f"{counts.get('waiting', 0)} waiting.")
        if not stopped and self.queue_panel.shutdown.isChecked():
            self.bottom.append_log("WARNING", "Shutting down the PC in 2 minutes (cancel: shutdown /a).")
            QueuePanel.schedule_shutdown()

    def _stop_worker(self) -> None:
        if self.running:
            return
        if self.engine.client:
            self.engine.client.stop()
            self.engine.client = None
            self.bottom.append_log("INFO", "Inference worker stopped, VRAM freed.")

    def _about(self) -> None:
        QMessageBox.about(self, APP_NAME, f"<b>{APP_NAME} {__version__}</b><br>"
                          "AI video keying with the official Lightricks LTX-2.5 AlphaGen model (IC-LoRA).<br>"
                          "Inference: ltx_pipelines.ic_lora, distilled, stage 1 (--skip-stage-2).<br>"
                          "The model picks the main subject itself – subject selection by prompt or mask is not supported.<br>"
                          "Weights license: LTX-2.x Community License (Lightricks).")

    # ------------------------------------------------------------------
    def _restore_ui(self) -> None:
        ui = self.settings.extra.get("ui", {})
        self.settings_panel.restore(ui.get("settings", {}))
        geo = ui.get("geometry")
        if geo:
            self.restoreGeometry(QByteArray.fromHex(geo.encode()))

    def _save_ui(self) -> None:
        self.settings.extra["ui"] = {"settings": self.settings_panel.state(),
                                     "geometry": bytes(self.saveGeometry().toHex()).decode()}
        self.settings.save()

    def closeEvent(self, e) -> None:  # noqa: N802
        if self._queue_running():
            r = QMessageBox.question(self, APP_NAME, "The queue is running. Quit anyway? The current job will resume "
                                                     "from finished segments next time.")
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.queue_runner.stop()
        if self.running:
            r = QMessageBox.question(self, APP_NAME, "Processing is running. Quit anyway? Finished segments are kept "
                                                     "and processing can resume.")
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.engine.request_cancel()
        self._save_ui()
        self.preview.shutdown()
        self.engine.shutdown()
        super().closeEvent(e)
