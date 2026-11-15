"""Středový panel – náhled: originál / matte / šachovnice / barva / před-po, timeline, přehrávání, zoom."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QComboBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..engine import MatteCache
from ..export import checkerboard, parse_color
from ..logging_setup import get_logger
from ..postprocess import MatteSettings, process_frame, upscale_matte
from ..video_io import FrameReader, VideoInfo, format_tc, read_frame
from . import theme

log = get_logger("preview")
MODES = [("orig", "Source"), ("matte", "Alpha matte"), ("checker", "Checker"),
         ("color", "Color"), ("compare", "Before / After")]
PLAYBACK_MAX_W = 1280


def to_u8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    if rgb.dtype == np.uint16:
        return (rgb >> 8).astype(np.uint8)
    return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def render_view(rgb: np.ndarray, alpha: np.ndarray | None, mode: str, bg_color: str, split: float) -> np.ndarray:
    """Složí obraz pro zobrazení (uint8 RGB)."""
    if mode == "orig" or alpha is None:
        return to_u8(rgb)
    if mode == "matte":
        g = np.round(np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)
    f = rgb.astype(np.float32) / (65535.0 if rgb.dtype == np.uint16 else 255.0)
    h, w = alpha.shape
    if mode == "color":
        back = np.broadcast_to(parse_color(bg_color), f.shape)
    else:
        back = checkerboard(h, w, max(8, w // 60))
    comp = f * alpha[..., None] + back * (1.0 - alpha[..., None])
    out = np.round(np.clip(comp, 0, 1) * 255).astype(np.uint8)
    if mode == "compare":
        x = int(np.clip(split, 0, 1) * w)
        out[:, :x] = to_u8(rgb)[:, :x]
        out[:, max(0, x - 1):x + 1] = theme.ACCENT_RGB
    return out


def np_to_qimage(img: np.ndarray) -> QImage:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    return QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# ----------------------------------------------------------------------------
class PreviewWorker(threading.Thread):
    """Načítání a skládání snímků mimo GUI vlákno ("poslední požadavek vyhrává")."""

    def __init__(self, emit_frame, emit_status) -> None:
        super().__init__(daemon=True)
        self.q: queue.Queue = queue.Queue()
        self.emit_frame = emit_frame
        self.emit_status = emit_status
        self.info: VideoInfo | None = None
        self.ffmpeg = ""
        self.cache: MatteCache | None = None
        self.cache_in = 0
        self.matte = MatteSettings()
        self.mode = "orig"
        self.bg = "#00b140"
        self.split = 0.5
        self._frame_cache: dict[int, np.ndarray] = {}
        self._playing = threading.Event()
        self._stop = threading.Event()

    def request(self, frame: int) -> None:
        self.q.put(("frame", frame))

    def play(self, start: int, end: int) -> None:
        self.q.put(("play", (start, end)))

    def stop_playback(self) -> None:
        self._playing.clear()

    def shutdown(self) -> None:
        self._stop.set()
        self._playing.clear()
        self.q.put(("quit", None))

    def _alpha_for(self, f: int, size: tuple[int, int] | None, rgb: np.ndarray | None) -> np.ndarray | None:
        c = self.cache
        if c is None:
            return None
        rel = f - self.cache_in
        if not 0 <= rel < c.total or not c.frame_ready(rel):
            return None
        raw = c.raw(rel, full_res=size is None)
        ms = self.matte
        if size is not None:  # přehrávání ve zmenšeném rozlišení – px parametry škálujeme
            raw = upscale_matte(raw, size[0], size[1])
            scale = size[0] / c.geo.src_w
            ms = MatteSettings.from_dict({**ms.to_dict(), "erode_dilate": ms.erode_dilate * scale,
                                          "feather": ms.feather * scale,
                                          "edge_radius": max(1, int(round(ms.edge_radius * scale))),
                                          "despeckle": int(ms.despeckle * scale * scale)})
        return process_frame(raw, rgb, ms)

    def run(self) -> None:
        while not self._stop.is_set():
            cmd, arg = self.q.get()
            # zahodit zastaralé požadavky
            while not self.q.empty():
                nxt = self.q.get()
                if nxt[0] == "quit":
                    return
                cmd, arg = nxt
            if cmd == "quit":
                return
            try:
                if cmd == "frame":
                    self._show(arg)
                elif cmd == "play":
                    self._play(*arg)
            except Exception as exc:  # noqa: BLE001
                log.exception("Preview failed")
                self.emit_status(f"Preview error: {exc}")

    def _show(self, f: int) -> None:
        if not self.info:
            return
        rgb = self._frame_cache.get(f)
        if rgb is None:
            rgb = read_frame(self.info, self.ffmpeg, f)
            if len(self._frame_cache) > 8:
                self._frame_cache.pop(next(iter(self._frame_cache)))
            self._frame_cache[f] = rgb
        alpha = self._alpha_for(f, None, rgb)
        self._emit(f, rgb, alpha)

    def _emit(self, f: int, rgb: np.ndarray, alpha: np.ndarray | None) -> None:
        img = render_view(rgb, alpha, self.mode, self.bg, self.split)
        msg = ""
        if self.mode != "orig" and alpha is None:
            msg = "No matte computed for this frame yet – showing source."
        self.emit_frame(f, np_to_qimage(img), msg)

    def _play(self, start: int, end: int) -> None:
        info = self.info
        if not info:
            return
        self._playing.set()
        scale = min(1.0, PLAYBACK_MAX_W / info.width)
        size = (max(2, int(info.width * scale) // 2 * 2), max(2, int(info.height * scale) // 2 * 2))
        reader = FrameReader(info, self.ffmpeg, start, max(1, end - start + 1), high_bit=False, size=size)
        period = info.fps_den / info.fps_num
        t0 = time.perf_counter()
        try:
            for i, rgb in enumerate(reader):
                if not self._playing.is_set() or not self.q.empty():
                    break
                f = start + i
                alpha = self._alpha_for(f, size, rgb)
                self._emit(f, rgb, alpha)
                delay = t0 + (i + 1) * period - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        finally:
            reader.close()
            self._playing.clear()
            self.emit_status("__stopped__")


# ----------------------------------------------------------------------------
class ImageView(QGraphicsView):
    splitChanged = Signal(float)  # noqa: N815

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.item)
        self.setBackgroundBrush(QColor("#0f1012"))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.fit = True
        self.compare = False
        self.message = ""

    def set_image(self, img: QImage) -> None:
        first = self.item.pixmap().isNull() or self.item.pixmap().size() != img.size()
        self.item.setPixmap(QPixmap.fromImage(img))
        self.scene().setSceneRect(QRectF(0, 0, img.width(), img.height()))
        if self.fit or first:
            self.fit_view()

    def fit_view(self) -> None:
        if not self.item.pixmap().isNull():
            self.fitInView(self.item, Qt.KeepAspectRatio)

    def set_zoom(self, factor: float | None) -> None:
        if factor is None:
            self.fit = True
            self.fit_view()
            return
        self.fit = False
        self.resetTransform()
        self.scale(factor, factor)

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        if self.fit:
            self.fit_view()

    def wheelEvent(self, e) -> None:  # noqa: N802
        self.fit = False
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def _split_from(self, pos) -> None:
        p = self.mapToScene(pos)
        w = self.item.pixmap().width()
        if w:
            self.splitChanged.emit(float(np.clip(p.x() / w, 0, 1)))

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if self.compare and e.button() == Qt.LeftButton:
            self._split_from(e.position().toPoint())
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self.compare and e.buttons() & Qt.LeftButton:
            self._split_from(e.position().toPoint())
            return
        super().mouseMoveEvent(e)

    def drawForeground(self, painter: QPainter, rect) -> None:  # noqa: N802
        if not self.message:
            return
        painter.save()
        painter.resetTransform()
        painter.setPen(QColor(theme.WARN))
        r = self.viewport().rect().adjusted(10, 10, -10, -10)
        painter.fillRect(r.x(), r.y(), r.width(), 26, QColor(0, 0, 0, 170))
        painter.drawText(r.adjusted(8, 4, 0, 0), Qt.AlignLeft | Qt.AlignTop, self.message)
        painter.restore()


class TimelineBar(QWidget):
    """Timeline se scrubbingem; zobrazuje IN/OUT a již spočítané úseky masky."""

    scrubbed = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(34)
        self.total = 1
        self.frame = 0
        self.in_f = 0
        self.out_f = 0
        self.ready: np.ndarray | None = None  # bool pro absolutní snímky
        self.setMouseTracking(False)

    def set_total(self, n: int) -> None:
        self.total = max(1, n)
        self.in_f, self.out_f = 0, self.total - 1
        self.ready = None
        self.update()

    def _x(self, f: float) -> float:
        return 8 + (self.width() - 16) * (f / max(1, self.total - 1))

    def _f(self, x: float) -> int:
        return int(round(np.clip((x - 8) / max(1, self.width() - 16), 0, 1) * (self.total - 1)))

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(theme.BG))
        p.fillRect(QRectF(8, 10, w - 16, 12), QColor(theme.PANEL_2))
        p.fillRect(QRectF(self._x(self.in_f), 10, max(2, self._x(self.out_f) - self._x(self.in_f)), 12),
                   QColor(*theme.ACCENT_RGB, 70))
        if self.ready is not None and self.ready.any():
            col = QColor(theme.OK)
            idx = np.flatnonzero(np.diff(np.concatenate([[0], self.ready.astype(np.int8), [0]])))
            for a, b in zip(idx[::2], idx[1::2]):
                p.fillRect(QRectF(self._x(a), 22, max(1.5, self._x(b - 1) - self._x(a)), 4), col)
        # ticks
        p.setPen(QPen(QColor(theme.BORDER), 1))
        steps = 10
        for i in range(steps + 1):
            x = 8 + (w - 16) * i / steps
            p.drawLine(QPointF(x, 6), QPointF(x, 10))
        p.setPen(QPen(QColor(theme.ACCENT), 2))
        for f in (self.in_f, self.out_f):
            p.drawLine(QPointF(self._x(f), 6), QPointF(self._x(f), 26))
        x = self._x(self.frame)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(QPointF(x, 2), QPointF(x, h - 2))
        p.end()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self.scrubbed.emit(self._f(e.position().x()))

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if e.buttons() & Qt.LeftButton:
            self.scrubbed.emit(self._f(e.position().x()))


# ----------------------------------------------------------------------------
class PreviewPanel(QWidget):
    frameChanged = Signal(int)  # noqa: N815
    _frameReady = Signal(int, QImage, str)  # noqa: N815
    _status = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.info: VideoInfo | None = None
        self.frame = 0
        self.playing = False
        self.worker = PreviewWorker(self._frameReady.emit, self._status.emit)
        self.worker.start()
        self._frameReady.connect(self._on_frame)
        self._status.connect(self._on_status)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        for i, (key, label) in enumerate(MODES):
            b = QToolButton()
            b.setText(label)
            b.setCheckable(True)
            b.setProperty("mode", key)
            b.setToolTip(f"Viewer mode: {label} (key {i + 1})")
            self.mode_group.addButton(b, i)
            bar.addWidget(b)
        self.mode_group.button(0).setChecked(True)
        self.mode_group.idClicked.connect(self._mode_changed)
        self.color_btn = QPushButton("■ Background color")
        self.color_btn.clicked.connect(self._pick_color)
        self._set_color_btn("#00b140")
        bar.addWidget(self.color_btn)
        bar.addStretch(1)
        bar.addWidget(QLabel("Zoom"))
        self.zoom = QComboBox()
        self.zoom.addItems(["Fit", "25%", "50%", "100%", "200%", "400%"])
        self.zoom.currentIndexChanged.connect(self._zoom_changed)
        bar.addWidget(self.zoom)
        lay.addLayout(bar)

        self.view = ImageView()
        self.view.splitChanged.connect(self._split_changed)
        lay.addWidget(self.view, 1)

        self.timeline = TimelineBar()
        self.timeline.scrubbed.connect(self.seek)
        lay.addWidget(self.timeline)

        tr = QHBoxLayout()
        self.btn_start = QToolButton(text="⏮")
        self.btn_prev = QToolButton(text="◀")
        self.btn_play = QToolButton(text="▶")
        self.btn_next = QToolButton(text="▶|")
        self.btn_end = QToolButton(text="⏭")
        tips = ["Go to IN", "Previous frame (←)", "Play / pause (Space)", "Next frame (→)", "Go to OUT"]
        for b, t in zip((self.btn_start, self.btn_prev, self.btn_play, self.btn_next, self.btn_end), tips):
            b.setToolTip(t)
            tr.addWidget(b)
        self.btn_start.clicked.connect(lambda: self.seek(self.timeline.in_f))
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_next.clicked.connect(lambda: self.step(1))
        self.btn_end.clicked.connect(lambda: self.seek(self.timeline.out_f))
        tr.addSpacing(12)
        self.frame_spin = QSpinBox()
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.valueChanged.connect(self.seek)
        tr.addWidget(QLabel("Frame"))
        tr.addWidget(self.frame_spin)
        self.tc_label = QLabel("00:00:00.000")
        self.tc_label.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        tr.addWidget(self.tc_label)
        tr.addStretch(1)
        self.status_label = QLabel("")
        self.status_label.setObjectName("Dim")
        tr.addWidget(self.status_label)
        lay.addLayout(tr)
        self._set_enabled(False)

    # ------------------------------------------------------------------
    def _set_enabled(self, on: bool) -> None:
        for w in (self.btn_start, self.btn_prev, self.btn_play, self.btn_next, self.btn_end, self.frame_spin):
            w.setEnabled(on)

    def set_source(self, info: VideoInfo, ffmpeg: str) -> None:
        self.stop()
        self.info = info
        self.worker.info = info
        self.worker.ffmpeg = ffmpeg
        self.worker._frame_cache.clear()
        self.worker.cache = None
        self.timeline.set_total(info.nb_frames)
        self.frame_spin.blockSignals(True)
        self.frame_spin.setRange(0, info.nb_frames - 1)
        self.frame_spin.blockSignals(False)
        self._set_enabled(True)
        self.view.fit = True
        self.seek(0, force=True)

    def set_cache(self, cache: MatteCache | None, in_frame: int) -> None:
        self.worker.cache = None if cache is None else MatteCache(cache.job_dir, cache.segs, cache.geo, cache.total)
        self.worker.cache_in = in_frame
        self.refresh_ready()
        self.refresh()

    def refresh_ready(self) -> None:
        c = self.worker.cache
        if c is None or not self.info:
            self.timeline.ready = None
        else:
            r = np.zeros(self.info.nb_frames, dtype=bool)
            rel = c.ready_mask()
            a = self.worker.cache_in
            r[a:a + len(rel)] = rel[: max(0, self.info.nb_frames - a)]
            self.timeline.ready = r
        self.timeline.update()

    def set_range(self, a: int, b: int) -> None:
        self.timeline.in_f, self.timeline.out_f = a, b
        self.timeline.update()

    def set_matte_settings(self, ms: MatteSettings) -> None:
        self.worker.matte = ms
        if not self.playing:
            self.refresh()

    def refresh(self) -> None:
        if self.info and not self.playing:
            self.worker.request(self.frame)

    def seek(self, f: int, force: bool = False) -> None:
        if not self.info:
            return
        f = int(np.clip(f, 0, self.info.nb_frames - 1))
        if f == self.frame and not force:
            return
        if self.playing:
            self.stop()
        self.frame = f
        self._update_ui()
        self.worker.request(f)

    def step(self, d: int) -> None:
        self.seek(self.frame + d)

    def toggle_play(self) -> None:
        if self.playing:
            self.stop()
            return
        if not self.info:
            return
        start = self.frame if self.frame < self.timeline.out_f else self.timeline.in_f
        self.playing = True
        self.btn_play.setText("⏸")
        self.worker.play(start, self.timeline.out_f)

    def stop(self) -> None:
        if self.playing:
            self.worker.stop_playback()
            self.playing = False
            self.btn_play.setText("▶")
            self.worker.request(self.frame)  # plné rozlišení po zastavení

    def _update_ui(self) -> None:
        self.timeline.frame = self.frame
        self.timeline.update()
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(self.frame)
        self.frame_spin.blockSignals(False)
        if self.info:
            self.tc_label.setText(format_tc(self.info.frame_to_time(self.frame)))
        self.frameChanged.emit(self.frame)

    def _on_frame(self, f: int, img: QImage, msg: str) -> None:
        if self.playing:
            self.frame = f
            self._update_ui()
        self.view.message = msg
        self.view.set_image(img)
        self.view.viewport().update()

    def _on_status(self, text: str) -> None:
        if text == "__stopped__":
            if self.playing:
                self.playing = False
                self.btn_play.setText("▶")
            return
        self.status_label.setText(text)

    def _mode_changed(self, idx: int) -> None:
        mode = MODES[idx][0]
        self.worker.mode = mode
        self.view.compare = mode == "compare"
        self.view.setDragMode(QGraphicsView.NoDrag if mode == "compare" else QGraphicsView.ScrollHandDrag)
        self.refresh()

    def set_mode_index(self, idx: int) -> None:
        if 0 <= idx < len(MODES):
            self.mode_group.button(idx).setChecked(True)
            self._mode_changed(idx)

    def _split_changed(self, v: float) -> None:
        self.worker.split = v
        self.refresh()

    def _set_color_btn(self, hex_color: str) -> None:
        self.worker.bg = hex_color
        self.color_btn.setStyleSheet(f"QPushButton {{ color: {hex_color}; }}")

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(self.worker.bg), self, "Background color for composite")
        if c.isValid():
            self._set_color_btn(c.name())
            if self.worker.mode == "color":
                self.refresh()

    def _zoom_changed(self, idx: int) -> None:
        self.view.set_zoom([None, 0.25, 0.5, 1.0, 2.0, 4.0][idx])

    def shutdown(self) -> None:
        self.worker.shutdown()

