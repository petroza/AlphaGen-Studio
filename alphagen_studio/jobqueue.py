"""Dávková fronta úloh (na noc): perzistentní seznam jobů zpracovávaný postupně.

* Fronta se ukládá do ``config/queue.json`` – přežije zavření aplikace i restart PC.
* Job v režimu Automatic nese jen klíč profilu; konkrétní rozlišení a úseky se dopočítají
  až při spuštění podle rozměrů daného videa a karty (stejně jako v GUI).
* Selhání jednoho jobu (např. nedostatek VRAM) frontu nezastaví – pokračuje se dalším.
* Hotové úseky masky zůstávají v cache, takže přerušený job po novém spuštění naváže.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config
from .backend.client import BackendError, Cancelled
from .engine import Engine, JobListener, JobSettings
from .gpu import list_gpus, ram_info
from .logging_setup import get_logger

log = get_logger("queue")

WAITING, RUNNING, DONE, FAILED, CANCELLED = "waiting", "running", "done", "failed", "cancelled"


@dataclass
class QueueItem:
    input_path: str
    settings: dict  # JobSettings.to_dict()
    profile: str | None = None  # klíč profilu (Automatic) – dopočítá se při běhu
    label: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = WAITING
    message: str = ""
    outputs: dict = field(default_factory=dict)
    added_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def name(self) -> str:
        return self.label or Path(self.input_path).name

    def job_settings(self) -> JobSettings:
        return JobSettings.from_dict(self.settings)

    def range_text(self) -> str:
        js = self.settings
        a, b = js.get("in_frame", 0), js.get("out_frame")
        return "full clip" if (a == 0 and b is None) else f"{a}–{(b - 1) if b else 'end'}"

    def duration_text(self) -> str:
        if self.started_at and self.finished_at:
            s = int(self.finished_at - self.started_at)
            return f"{s // 60}:{s % 60:02d}"
        return ""


def _queue_file() -> Path:
    return config.CONFIG_DIR / "queue.json"


class JobQueue:
    def __init__(self) -> None:
        self.items: list[QueueItem] = []
        self.lock = threading.RLock()
        self.load()

    # ------------------------------------------------------------------
    def load(self) -> None:
        try:
            data = json.loads(_queue_file().read_text(encoding="utf-8"))
            items = [QueueItem(**{k: v for k, v in d.items() if k in QueueItem.__dataclass_fields__}) for d in data]
        except (OSError, ValueError, TypeError):
            items = []
        for it in items:  # rozběhnutý job z minula (pád / zavření) se vrátí do fronty
            if it.status == RUNNING:
                it.status, it.message = WAITING, "interrupted – will resume from finished segments"
        self.items = items

    def save(self) -> None:
        with self.lock:
            path = _queue_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps([asdict(i) for i in self.items], indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)

    # ------------------------------------------------------------------
    def add(self, item: QueueItem) -> QueueItem:
        with self.lock:
            self.items.append(item)
            self.save()
        return item

    def remove(self, ids: list[str]) -> None:
        with self.lock:
            self.items = [i for i in self.items if i.id not in ids or i.status == RUNNING]
            self.save()

    def move(self, item_id: str, delta: int) -> None:
        with self.lock:
            idx = next((n for n, i in enumerate(self.items) if i.id == item_id), None)
            if idx is None:
                return
            new = max(0, min(len(self.items) - 1, idx + delta))
            self.items.insert(new, self.items.pop(idx))
            self.save()

    def retry_failed(self) -> None:
        with self.lock:
            for i in self.items:
                if i.status in (FAILED, CANCELLED):
                    i.status, i.message = WAITING, ""
            self.save()

    def clear_finished(self) -> None:
        with self.lock:
            self.items = [i for i in self.items if i.status != DONE]
            self.save()

    def next_waiting(self) -> QueueItem | None:
        with self.lock:
            return next((i for i in self.items if i.status == WAITING), None)

    def counts(self) -> dict[str, int]:
        out = {WAITING: 0, RUNNING: 0, DONE: 0, FAILED: 0, CANCELLED: 0}
        for i in self.items:
            out[i.status] = out.get(i.status, 0) + 1
        return out


class QueueRunner:
    """Zpracuje čekající joby postupně ve vlákně. ``on_change`` se volá po každé změně stavu."""

    def __init__(self, engine: Engine, queue: JobQueue, listener: JobListener,
                 on_change: Callable[[], None] | None = None,
                 on_finished: Callable[[dict], None] | None = None) -> None:
        self.engine = engine
        self.queue = queue
        self.listener = listener
        self.on_change = on_change or (lambda: None)
        self.on_finished = on_finished or (lambda counts: None)
        self.stop_requested = threading.Event()
        self.thread: threading.Thread | None = None

    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def start(self) -> None:
        if self.running():
            return
        self.stop_requested.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Zastaví frontu; aktuální job se přeruší (hotové úseky zůstanou v cache)."""
        self.stop_requested.set()
        self.engine.request_cancel()

    def _prepare(self, item: QueueItem) -> JobSettings:
        js = item.job_settings()
        if item.profile:
            from .profiles import apply_profile
            from .video_io import probe

            info = probe(item.input_path, self.engine.settings.ffprobe_path())
            gpu = next((g for g in list_gpus() if g.index == js.gpu_index), None)
            apply_profile(js, item.profile, info.width, info.height, gpu, ram_info()[1])
        return js

    def _loop(self) -> None:
        gpus = {g.index: g for g in list_gpus()}
        while not self.stop_requested.is_set():
            item = self.queue.next_waiting()
            if item is None:
                break
            item.status, item.message, item.started_at, item.finished_at = RUNNING, "", time.time(), 0.0
            self.queue.save()
            self.on_change()
            self.listener.stage(f"Queue: {item.name}")
            try:
                js = self._prepare(item)
                g = gpus.get(js.gpu_index)
                self.engine.cancel.clear()
                report = self.engine.run(item.input_path, js, self.listener, False, g.total_gb if g else 0.0)
                item.status, item.outputs = DONE, report.get("outputs", {})
                bad = [k for k, v in report.get("checks", {}).items() if not v.get("ok")]
                item.message = "OK" if not bad else "finished, check failed: " + ", ".join(bad)
            except Cancelled:
                item.status, item.message = CANCELLED, "stopped – finished segments are kept"
            except BackendError as exc:
                if self.stop_requested.is_set():
                    item.status, item.message = CANCELLED, "stopped – finished segments are kept"
                else:
                    item.status, item.message = FAILED, f"{exc.kind}: {exc}"
                    if exc.details:
                        log.error("Queue job %s traceback:\n%s", item.name, exc.details)
            except Exception as exc:  # noqa: BLE001 - jeden vadný job nesmí zastavit noční frontu
                log.exception("Queue job failed: %s", item.name)
                item.status, item.message = FAILED, f"{type(exc).__name__}: {exc}"
            item.finished_at = time.time()
            self.queue.save()
            self.on_change()
            self.listener.log("INFO" if item.status == DONE else "ERROR",
                              f"Queue: {item.name} → {item.status} {item.message}")
        if self.engine.client:  # uvolnit VRAM po dokončení fronty
            self.engine.client.stop()
            self.engine.client = None
        self.on_finished(self.queue.counts())
