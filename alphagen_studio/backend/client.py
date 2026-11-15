"""Klient inference workeru – spouští ``worker_main.py`` v prostředí LTX-2 jako samostatný proces.

Oddělený proces = GUI zůstává responzivní, OOM/pád CUDA neshodí aplikaci,
a zrušení úlohy je spolehlivé (proces se ukončí a VRAM se uvolní).
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ..logging_setup import get_logger
from ..video_io import CREATE_NO_WINDOW

log = get_logger("backend")
WORKER_SCRIPT = Path(__file__).with_name("worker_main.py")
PROTO = "@@AG "
_TQDM_RE = re.compile(r"\r|\x1b\[[0-9;]*[A-Za-z]")
# Neškodný šum knihoven při ukončení/importu (OpenImageIO nanobind, colour-science bez matplotlib)
_NOISE = ("nanobind:", "- leaked ", "- ... skipped", "ColourUsageWarning", "colour-science.org",
          "warn(*args, **kwargs)", "this is likely caused by a reference counting")


class BackendError(RuntimeError):
    def __init__(self, message: str, kind: str = "other", details: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.details = details


class Cancelled(RuntimeError):
    pass


class WorkerClient:
    def __init__(self, python: str | Path, gpu_index: int | None = None,
                 on_log: Callable[[str, str], None] | None = None, hf_home: str | None = None,
                 script: Path | None = None) -> None:
        self.python = str(python)
        self.script = Path(script) if script else WORKER_SCRIPT
        self.gpu_index = gpu_index
        self.on_log = on_log or (lambda lvl, msg: None)
        self.hf_home = hf_home
        self.proc: subprocess.Popen | None = None
        self.events: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self.loaded_cfg: dict | None = None

    # ------------------------------------------------------------------
    def start(self, timeout: float = 120) -> None:
        if not Path(self.python).exists():
            raise BackendError(
                f"Inference environment not found ({self.python}). Run setup.bat (installs LTX-2).",
                kind="missing_env",
            )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if self.gpu_index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if self.hf_home:
            env["HF_HOME"] = self.hf_home
        self.proc = subprocess.Popen(  # noqa: S603
            [self.python, "-u", str(self.script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, creationflags=CREATE_NO_WINDOW, cwd=str(self.script.parent),
        )
        t1 = threading.Thread(target=self._read_stdout, daemon=True)
        t2 = threading.Thread(target=self._read_stderr, daemon=True)
        t1.start()
        t2.start()
        self._threads = [t1, t2]
        ev = self.wait_for({"ready"}, timeout=timeout)
        log.info("Inference worker running (pid %s, Python %s)", ev.get("pid"), ev.get("python"))

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for raw in self.proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith(PROTO):
                try:
                    self.events.put(json.loads(line[len(PROTO):]))
                except json.JSONDecodeError:
                    self.on_log("WARNING", line)
            elif line.strip():
                self.on_log("INFO", line)
        self.events.put({"ev": "exit", "code": self.proc.wait()})

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        buf = b""
        last_progress = 0.0
        while True:
            chunk = self.proc.stderr.read1(4096) if hasattr(self.proc.stderr, "read1") else self.proc.stderr.read(4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for p in parts:
                text = _TQDM_RE.sub("", p.decode("utf-8", errors="replace")).strip()
                if not text or any(n in text for n in _NOISE):
                    continue
                is_progress = "%|" in text or "it/s" in text or "s/it" in text
                if is_progress:  # tqdm – posílej max 1× za sekundu
                    if time.time() - last_progress < 1.0:
                        continue
                    last_progress = time.time()
                    self.on_log("PROGRESS", text)
                else:
                    level = "ERROR" if ("Error" in text or "Traceback" in text) else "INFO"
                    self.on_log(level, text)

    # ------------------------------------------------------------------
    def send(self, obj: dict) -> None:
        if not self.alive():
            raise BackendError("Inference worker is not running.", kind="dead")
        try:
            self.proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise BackendError("Lost connection to the inference worker.", kind="dead") from exc

    def wait_for(self, kinds: set[str], timeout: float | None = None, cancel: threading.Event | None = None,
                 on_event: Callable[[dict], None] | None = None) -> dict:
        deadline = None if timeout is None else time.time() + timeout
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled("Cancelled by user.")
            remaining = 0.25 if deadline is None else min(0.25, deadline - time.time())
            if remaining <= 0:
                raise BackendError("Inference worker timed out.", kind="timeout")
            try:
                ev = self.events.get(timeout=remaining)
            except queue.Empty:
                continue
            kind = ev.get("ev")
            if kind == "log":
                self.on_log(ev.get("level", "INFO"), ev.get("message", ""))
                continue
            if kind == "error":
                raise BackendError(ev.get("message", "Unknown error"), kind=ev.get("kind", "other"),
                                   details=ev.get("traceback", ""))
            if kind == "exit":
                raise BackendError(f"Inference worker exited (code {ev.get('code')}). See the log for details.", kind="dead")
            if kind in kinds:
                return ev
            if on_event:
                on_event(ev)

    def load(self, cfg: dict, cancel: threading.Event | None = None) -> dict:
        self.send({"cmd": "load", **cfg})
        ev = self.wait_for({"loaded"}, cancel=cancel)
        self.loaded_cfg = cfg
        return ev

    def run_segment(self, req: dict, cancel: threading.Event | None = None,
                    on_event: Callable[[dict], None] | None = None) -> dict:
        self.send({"cmd": "run", **req})
        return self.wait_for({"done"}, cancel=cancel, on_event=on_event)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, graceful: bool = True) -> None:
        if not self.proc:
            return
        if graceful and self.alive():
            try:
                self.send({"cmd": "quit"})
                self.proc.wait(timeout=10)
            except (BackendError, subprocess.TimeoutExpired):
                pass
        if self.alive():
            self.proc.kill()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None
        self.loaded_cfg = None


def check_environment(python: str | Path, timeout: float = 180) -> dict:
    """Ověří inference prostředí: torch, CUDA, import ltx_pipelines.ic_lora."""
    code = (
        "import json,sys\n"
        "r={'python':sys.version.split()[0]}\n"
        "try:\n"
        " import torch; r['torch']=torch.__version__; r['cuda']=torch.version.cuda;"
        " r['cuda_available']=torch.cuda.is_available();"
        " r['gpus']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []\n"
        "except Exception as e: r['torch_error']=str(e)\n"
        "try:\n"
        " import ltx_pipelines.ic_lora as m; r['ltx_pipelines']=True\n"
        "except Exception as e: r['ltx_error']=repr(e)[:500]\n"
        "try:\n"
        " import triton; r['triton']=triton.__version__\n"
        "except Exception: r['triton']=None\n"
        "print('@@AG '+json.dumps(r))\n"
    )
    if not Path(python).exists():
        return {"ok": False, "error": f"Inference environment Python not found: {python}"}
    res = subprocess.run([str(python), "-c", code], capture_output=True, timeout=timeout,  # noqa: S603
                         creationflags=CREATE_NO_WINDOW, check=False)
    for line in res.stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith(PROTO):
            data = json.loads(line[len(PROTO):])
            data["ok"] = bool(data.get("cuda_available") and data.get("ltx_pipelines"))
            return data
    return {"ok": False, "error": res.stderr.decode(errors="replace")[-1500:]}
