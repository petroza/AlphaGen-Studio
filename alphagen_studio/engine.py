"""Orchestrace úlohy: příprava segmentů → AlphaGen inference → prolnutí → postprocess → export.

Fáze inference je cachovaná po segmentech (``work/<job>/seg_XXXX/*.png`` + ``DONE``),
takže úlohu lze přerušit a navázat od posledního dokončeného segmentu, a export
s jinými parametry masky lze spustit znovu bez další inference.
"""

from __future__ import annotations

import hashlib
import os
import json
import shutil
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .backend.client import BackendError, Cancelled, WorkerClient
from .config import AppSettings
from .export import Exporter, ExportSettings
from .logging_setup import get_logger
from .models import ModelManager
from .postprocess import MatteSettings, TemporalSmoother, apply_color, process_frame, upscale_matte
from .segmentation import (
    Geometry,
    Segment,
    blend_plan,
    compute_geometry,
    crop_pad,
    pad_frame,
    plan_segments,
)
from .video_io import FFmpegWriter, FrameReader, VideoError, VideoInfo, normalize_vfr, probe, read_frames

log = get_logger("engine")
ENGINE_VERSION = 1


@dataclass
class JobSettings:
    in_frame: int = 0
    out_frame: int | None = None  # exkluzivní; None = do konce
    long_side: int = 0  # 0 = nativní (max 1920×1088)
    segment_frames: int = 121
    overlap: int = 16
    seed: int = 1234
    gpu_index: int = 0
    quantization: str = "auto"
    offload: str = "auto"
    pad_mode: str = "edge"
    vae: str = "auto"  # auto | diffvae (dokumentace) | conv (oficiální úsporný konvoluční VAE)
    matte: MatteSettings = field(default_factory=MatteSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    output_dir: str = ""
    base_name: str = ""

    def inference_key(self, info: VideoInfo, geo: Geometry) -> dict:
        """Parametry, které mění výsledek modelu (= klíč cache)."""
        return {
            "engine": ENGINE_VERSION, "input": info.path, "frames": info.nb_frames,
            "in": self.in_frame, "out": self.out_frame, "geo": geo.to_dict(),
            "seg": self.segment_frames, "overlap": self.overlap, "seed": self.seed, "pad": self.pad_mode,
            **({"vae": self.vae} if self.vae != "diffvae" else {}),
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> JobSettings:
        d = dict(d or {})
        m = MatteSettings.from_dict(d.pop("matte", {}))
        e = ExportSettings.from_dict(d.pop("export", {}))
        js = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        js.matte, js.export = m, e
        return js


class JobListener:
    """Rozhraní pro průběh – GUI implementuje přes Qt signály, CLI přes print."""

    def log(self, level: str, msg: str) -> None: ...
    def stage(self, text: str) -> None: ...
    def progress(self, done: int, total: int, eta_s: float | None, detail: str) -> None: ...
    def segment_done(self, index: int, total: int) -> None: ...
    def finished(self, report: dict) -> None: ...
    def failed(self, message: str, kind: str, suggestions: list[str]) -> None: ...


# ----------------------------------------------------------------------------
class MatteCache:
    """Čte hotové segmenty a skládá surovou (neupravenou) masku libovolného snímku."""

    def __init__(self, job_dir: Path, segs: list[Segment], geo: Geometry, total: int) -> None:
        self.job_dir = Path(job_dir)
        self.segs = segs
        self.geo = geo
        self.total = total
        self.plan = blend_plan(segs, total)
        self._lru: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def seg_dir(self, k: int) -> Path:
        return self.job_dir / f"seg_{k:04d}"

    def segment_ready(self, k: int) -> bool:
        return (self.seg_dir(k) / "DONE").exists()

    def frame_ready(self, f: int) -> bool:
        return all(self.segment_ready(k) for k, _, _ in self.plan[f])

    def ready_mask(self) -> np.ndarray:
        ready = {k for k in range(len(self.segs)) if self.segment_ready(k)}
        return np.array([all(k in ready for k, _, _ in items) for items in self.plan], dtype=bool)

    def _load(self, k: int, i: int) -> np.ndarray:
        key = (k, i)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        p = self.seg_dir(k) / f"{i:06d}.png"
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise VideoError(f"Corrupted matte frame {p}")
        if img.ndim == 3:
            img = img[..., 0]
        a = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
        a = crop_pad(a, self.geo)
        self._lru[key] = a
        if len(self._lru) > 24:
            self._lru.popitem(last=False)
        return a

    def raw(self, f: int, full_res: bool = True) -> np.ndarray:
        """Prolnutá maska snímku ``f`` (relativně k IN) v rozlišení zdroje."""
        acc = None
        for k, i, w in self.plan[f]:
            a = self._load(k, i)
            acc = a * w if acc is None else acc + a * w
        if full_res:
            acc = upscale_matte(acc, self.geo.src_w, self.geo.src_h)
        return acc


# ----------------------------------------------------------------------------
def resolve_memory(js: JobSettings, gpu_total_gb: float) -> tuple[str, str]:
    from .gpu import GpuInfo, recommend

    from .gpu import ram_info

    avail_ram, total_ram = ram_info()
    plan = recommend(GpuInfo(js.gpu_index, "", int(gpu_total_gb * 1024), 0, 0, ""), total_ram) if gpu_total_gb else None
    q = js.quantization if js.quantization != "auto" else (plan.quantization or "none") if plan else "fp8-cast"
    o = js.offload if js.offload != "auto" else (plan.offload if plan else "cpu")
    if js.offload == "auto" and o == "cpu":
        # Model po částech z RAM potřebuje volnou RAM (FP8 transformer ~21 GB + rezerva).
        # Při nedostatku raději disk (pomalejší) než pád celého PC na swap.
        from .gpu import ram_info

        avail, _ = ram_info()
        if avail and avail < MIN_FREE_RAM_GB:
            o = "disk"
    if js.quantization == "auto" and q == "none" and o == "cpu" and avail_ram and avail_ram < 64:
        q = "fp8-cast"  # bf16 streamování potřebuje volnou RAM; jinak poloviční FP8
    return q, o


MIN_FREE_RAM_GB = 32


def resolve_vae(vae: str, gpu_total_gb: float, conv_available: bool) -> str:
    """auto → úsporný konvoluční VAE na kartách < 40 GB (je-li stažený), jinak DiffVAE z dokumentace."""
    if vae != "auto":
        return vae
    if gpu_total_gb and gpu_total_gb < 40 and conv_available:
        return "conv"
    return "diffvae"


class Engine:
    """Spouští jednu úlohu v pracovním vlákně. Worker proces se mezi úlohami recykluje."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.models = ModelManager(settings)
        self.client: WorkerClient | None = None
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self._client_key = None
        self.gpu_total_gb = 0.0  # pro volbu "auto" (VAE); nastavuje GUI/CLI
        # Jen pro automatické testy synchronizace: (python, skript) náhradního workeru
        self.worker_override: tuple[str, Path] | None = None

    # ------------------------------------------------------------------
    def prepare(self, input_path: str, js: JobSettings) -> tuple[VideoInfo, Geometry, list[Segment], Path, int]:
        ffprobe = self.settings.ffprobe_path()
        info = probe(input_path, ffprobe)
        if info.is_vfr:
            cfr = Path(self.settings.cache_dir) / f"cfr_{_short_hash(input_path + str(Path(input_path).stat().st_mtime))}.mkv"
            if not cfr.exists():
                normalize_vfr(info, self.settings.ffmpeg_path(), cfr)
            info = probe(str(cfr), ffprobe)
        geo, segs, job_dir, total, key = self.plan_job(info, js)
        job_dir.mkdir(parents=True, exist_ok=True)
        meta = {"key": key, "info": info.to_dict(), "geometry": geo.to_dict(),
                "segments": [s.to_dict() for s in segs], "total": total}
        (job_dir / "job.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return info, geo, segs, job_dir, total

    def plan_job(self, info: VideoInfo, js: JobSettings) -> tuple[Geometry, list[Segment], Path, int, dict]:
        """Bez vedlejších efektů: geometrie, segmenty a složka cache pro daný vstup a nastavení."""
        out = info.nb_frames if js.out_frame is None else min(js.out_frame, info.nb_frames)
        if not 0 <= js.in_frame < out:
            raise VideoError(f"Invalid IN/OUT range: {js.in_frame}–{out}")
        total = out - js.in_frame
        geo = compute_geometry(info.width, info.height, js.long_side or None)
        segs = plan_segments(total, js.segment_frames, js.overlap)
        js.vae = resolve_vae(js.vae, self.gpu_total_gb, self.models.has_conv_vae())
        key = js.inference_key(info, geo)
        src = Path(info.seq_pattern).parent if info.is_sequence else Path(info.path)
        st = src.stat()
        key["mtime"], key["size"] = st.st_mtime, st.st_size
        name = (src.name if info.is_sequence else src.stem)[:40]
        job_dir = Path(self.settings.work_dir) / f"{name}_{_short_hash(json.dumps(key, sort_keys=True))}"
        return geo, segs, job_dir, total, key

    def find_cache(self, info: VideoInfo, js: JobSettings) -> MatteCache | None:
        try:
            geo, segs, job_dir, total, _ = self.plan_job(info, js)
        except (VideoError, ValueError, OSError):
            return None
        if not job_dir.exists():
            return None
        return MatteCache(job_dir, segs, geo, total)

    def start(self, input_path: str, js: JobSettings, listener: JobListener, export_only: bool = False,
              gpu_total_gb: float = 0.0) -> None:
        if self.thread and self.thread.is_alive():
            raise RuntimeError("A job is already running.")
        self.cancel.clear()
        self.thread = threading.Thread(target=self._run_safe, args=(input_path, js, listener, export_only,
                                                                    gpu_total_gb), daemon=True)
        self.thread.start()

    def request_cancel(self) -> None:
        self.cancel.set()
        if self.client:  # zrušení uprostřed segmentu = ukončit proces (uvolní VRAM)
            threading.Thread(target=self.client.stop, kwargs={"graceful": False}, daemon=True).start()

    def shutdown(self) -> None:
        self.cancel.set()
        if self.client:
            self.client.stop()
            self.client = None

    # ------------------------------------------------------------------
    def _run_safe(self, input_path, js, listener, export_only, gpu_total_gb) -> None:
        try:
            report = self.run(input_path, js, listener, export_only, gpu_total_gb)
            listener.finished(report)
        except Cancelled:
            listener.failed("Processing cancelled. Finished segments are kept – you can resume.", "cancelled", [])
        except BackendError as exc:
            if exc.kind == "dead" and self.cancel.is_set():
                listener.failed("Processing cancelled. Finished segments are kept – you can resume.", "cancelled", [])
                return
            if exc.details:
                log.error("Backend traceback:\n%s", exc.details)
            listener.failed(str(exc), exc.kind, _suggestions(exc.kind, js))
        except Exception as exc:  # noqa: BLE001
            log.exception("Job failed")
            listener.failed(f"{type(exc).__name__}: {exc}", "other", [])
        finally:
            if self.cancel.is_set() and self.client:
                self.client.stop(graceful=False)
                self.client = None

    def run(self, input_path: str, js: JobSettings, L: JobListener, export_only: bool = False,  # noqa: N803
            gpu_total_gb: float = 0.0) -> dict:
        L.stage("Analyzing input")
        if gpu_total_gb:
            self.gpu_total_gb = gpu_total_gb
        info, geo, segs, job_dir, total = self.prepare(input_path, js)
        for w in info.warnings:
            L.log("WARNING", w)
        L.log("INFO", geo.describe_cs())
        L.log("INFO", f"Range: {total} frames ({js.in_frame}–{js.in_frame + total - 1}), "
                      f"{len(segs)} segment(s) of ≤{js.segment_frames} frames, overlap {js.overlap}.")
        if geo.downscaled:
            L.log("WARNING", f"Actual matte working resolution is {geo.work_w}×{geo.work_h}; "
                             f"the matte is upscaled to {geo.src_w}×{geo.src_h}.")
        cache = MatteCache(job_dir, segs, geo, total)
        todo = [k for k in range(len(segs)) if not cache.segment_ready(k)]
        if export_only and todo:
            raise VideoError(f"{len(todo)} matte segment(s) missing – run processing first.")
        if todo:
            self._infer(info, geo, segs, job_dir, todo, js, L, gpu_total_gb)
        elif not export_only:
            L.log("INFO", "All segments already computed (cache) – skipping inference.")
        return self._export(info, geo, cache, total, js, L, job_dir)

    # ------------------------------------------------------------------
    def _write_reference(self, info: VideoInfo, geo: Geometry, seg: Segment, js: JobSettings, path: Path) -> None:
        """Referenční klip pro model: zmenšení (je-li třeba), padding na násobek 32,
        doplnění na 8n+1 opakováním posledního snímku. Bezztrátově (libx264rgb crf 0)."""
        ffmpeg = self.settings.ffmpeg_path()
        frames = read_frames(info, ffmpeg, js.in_frame + seg.start, seg.length, high_bit=False)
        wr = FFmpegWriter(ffmpeg, path, geo.pad_w, geo.pad_h, "rgb24", info.fps,
                          ["-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast", "-g", "1"])
        try:
            last = None
            for f in frames:
                if geo.downscaled:
                    f = cv2.resize(f, (geo.work_w, geo.work_h), interpolation=cv2.INTER_AREA)
                last = pad_frame(f, geo, js.pad_mode)
                wr.write(last)
            for _ in range(seg.model_frames - seg.length):
                wr.write(last)
            wr.close()
        except BaseException:
            wr.abort()
            raise

    def _ensure_client(self, js: JobSettings, probe_video: Path, L: JobListener, gpu_total_gb: float) -> WorkerClient:
        q, o = resolve_memory(js, gpu_total_gb)
        models = self.models.paths()
        if js.vae == "conv":
            if not self.models.has_conv_vae():
                raise BackendError("The lightweight convolutional VAE is not downloaded (Model Manager).", kind="missing_model")
            models["video_vae"] = self.models.conv_vae_path()
            L.log("INFO", "Decoder: convolutional video VAE (low-memory mode).")
        cfg = {"models": models, "quantization": q, "offload": o,
               "cache_dir": str(Path(self.settings.cache_dir) / "prompt_embeddings"),
               "lora_strength": 1.0, "conditioning_strength": 1.0, "probe_video": str(probe_video)}
        key = (js.gpu_index, q, o, json.dumps(cfg["models"], sort_keys=True))
        if self.client and self.client.alive() and self._client_key == key:
            return self.client
        if self.client:
            self.client.stop()
        L.stage("Starting inference worker")
        python, script = self.worker_override or (self.settings.ltx_python_path(), None)
        self.client = WorkerClient(python, js.gpu_index, on_log=L.log,
                                   hf_home=str(Path(self.settings.cache_dir) / "hf"), script=script)
        self.client.start()
        if o == "disk" and js.offload == "auto":
            L.log("WARNING", f"Less than {MIN_FREE_RAM_GB} GB of free RAM – the model will stream from disk (slower). "
                             "Close heavy applications for faster processing.")
        L.stage(f"Loading AlphaGen model (quantization {q}, offload {o}) – first load takes minutes")
        ev = self.client.load(cfg, cancel=self.cancel)
        L.log("INFO", f"Model loaded in {ev.get('seconds')} s.")
        self._client_key = key
        return self.client

    # ------------------------------------------------------------------
    def _record_perf(self, geo: Geometry, frames: int, seconds: float, js: JobSettings) -> None:
        """Uloží naměřený čas segmentu pro odhady (settings.extra['perf'], posledních 60 měření)."""
        perf = load_perf()
        perf.append({"mpx": round(geo.pad_w * geo.pad_h / 1e6, 3), "frames": frames,
                     "sec": round(seconds, 1), "vae": js.vae, "gpu": js.gpu_index})
        save_perf(perf[-60:])

    def estimate_seconds(self, geo: Geometry, segs: list[Segment], vae: str, gpu_index: int = 0) -> float | None:
        """Odhad času inference z předchozích měření na tomto PC: čas ≈ k · snímky · Mpx + režie segmentu."""
        perf = [p for p in load_perf() if p.get("gpu", 0) == gpu_index]
        same = [p for p in perf if p.get("vae") == vae] or perf
        if not same:
            return None
        x = np.array([p["frames"] * p["mpx"] for p in same], dtype=float)
        y = np.array([p["sec"] for p in same], dtype=float)
        if len(same) >= 3 and np.ptp(x) > 1e-6:
            k, c = np.polyfit(x, y, 1)
            k, c = max(k, 1e-3), max(c, 0.0)
        else:
            k, c = float((y / np.maximum(x, 1e-6)).mean()), 0.0
        mpx = geo.pad_w * geo.pad_h / 1e6
        return float(sum(k * s.model_frames * mpx + c for s in segs) + 30)  # +30 s start a načtení

    def _infer(self, info, geo, segs, job_dir, todo, js, L, gpu_total_gb) -> None:  # noqa: N803
        missing = self.models.missing()
        if missing:
            raise BackendError("Missing model files: " + ", ".join(f.filename for f in missing) +
                               ". Open the Model Manager.", kind="missing_model")
        total_model_frames = sum(segs[k].model_frames for k in todo)
        done_frames = 0
        sec_per_frame: list[float] = []
        ref_dir = job_dir / "_ref"
        ref_dir.mkdir(exist_ok=True)
        for n, k in enumerate(todo):
            if self.cancel.is_set():
                raise Cancelled()
            seg = segs[k]
            L.stage(f"Segment {k + 1}/{len(segs)} – preparing input")
            ref = ref_dir / f"ref_{k:04d}.mkv"
            self._write_reference(info, geo, seg, js, ref)
            client = self._ensure_client(js, ref, L, gpu_total_gb)
            seg_dir = job_dir / f"seg_{k:04d}"
            if seg_dir.exists():
                shutil.rmtree(seg_dir)
            L.stage(f"Segment {k + 1}/{len(segs)} – AlphaGen inference ({seg.model_frames} frames)")
            t0 = time.time()

            def on_ev(ev, _base=done_frames, _seg=seg, _k=k):
                if ev.get("ev") == "frames":
                    L.progress(_base + int(ev.get("written", 0)), total_model_frames,
                               _eta(sec_per_frame, total_model_frames - _base - int(ev.get("written", 0))),
                               f"Segment {_k + 1}/{len(segs)}: decoded {ev.get('written')}/{_seg.model_frames}")

            req = {"id": k, "video": str(ref), "out_dir": str(seg_dir), "cli_width": geo.cli_w,
                   "cli_height": geo.cli_h, "num_frames": seg.model_frames, "fps": info.fps_float, "seed": js.seed}
            ev = client.run_segment(req, cancel=self.cancel, on_event=on_ev)
            dt = time.time() - t0
            sec_per_frame.append(dt / seg.model_frames)
            if n > 0 or len(todo) == 1:  # první segment obsahuje načtení modelu – do statistiky jen „čisté“
                self._record_perf(geo, seg.model_frames, dt, js)
            done_frames += seg.model_frames
            L.log("INFO", f"Segment {k + 1}/{len(segs)} done in {dt:.1f} s, peak VRAM (torch) "
                          f"{ev.get('peak_vram_gb')} GB.")
            L.progress(done_frames, total_model_frames, _eta(sec_per_frame, total_model_frames - done_frames),
                       f"Segments done {n + 1}/{len(todo)}")
            L.segment_done(k, len(segs))
            ref.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    def _export(self, info, geo, cache: MatteCache, total, js: JobSettings, L, job_dir) -> dict:  # noqa: N803
        L.stage("Export – blending segments, refining matte, writing")
        out_dir = Path(js.output_dir or self.settings.output_dir)
        base = js.base_name or f"{Path(info.path).stem}_alphagen"
        ffmpeg, ffprobe = self.settings.ffmpeg_path(), self.settings.ffprobe_path()
        ms = js.matte
        exporter = Exporter(info, js.export, out_dir, base, js.in_frame, total, ffmpeg, ffprobe)
        reader = FrameReader(info, ffmpeg, js.in_frame, total)
        smoother = TemporalSmoother(ms.temporal) if ms.temporal > 0 else None
        pending_rgb: list[np.ndarray] = []
        t0 = time.time()
        written = 0

        # Úprava masky (zpřesnění hrany, morfologie, despill) je na snímek nezávislá → paralelně ve vláknech
        # (numpy/OpenCV uvolňují GIL); zápis zůstává v původním pořadí snímků.
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        workers = max(2, min(12, (os.cpu_count() or 4) - 2))
        pool = ThreadPoolExecutor(max_workers=workers)
        inflight: deque = deque()

        def compute(alpha_raw: np.ndarray, rgb: np.ndarray):
            a = process_frame(alpha_raw, rgb, ms)
            color = apply_color(rgb, a, ms) if ms.despill != "off" else None
            return rgb, a, color

        def drain(limit: int) -> None:
            nonlocal written
            while len(inflight) > limit:
                rgb, a, color = inflight.popleft().result()
                exporter.write(rgb, a, color)
                written += 1
                if written % 5 == 0 or written == total:
                    rate = written / max(1e-3, time.time() - t0)
                    L.progress(written, total, (total - written) / rate if written >= 10 else None,
                               f"Exporting frame {written}/{total}")

        def emit(alpha_raw: np.ndarray, rgb: np.ndarray) -> None:
            inflight.append(pool.submit(compute, alpha_raw, rgb))
            drain(workers * 2)

        try:
            for f, rgb in enumerate(reader):
                if self.cancel.is_set():
                    raise Cancelled()
                raw = cache.raw(f)
                if smoother is None:
                    emit(raw, rgb)
                    continue
                pending_rgb.append(rgb.copy())
                for a in smoother.push(raw):
                    emit(a, pending_rgb.pop(0))
            if reader.read != total:
                raise VideoError(f"Read {reader.read}/{total} frames from source. {reader.stderr_text()}")
            if smoother is not None:
                for a in smoother.flush():
                    emit(a, pending_rgb.pop(0))
            drain(0)
            report = exporter.close()
        except BaseException:
            exporter.abort()
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        report.update({"geometry": geo.to_dict(), "segments": len(cache.segs), "job_dir": str(job_dir),
                       "seconds_export": round(time.time() - t0, 1)})
        for key, chk in report.get("checks", {}).items():
            lvl = "INFO" if chk.get("ok") else "ERROR"
            L.log(lvl, f"Check {key}: {'OK' if chk.get('ok') else 'FAILED'} {json.dumps(chk, ensure_ascii=False)}")
        return report


def _perf_file() -> Path:
    from . import config

    return config.CONFIG_DIR / "perf.json"


def load_perf() -> list[dict]:
    """Naměřené časy segmentů na tomto PC (samostatný soubor – GUI nastavení ho nepřepíše)."""
    try:
        return json.loads(_perf_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def save_perf(perf: list[dict]) -> None:
    try:
        _perf_file().parent.mkdir(parents=True, exist_ok=True)
        _perf_file().write_text(json.dumps(perf, indent=1), encoding="utf-8")
    except OSError:
        pass


def _eta(sec_per_frame: list[float], remaining_frames: int) -> float | None:
    """ETA jen z reálně naměřených segmentů (první segment obsahuje i warm-up – vynecháme ho, je-li víc)."""
    if not sec_per_frame:
        return None
    sample = sec_per_frame[1:] if len(sec_per_frame) > 2 else sec_per_frame
    return float(np.mean(sample[-4:])) * max(0, remaining_frames)


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _suggestions(kind: str, js: JobSettings) -> list[str]:
    if kind == "oom":
        out = []
        if js.long_side == 0 or js.long_side > 960:
            out.append("Lower the working resolution (e.g. 960 or 768 px).")
        if js.segment_frames > 49:
            out.append(f"Shorten the segment (e.g. {max(17, js.segment_frames - 48)} frames).")
        if js.offload in ("auto", "none"):
            out.append("Enable weight offload to RAM (cpu) or to disk.")
        return out
    if kind == "missing_model":
        return ["Open the Model Manager and download the missing files."]
    if kind == "missing_env":
        return ["Run setup.bat (installs the LTX-2 inference environment)."]
    return []
