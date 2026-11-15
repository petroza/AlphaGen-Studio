"""Příkazová řádka (smoke test, dávkové zpracování).

  python -m alphagen_studio.cli check
  python -m alphagen_studio.cli plan  vstup.mp4 [--long-side 960]
  python -m alphagen_studio.cli run   vstup.mp4 [--in 0 --out 49 --long-side 768 --segment 49 --formats prores4444,matte_png]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

from .backend.client import check_environment
from .config import AppSettings
from .engine import Engine, JobListener, JobSettings
from .export import FORMAT_LABELS
from .gpu import list_gpus, ram_info, recommend
from .logging_setup import setup_logging
from .models import ModelManager
from .segmentation import compute_geometry, plan_segments
from .video_io import probe


class PrintListener(JobListener):
    def __init__(self) -> None:
        self.done = threading.Event()
        self.ok = False
        self.report: dict = {}

    def log(self, level, msg):
        if level != "PROGRESS":
            print(f"[{level}] {msg}", flush=True)

    def stage(self, text):
        print(f"== {text}", flush=True)

    def progress(self, done, total, eta_s, detail):
        eta = f", ~{eta_s / 60:.1f} min left" if eta_s else ""
        print(f"   {done}/{total} {detail}{eta}", flush=True)

    def finished(self, report):
        self.ok, self.report = True, report
        print(json.dumps(report, indent=2, ensure_ascii=False))
        self.done.set()

    def failed(self, message, kind, suggestions):
        print(f"!! ERROR ({kind}): {message}", flush=True)
        for s in suggestions:
            print(f"   → {s}")
        self.done.set()


def cmd_check(s: AppSettings) -> int:
    print("FFmpeg :", s.ffmpeg_path() or "NOT FOUND")
    print("FFprobe:", s.ffprobe_path() or "NOT FOUND")
    gpus = list_gpus()
    avail, total = ram_info()
    print(f"RAM    : {avail:.1f} / {total:.1f} GB free")
    for g in gpus:
        plan = recommend(g, total)
        print("GPU    :", g.label(), f"driver {g.driver}")
        print(f"         recommended: quantization={plan.quantization}, offload={plan.offload}, "
              f"resolution ≤{plan.recommended_long_side}px, segment {plan.recommended_segment}")
        for n in plan.notes:
            print("         -", n)
    if not gpus:
        print("GPU    : no NVIDIA GPU found")
    env = check_environment(s.ltx_python_path())
    print("LTX env:", json.dumps(env, ensure_ascii=False))
    mm = ModelManager(s)
    for st in mm.status():
        flag = "OK " if st["size_ok"] else ("PART" if st["partial"] else "MISSING")
        print(f"Model  : [{flag}] {st['label']}  ({st['expected'] / 1024**3:.2f} GB)  {st['path']}")
    return 0 if env.get("ok") and mm.all_present() and gpus else 1


def cmd_plan(s: AppSettings, a) -> int:
    info = probe(a.input, s.ffprobe_path())
    print(info.summary_cs())
    for w in info.warnings:
        print("! ", w)
    geo = compute_geometry(info.width, info.height, a.long_side or None)
    print(geo.describe_cs(), f"| CLI --width {geo.cli_w} --height {geo.cli_h} (--skip-stage-2)")
    total = (a.out or info.nb_frames) - a.in_frame
    for seg in plan_segments(total, a.segment, a.overlap):
        print(f"  segment {seg.index}: frames {seg.start}–{seg.end - 1} ({seg.length}), model {seg.model_frames}, "
              f"blend in {seg.ramp_in} / out {seg.ramp_out}")
    return 0


def cmd_run(s: AppSettings, a) -> int:
    js = JobSettings(in_frame=a.in_frame, out_frame=a.out, long_side=a.long_side, segment_frames=a.segment,
                     overlap=a.overlap, seed=a.seed, gpu_index=a.gpu, quantization=a.quantization, offload=a.offload,
                     output_dir=a.output or "", vae=a.vae)
    js.export.formats = [f.strip() for f in a.formats.split(",") if f.strip()]
    bad = [f for f in js.export.formats if f not in FORMAT_LABELS]
    if bad:
        print("Unknown formats:", bad, "allowed:", list(FORMAT_LABELS))
        return 2
    gpus = {g.index: g for g in list_gpus()}
    g = gpus.get(a.gpu)
    if a.profile:  # stejné nastavení jako režim Automatic v GUI
        from .gpu import ram_info
        from .profiles import apply_profile

        info = probe(a.input, s.ffprobe_path())
        apply_profile(js, a.profile, info.width, info.height, g, ram_info()[1])
        print(f"Profile {a.profile}: long side {js.long_side or 'full'}, segment {js.segment_frames}, "
              f"overlap {js.overlap}, refine edges {js.matte.edge_refine}")
    eng = Engine(s)
    eng.gpu_total_gb = g.total_gb if g else 0.0
    listener = PrintListener()
    t0 = time.time()
    eng.start(a.input, js, listener, export_only=a.export_only, gpu_total_gb=g.total_gb if g else 0.0)
    try:
        while not listener.done.wait(0.5):
            pass
    except KeyboardInterrupt:
        print("Cancelling…")
        eng.request_cancel()
        listener.done.wait(60)
    finally:
        eng.shutdown()
    print(f"Total {time.time() - t0:.1f} s")
    return 0 if listener.ok else 1


def cmd_queue(s: AppSettings) -> int:
    """Zpracuje uloženou frontu z GUI (config/queue.json) bez okna – např. z Plánovače úloh."""
    import threading as _th

    from .jobqueue import JobQueue, QueueRunner

    q = JobQueue()
    print(f"Queue: {sum(1 for i in q.items if i.status == 'waiting')} waiting job(s)")
    eng = Engine(s)
    done = _th.Event()
    runner = QueueRunner(eng, q, PrintListener(), on_finished=lambda c: (print("Queue finished:", c), done.set()))
    runner.start()
    try:
        while not done.wait(0.5):
            pass
    except KeyboardInterrupt:
        print("Stopping the queue…")
        runner.stop()
        done.wait(120)
    finally:
        eng.shutdown()
    return 0 if not any(i.status == "failed" for i in q.items) else 1


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    s = AppSettings.load()
    s.ensure_dirs()
    p = argparse.ArgumentParser(prog="alphagen", description="AlphaGen Studio – CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="Check FFmpeg, GPU, inference environment and models")
    sub.add_parser("queue", help="Run the saved job queue (from the GUI) without a window")
    for name in ("plan", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("--in", dest="in_frame", type=int, default=0)
        sp.add_argument("--out", type=int, default=None, help="exclusive end (frame)")
        sp.add_argument("--long-side", type=int, default=0)
        sp.add_argument("--segment", type=int, default=121)
        sp.add_argument("--overlap", type=int, default=16)
        if name == "run":
            sp.add_argument("--seed", type=int, default=1234)
            sp.add_argument("--gpu", type=int, default=0)
            sp.add_argument("--quantization", default="auto")
            sp.add_argument("--offload", default="auto")
            sp.add_argument("--formats", default="prores4444,matte_png,preview")
            sp.add_argument("--output", default="")
            sp.add_argument("--export-only", action="store_true")
            sp.add_argument("--vae", choices=["auto", "diffvae", "conv"], default="auto")
            sp.add_argument("--profile", choices=["turbo", "fast", "normal", "high"], default=None,
                            help="speed profile like the GUI Automatic mode (overrides long-side/segment/overlap)")
    a = p.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(s)
    if a.cmd == "queue":
        return cmd_queue(s)
    if a.cmd == "plan":
        return cmd_plan(s, a)
    return cmd_run(s, a)


if __name__ == "__main__":
    sys.exit(main())
