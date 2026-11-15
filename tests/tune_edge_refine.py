"""Ladění zpřesnění hrany (guided filter) pro Turbo/Fast proti Normal jako referenci.

Bere surové (neupravené) masky z cache úloh srovnání profilů a měří odchylku od Normal
v okrajovém pásmu pro různé parametry. Spouští se jen na CPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alphagen_studio.engine import MatteCache  # noqa: E402
from alphagen_studio.postprocess import refine_edges  # noqa: E402
from alphagen_studio.segmentation import Geometry, Segment  # noqa: E402

CLIPS = ["vlasy_obloha_1080", "muz_zed_1080", "kudrny_720"]
FRAMES = [6, 18, 30, 42]


def caches(clip: str) -> dict[int, MatteCache]:
    out = {}
    for jd in (ROOT / "work").glob(f"{clip}_*"):
        meta = json.loads((jd / "job.json").read_text(encoding="utf-8"))
        if meta["total"] != 49 or not (jd / "seg_0000" / "DONE").exists():
            continue
        out[meta["key"]["geo"]["work_w"]] = MatteCache(jd, [Segment(**s) for s in meta["segments"]],
                                                      Geometry(**meta["geometry"]), meta["total"])
    return out


def frame(clip: str, idx: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(ROOT / "tests" / "data" / "cmp" / f"{clip}.mp4"),
                          "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "pipe:1"], capture_output=True, check=True).stdout
    return raw


def main() -> None:
    params = [(0, 0.0)] + [(r, s) for r in (2, 3, 5, 8) for s in (0.3, 0.5, 0.8, 1.0)]
    results: dict[tuple[int, int, float], list[float]] = {}
    for clip in CLIPS:
        cs = caches(clip)
        ref_w = max(cs)
        ref = cs[ref_w]
        for w, c in cs.items():
            if w == ref_w:
                continue
            for f in FRAMES:
                r = ref.raw(f)
                h, wd = r.shape
                rgb = np.frombuffer(frame(clip, f), np.uint8).reshape(h, wd, 3)
                band = cv2.dilate(((r > 0.02) & (r < 0.98)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                low = c.raw(f)
                for rad, st in params:
                    out = low if st == 0 else refine_edges(low, rgb, rad, st)
                    results.setdefault((w, rad, st), []).append(float(np.abs(out - r)[band].mean()))
    by_w: dict[int, list] = {}
    for (w, rad, st), v in results.items():
        by_w.setdefault(w, []).append((np.mean(v), rad, st))
    for w, rows in sorted(by_w.items()):
        rows.sort()
        base = next(m for m, r_, s_ in rows if s_ == 0)
        print(f"work width {w}: bez zpřesnění {base:.4f}")
        for m, rad, st in rows[:5]:
            print(f"   radius {rad}, strength {st}: {m:.4f}  ({(1 - m / base) * 100:+.1f} % lepší)")


if __name__ == "__main__":
    main()
