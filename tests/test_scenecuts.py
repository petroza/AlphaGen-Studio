"""Detekce střihů a plánování úseků po záběrech."""

from __future__ import annotations

import os
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from alphagen_studio.engine import JobSettings
from alphagen_studio.scenecuts import _features, find_cuts
from alphagen_studio.segmentation import blend_plan, plan_segments_shots
from alphagen_studio.video_io import FFmpegWriter
from tests.conftest import BLOCK, FFMPEG, decode_index, make_frame

rng = np.random.default_rng(1)


def shot(n: int, color, w=128, h=72, motion=2):
    """Záběr: textura v barvě ``color`` posouvající se o ``motion`` px na snímek."""
    base = cv2.GaussianBlur((rng.random((h, w * 3, 3)) * 160).astype(np.float32), (0, 0), 3)
    out = []
    for i in range(n):
        crop = base[:, i * motion:i * motion + w]
        out.append(np.clip(crop + np.array(color, np.float32), 0, 255).astype(np.uint8))
    return out


def feats(frames):
    return [_features(f) for f in frames]


def test_hard_cuts_found_exactly():
    frames = shot(30, (30, 120, 40)) + shot(25, (180, 60, 60)) + shot(40, (40, 60, 190))
    assert find_cuts(feats(frames)) == [30, 55]


def test_single_shot_motion_and_flash_are_not_cuts():
    frames = shot(60, (100, 100, 100), motion=4)
    frames[30] = np.full_like(frames[30], 250)  # záblesk (blesk fotoaparátu)
    assert find_cuts(feats(frames)) == []


def test_fast_changing_content_without_cut():
    # každý snímek jiný šum stejné barvy – „rychlý pohyb“, žádný střih
    frames = [np.clip(rng.random((72, 128, 3)) * 255, 0, 255).astype(np.uint8) for _ in range(40)]
    assert find_cuts(feats(frames)) == []


def test_very_short_shots_merged():
    frames = shot(30, (30, 120, 40)) + shot(3, (200, 200, 30)) + shot(30, (40, 60, 190))
    cuts = find_cuts(feats(frames))
    assert len(cuts) == 1 and cuts[0] in (30, 33)


def test_plan_never_crosses_cut():
    cuts = [70, 75, 200]
    segs = plan_segments_shots(260, 49, 8, cuts)
    bounds = [0, 70, 75, 200, 260]
    for s in segs:
        assert any(a <= s.start and s.end <= b for a, b in zip(bounds, bounds[1:])), s
        assert (s.model_frames - 1) % 8 == 0
        if s.start in cuts:
            assert s.ramp_in == 0
    plan = blend_plan(segs, 260)
    for c in cuts:  # snímek na střihu i těsně před ním jen z jednoho segmentu (bez prolnutí)
        assert len(plan[c]) == 1 and len(plan[c - 1]) == 1
    assert [i.index for i in segs] == list(range(len(segs)))


def test_plan_without_cuts_matches_plain():
    from alphagen_studio.segmentation import plan_segments

    assert plan_segments_shots(300, 49, 8, []) == plan_segments(300, 49, 8)


def test_engine_splits_at_cut(tmp_path):
    from tests.test_engine_sync import make_engine, run

    path = tmp_path / "cut.mp4"
    wr = FFmpegWriter(FFMPEG, path, 640, 360, "rgb24", Fraction(25),
                      ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"])
    for i in range(120):
        f = make_frame(i, 640, 360)
        if i >= 60:  # druhý záběr: jiný obraz pod kódem snímku
            f[BLOCK + 16:] = 255 - f[BLOCK + 16:]
        wr.write(f)
    wr.close()
    eng = make_engine(tmp_path)
    js = JobSettings(in_frame=0, out_frame=120, segment_frames=49, overlap=8,
                     output_dir=str(tmp_path / "o"), base_name="cut")
    js.export.formats = ["matte_png"]
    lis = run(eng, path, js)
    eng.shutdown()
    assert lis.error is None, lis.error
    assert any("scene cut" in m for _, m in lis.logs), lis.logs
    files = sorted(os.listdir(lis.report["outputs"]["matte_png"]))
    assert len(files) == 120
    for k in (0, 58, 59, 60, 61, 119):
        m = cv2.imread(os.path.join(lis.report["outputs"]["matte_png"], files[k]), cv2.IMREAD_UNCHANGED)
        g = (m.astype(np.float32) / 257.0).astype(np.uint8)
        assert decode_index(np.repeat(g[..., None], 3, axis=2)) == k
    import json

    meta = json.loads((Path(lis.report["job_dir"]) / "job.json").read_text(encoding="utf-8"))
    assert meta["key"]["cuts"] == [60]
    assert all(s["start"] + s["length"] <= 60 or s["start"] >= 60 for s in meta["segments"])


if __name__ == "__main__":
    sys.exit(0)
