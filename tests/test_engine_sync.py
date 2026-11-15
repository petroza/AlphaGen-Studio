"""End-to-end test synchronizace celé pipeline s TESTOVACÍM workerem (ne AI – viz fake_worker.py)."""

from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from alphagen_studio.config import AppSettings
from alphagen_studio.engine import Engine, JobListener, JobSettings
from tests.conftest import FFMPEG, FFPROBE, decode_index

FAKE = Path(__file__).with_name("fake_worker.py")
COUNTER = Path(__file__).with_name("_fake_worker_runs.txt")


class L(JobListener):
    def __init__(self):
        self.logs = []
        self.report = None
        self.error = None
        self.done = threading.Event()

    def log(self, level, msg):
        self.logs.append((level, msg))

    def finished(self, report):
        self.report = report
        self.done.set()

    def failed(self, message, kind, suggestions):
        self.error = (message, kind)
        self.done.set()


def make_engine(tmp_path) -> Engine:
    s = AppSettings(models_dir=str(tmp_path / "m"), cache_dir=str(tmp_path / "c"), work_dir=str(tmp_path / "w"),
                    output_dir=str(tmp_path / "o"), ffmpeg=FFMPEG, ffprobe=FFPROBE)
    s.ensure_dirs()
    eng = Engine(s)
    eng.models.missing = lambda: []  # testovací worker modely nepotřebuje
    eng.worker_override = (sys.executable, FAKE)
    return eng


def run(eng, path, js):
    lis = L()
    eng.start(str(path), js, lis)
    assert lis.done.wait(300), "timeout"
    return lis


@pytest.mark.parametrize("long_side", [0, 640])
def test_matte_sync_through_segments_and_blend(media, tmp_path, long_side):
    COUNTER.unlink(missing_ok=True)
    eng = make_engine(tmp_path)
    js = JobSettings(in_frame=13, out_frame=13 + 130, long_side=long_side, segment_frames=49, overlap=8,
                     output_dir=str(tmp_path / "o"), base_name="sync")
    js.export.formats = ["matte_png", "prores4444"]
    js.matte.temporal = 0.0
    lis = run(eng, media["mp4"], js)
    eng.shutdown()
    assert lis.error is None, lis.error
    rep = lis.report
    assert rep["segments"] >= 3
    assert rep["checks"]["prores4444"]["ok"] and rep["checks"]["prores4444"]["frames"] == 130
    files = sorted(os.listdir(rep["outputs"]["matte_png"]))
    assert len(files) == 130
    for k in (0, 1, 35, 36, 40, 41, 42, 77, 78, 129):  # včetně snímků v překryvech segmentů
        m = cv2.imread(os.path.join(rep["outputs"]["matte_png"], files[k]), cv2.IMREAD_UNCHANGED)
        g = (m.astype(np.float32) / 257.0)
        idx = decode_index(np.repeat(g[..., None], 3, axis=2).astype(np.uint8))
        assert idx == 13 + k, f"výstupní snímek {k}: maska patří snímku {idx}, čekáno {13 + k}"
        assert files[k].endswith(f"_{13 + k:06d}.png")


def test_resume_only_missing_segment(media, tmp_path):
    COUNTER.unlink(missing_ok=True)
    eng = make_engine(tmp_path)
    js = JobSettings(in_frame=0, out_frame=100, long_side=640, segment_frames=33, overlap=8,
                     output_dir=str(tmp_path / "o"), base_name="res")
    js.export.formats = ["matte_png"]
    lis = run(eng, media["mp4"], js)
    assert lis.error is None, lis.error
    first = COUNTER.read_text().split()
    job_dir = Path(lis.report["job_dir"])
    # simulace přerušení: segment 2 nedokončen
    shutil.rmtree(job_dir / "seg_0002")
    COUNTER.unlink()
    lis = run(eng, media["mp4"], js)
    assert lis.error is None, lis.error
    assert COUNTER.read_text().split() == ["2"]  # přepočítal se jen chybějící segment
    COUNTER.unlink()
    lis = run(eng, media["mp4"], js)  # vše v cache → bez inference
    assert not COUNTER.exists()
    assert any("cache" in m for _, m in lis.logs)
    eng.shutdown()
    assert len(first) >= 3


def test_short_clip_padded_to_8n1(media, tmp_path):
    eng = make_engine(tmp_path)
    js = JobSettings(in_frame=5, out_frame=5 + 12, long_side=512, output_dir=str(tmp_path / "o"), base_name="short")
    js.export.formats = ["matte_png"]
    lis = run(eng, media["hevc"], js)
    eng.shutdown()
    assert lis.error is None, lis.error
    files = sorted(os.listdir(lis.report["outputs"]["matte_png"]))
    assert len(files) == 12  # model dostal 17 snímků (8n+1), ve výstupu jen 12 skutečných
    m = cv2.imread(os.path.join(lis.report["outputs"]["matte_png"], files[-1]), cv2.IMREAD_UNCHANGED)
    g = (m.astype(np.float32) / 257.0).astype(np.uint8)
    assert decode_index(np.repeat(g[..., None], 3, axis=2)) == 16
