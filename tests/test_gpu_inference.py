"""SKUTEČNÁ AlphaGen inference – vyžaduje GPU, stažené modely a prostředí LTX-2.

Spuštění:  run_tests.bat -m gpu      (bez modelů se test přeskočí)
"""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np
import pytest

from alphagen_studio.config import AppSettings
from alphagen_studio.engine import Engine, JobListener, JobSettings
from alphagen_studio.gpu import list_gpus

pytestmark = pytest.mark.gpu


class L(JobListener):
    def __init__(self):
        self.report, self.error, self.done = None, None, threading.Event()

    def log(self, level, msg):
        print(f"[{level}] {msg}")

    def finished(self, report):
        self.report = report
        self.done.set()

    def failed(self, message, kind, suggestions):
        self.error = (message, kind, suggestions)
        self.done.set()


def test_real_alphagen_short_clip(media, tmp_path):
    s = AppSettings.load()
    eng = Engine(s)
    if eng.models.missing():
        pytest.skip("Modely AlphaGen nejsou stažené")
    gpus = list_gpus()
    if not gpus:
        pytest.skip("Bez NVIDIA GPU")
    js = JobSettings(in_frame=0, out_frame=17, long_side=512, segment_frames=17, overlap=0,
                     output_dir=str(tmp_path), base_name="gpu")
    js.export.formats = ["matte_png", "prores4444"]
    lis = L()
    eng.start(str(media["mp4"]), js, lis, gpu_total_gb=gpus[0].total_gb)
    assert lis.done.wait(3 * 3600)
    eng.shutdown()
    assert lis.error is None, lis.error
    files = sorted(os.listdir(lis.report["outputs"]["matte_png"]))
    assert len(files) == 17
    m = cv2.imread(os.path.join(lis.report["outputs"]["matte_png"], files[8]), cv2.IMREAD_UNCHANGED)
    assert m.std() > 1000, "maska je konstantní – model nic neoddělil"
    assert lis.report["checks"]["prores4444"]["ok"]
    _ = np  # (pro případné rozšíření o metriky)
