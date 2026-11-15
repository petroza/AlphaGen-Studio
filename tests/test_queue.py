"""Dávková fronta: postupné zpracování, vadný job frontu nezastaví, perzistence a obnova po pádu."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from alphagen_studio.config import AppSettings
from alphagen_studio.engine import Engine, JobListener, JobSettings
from alphagen_studio.jobqueue import DONE, FAILED, RUNNING, WAITING, JobQueue, QueueItem, QueueRunner
from tests.conftest import FFMPEG, FFPROBE

FAKE = Path(__file__).with_name("fake_worker.py")


class L(JobListener):
    def __init__(self):
        self.logs = []

    def log(self, level, msg):
        self.logs.append((level, msg))


def test_queue_runs_all_and_survives_failure(media, tmp_path):
    s = AppSettings(models_dir=str(tmp_path / "m"), cache_dir=str(tmp_path / "c"), work_dir=str(tmp_path / "w"),
                    output_dir=str(tmp_path / "o"), ffmpeg=FFMPEG, ffprobe=FFPROBE)
    s.ensure_dirs()
    eng = Engine(s)
    eng.models.missing = lambda: []
    eng.worker_override = (sys.executable, FAKE)
    q = JobQueue()
    assert q.items == []
    base = JobSettings(in_frame=0, out_frame=20, long_side=512, segment_frames=17, overlap=0,
                       output_dir=str(tmp_path / "o"))
    base.export.formats = ["matte_png"]
    q.add(QueueItem(str(media["mp4"]), base.to_dict(), label="a"))
    q.add(QueueItem(str(tmp_path / "missing.mp4"), base.to_dict(), label="broken"))
    q.add(QueueItem(str(media["hevc"]), base.to_dict(), profile="turbo", label="b"))
    done = threading.Event()
    runner = QueueRunner(eng, q, L(), on_finished=lambda c: done.set())
    runner.start()
    assert done.wait(300)
    st = {i.label: i.status for i in q.items}
    assert st == {"a": DONE, "broken": FAILED, "b": DONE}, [(i.label, i.status, i.message) for i in q.items]
    assert all(i.outputs.get("matte_png") for i in q.items if i.status == DONE)
    # perzistence: nová instance načte stejnou frontu; rozběhnutý job se vrátí do čekajících
    q.items[1].status = RUNNING
    q.save()
    q2 = JobQueue()
    assert [i.label for i in q2.items] == ["a", "broken", "b"]
    assert q2.items[1].status == WAITING
    q2.retry_failed()
    q2.clear_finished()
    assert [i.label for i in q2.items] == ["broken"]
