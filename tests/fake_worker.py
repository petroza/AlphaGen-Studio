"""TESTOVACÍ náhrada inference workeru – NENÍ to AI a nikdy se nepoužívá v aplikaci.

Mluví stejným protokolem jako ``backend/worker_main.py``, ale jako "masku" vrací
červený kanál referenčního klipu, který mu engine poslal. Díky tomu lze end-to-end
ověřit synchronizaci: padding, zmenšení, 8n+1 doplnění, prolnutí segmentů a export
musí vrátit masku, jejíž zakódované číslo snímku odpovídá číslu zdrojového snímku.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from alphagen_studio.config import find_tool  # noqa: E402

FFMPEG = find_tool("ffmpeg")
COUNTER = Path(__file__).with_name("_fake_worker_runs.txt")


def emit(ev, **kw):
    sys.stdout.write("@@AG " + json.dumps({"ev": ev, **kw}) + "\n")
    sys.stdout.flush()


def main():
    emit("ready", pid=0, python=sys.version.split()[0])
    for line in sys.stdin:
        req = json.loads(line)
        if req["cmd"] == "load":
            emit("loaded", seconds=0.0)
        elif req["cmd"] == "run":
            w, h = req["cli_width"] // 2, req["cli_height"] // 2
            raw = subprocess.run([FFMPEG, "-v", "error", "-i", req["video"], "-f", "rawvideo", "-pix_fmt", "rgb24",
                                  "pipe:1"], capture_output=True, check=True).stdout
            frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
            if len(frames) != req["num_frames"]:
                emit("error", kind="other", message=f"ref má {len(frames)} snímků, čekáno {req['num_frames']}")
                continue
            out = Path(req["out_dir"])
            out.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                cv2.imwrite(str(out / f"{i:06d}.png"), f[..., 0].astype(np.uint16) * 257)
            (out / "DONE").write_text("{}")
            with COUNTER.open("a") as fh:
                fh.write(f"{req['id']}\n")
            emit("done", id=req["id"], frames=len(frames), seconds=0.01, peak_vram_gb=0)
        elif req["cmd"] == "quit":
            emit("bye")
            return


if __name__ == "__main__":
    main()
