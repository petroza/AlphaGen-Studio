"""Syntetická testovací média: každý snímek nese své číslo ve dvou blocích (jas),
takže lze ověřit snímkově přesné čtení i synchronizaci výstupů."""

from __future__ import annotations

import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphagen_studio.config import find_tool  # noqa: E402
from alphagen_studio.video_io import FFmpegWriter  # noqa: E402

FFMPEG = find_tool("ffmpeg")
FFPROBE = find_tool("ffprobe")
BLOCK = 48


def make_frame(i: int, w: int, h: int, dtype=np.uint8) -> np.ndarray:
    """Pozadí + pohybující se kruh (subjekt) + 2 bloky s kódem čísla snímku."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[...] = (0.25, 0.35, 0.45)
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w * (0.3 + 0.4 * ((i % 50) / 50.0))
    r = min(w, h) * 0.22
    circle = ((xx - cx) ** 2 + (yy - h / 2) ** 2) <= r * r
    img[circle] = (0.9, 0.8, 0.7)
    a, b = i % 20, i // 20
    img[8:8 + BLOCK, 8:8 + BLOCK] = (20 + a * 10) / 255.0
    img[8:8 + BLOCK, 16 + BLOCK:16 + 2 * BLOCK] = (20 + b * 10) / 255.0
    if dtype == np.uint16:
        return np.round(img * 65535).astype(np.uint16)
    return np.round(img * 255).astype(np.uint8)


def decode_index(frame: np.ndarray) -> int:
    f = frame.astype(np.float32)
    if frame.dtype == np.uint16:
        f = f / 257.0
    a = f[8 + 12:8 + BLOCK - 12, 8 + 12:8 + BLOCK - 12].mean()
    b = f[8 + 12:8 + BLOCK - 12, 16 + BLOCK + 12:16 + 2 * BLOCK - 12].mean()
    return int(round((a - 20) / 10)) + 20 * int(round((b - 20) / 10))


def _write_video(path: Path, n: int, w: int, h: int, fps: Fraction, args: list[str], audio: bool = False,
                 high: bool = False) -> Path:
    extra = ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={n / float(fps) + 1}"] if audio else None
    maps = ["-map", "0:v:0", "-map", "1:a:0", "-shortest"] if audio else None
    wr = FFmpegWriter(FFMPEG, path, w, h, "rgb48le" if high else "rgb24", fps, args, extra, maps)
    for i in range(n):
        wr.write(make_frame(i, w, h, np.uint16 if high else np.uint8))
    wr.close()
    return path


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Testy nikdy nesmí přepsat skutečné config/settings.json uživatele."""
    from alphagen_studio import config, models

    cfg = tmp_path / "config"
    monkeypatch.setattr(config, "CONFIG_DIR", cfg)
    monkeypatch.setattr(config, "SETTINGS_FILE", cfg / "settings.json")
    monkeypatch.setattr(models, "CONFIG_DIR", cfg)
    monkeypatch.setattr(models, "MANIFEST_CACHE", cfg / "model_manifest.json")


@pytest.fixture(scope="session")
def media(tmp_path_factory) -> dict:
    if not FFMPEG or not FFPROBE:
        pytest.skip("FFmpeg není k dispozici")
    d = tmp_path_factory.mktemp("media")
    out = {}
    out["mp4"] = _write_video(d / "h264_720p.mp4", 150, 1280, 720, Fraction(25),
                              ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-bf", "3", "-g", "25",
                               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                               "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-c:a", "aac"], audio=True)
    out["prores"] = _write_video(d / "prores_10bit.mov", 60, 640, 360, Fraction(24000, 1001),
                                 ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
                                  "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-colorspace", "bt709"],
                                 high=True)
    out["hevc"] = _write_video(d / "hevc.mp4", 40, 640, 352, Fraction(30),
                               ["-c:v", "libx265", "-crf", "14", "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
                                "-x265-params", "log-level=error",
                                "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-colorspace", "bt709"])
    seq = d / "seq"
    seq.mkdir()
    import cv2

    for i in range(30):
        cv2.imwrite(str(seq / f"shot_{1001 + i:04d}.png"), make_frame(i, 320, 240)[..., ::-1])
    out["seq"] = seq / "shot_1001.png"
    # Rotace z metadat (mobilní video natočené na výšku)
    rot = d / "rotated.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-display_rotation", "90", "-i", str(out["mp4"]), "-c", "copy",
                    str(rot)], check=True)
    out["rotated"] = rot
    return out
