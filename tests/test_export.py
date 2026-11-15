from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import json  # noqa: E402
import subprocess  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from alphagen_studio.export import Exporter, ExportSettings  # noqa: E402
from alphagen_studio.postprocess import MatteSettings, TemporalSmoother, decontaminate_edges, process_frame  # noqa: E402
from alphagen_studio.video_io import FrameReader, probe  # noqa: E402
from tests.conftest import FFMPEG, FFPROBE, decode_index  # noqa: E402


def synthetic_alpha(i: int, w: int, h: int) -> np.ndarray:
    """Měkký gradient + kruh – každý snímek jiný, aby šlo ověřit synchronizaci."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.clip((xx / w) * 0.8 + 0.1 * np.sin(i / 3.0), 0, 1)
    a[((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (h / 4) ** 2] = 1.0
    return a.astype(np.float32)


def _decode(path, ffmpeg, w, h, pix, n=None, vf=None):
    cmd = [ffmpeg, "-v", "error", "-i", str(path)]
    if vf:
        cmd += ["-vf", vf]
    if n:
        cmd += ["-frames:v", str(n)]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix, "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    ch = {"rgba64le": 4, "gray16le": 1, "rgb24": 3}[pix]
    dt = np.uint16 if pix in ("rgba64le", "gray16le") else np.uint8
    return np.frombuffer(raw, dtype=dt).reshape(-1, h, w, ch) if ch > 1 else np.frombuffer(raw, dtype=dt).reshape(-1, h, w)


def test_full_export_mp4(media, tmp_path):
    info = probe(media["mp4"], FFPROBE)
    in_f, n = 10, 40
    s = ExportSettings(formats=["prores4444", "rgba_png", "rgba_exr", "matte_png", "matte_mov", "matte_h264", "preview"])
    ex = Exporter(info, s, tmp_path, "test", in_f, n, FFMPEG, FFPROBE)
    src = []
    for i, rgb in enumerate(FrameReader(info, FFMPEG, in_f, n)):
        src.append(rgb.copy())
        ex.write(rgb, synthetic_alpha(in_f + i, info.width, info.height))
    report = ex.close()
    checks = report["checks"]
    assert all(c["ok"] for c in checks.values()), json.dumps(checks, indent=1)

    # ProRes 4444: skutečný alfa kanál, správný počet snímků, alfa = zapsaná maska
    pr = checks["prores4444"]
    assert pr["pix_fmt"].startswith("yuva444p")
    assert pr["frames"] == n and pr["alpha_match"]
    dec = _decode(report["outputs"]["prores4444"], FFMPEG, info.width, info.height, "rgba64le")
    assert dec.shape[0] == n
    for i in (0, 17, n - 1):
        a = dec[i, ..., 3].astype(np.float32) / 65535
        assert np.abs(a - synthetic_alpha(in_f + i, info.width, info.height)).max() < 2 / 1023
        # straight alpha: RGB zůstává originální (YUV 10bit – tolerance) i tam, kde je alfa nízká
        assert decode_index((dec[i, ..., :3] >> 8).astype(np.uint8)) == in_f + i
        assert np.abs((dec[i, ..., :3] >> 8).astype(np.float32) - src[i]).mean() < 1.5
    # zvuk je přítomný a délkou odpovídá
    a = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration",
                        "-of", "csv=p=0", report["outputs"]["prores4444"]], capture_output=True, text=True).stdout
    assert abs(float(a.strip()) - n / 25) < 0.05

    # PNG RGBA (8bit zdroj) = bit-přesné původní RGB + alfa; číslování podle zdrojových snímků
    pngs = sorted(os.listdir(report["outputs"]["rgba_png"]))
    assert len(pngs) == n and pngs[0].endswith(f"_{in_f:06d}.png")
    img = cv2.imread(os.path.join(report["outputs"]["rgba_png"], pngs[5]), cv2.IMREAD_UNCHANGED)
    assert img.dtype == np.uint8 and img.shape[2] == 4
    assert np.array_equal(img[..., 2::-1], src[5])
    assert np.abs(img[..., 3] / 255.0 - synthetic_alpha(in_f + 5, info.width, info.height)).max() <= 0.5 / 255 + 1e-6

    # EXR: RGBA half, premultiplied
    exrs = sorted(os.listdir(report["outputs"]["rgba_exr"]))
    import OpenEXR

    with OpenEXR.File(os.path.join(report["outputs"]["rgba_exr"], exrs[3])) as fh:
        e = fh.channels()["RGBA"].pixels.astype(np.float32)
    assert e.shape == (info.height, info.width, 4)
    al = e[..., 3]
    rgb = e[..., :3]
    expect = src[3].astype(np.float32) / 255 * al[..., None]
    assert np.abs(rgb - expect).max() < 2e-3

    # Matte PNG 16bit
    mp = sorted(os.listdir(report["outputs"]["matte_png"]))
    m = cv2.imread(os.path.join(report["outputs"]["matte_png"], mp[7]), cv2.IMREAD_UNCHANGED)
    assert m.dtype == np.uint16 and m.ndim == 2
    assert np.abs(m / 65535.0 - synthetic_alpha(in_f + 7, info.width, info.height)).max() < 1e-4

    # Luma matte H.264 – snímky sedí a jas odpovídá masce (8bit, ztrátové → tolerance)
    lm = _decode(report["outputs"]["matte_h264"], FFMPEG, info.width, info.height, "gray16le",
                 vf="scale=in_range=tv:out_range=pc,format=gray16le")
    assert lm.shape[0] == n
    for i in (0, n // 2, n - 1):
        assert np.abs(lm[i] / 65535.0 - synthetic_alpha(in_f + i, info.width, info.height)).mean() < 0.01

    # Matte MOV – synchronizace po snímcích
    mm = _decode(report["outputs"]["matte_mov"], FFMPEG, info.width, info.height, "gray16le",
                 vf="scale=in_range=tv:out_range=pc,format=gray16le")
    assert mm.shape[0] == n
    for i in (0, n // 2, n - 1):
        assert np.abs(mm[i] / 65535.0 - synthetic_alpha(in_f + i, info.width, info.height)).mean() < 0.01


def test_prores_high_bit_premultiplied(media, tmp_path):
    info = probe(media["prores"], FFPROBE)
    s = ExportSettings(formats=["prores4444", "rgba_png"], alpha_mode="premultiplied")
    ex = Exporter(info, s, tmp_path, "hb", 0, 12, FFMPEG, FFPROBE)
    src = []
    for i, rgb in enumerate(FrameReader(info, FFMPEG, 0, 12)):
        assert rgb.dtype == np.uint16
        src.append(rgb.copy())
        ex.write(rgb, synthetic_alpha(i, info.width, info.height))
    report = ex.close()
    assert report["checks"]["prores4444"]["ok"]
    png = sorted(os.listdir(report["outputs"]["rgba_png"]))[4]
    img = cv2.imread(os.path.join(report["outputs"]["rgba_png"], png), cv2.IMREAD_UNCHANGED)
    assert img.dtype == np.uint16  # 10bit zdroj → 16bit PNG
    a = img[..., 3].astype(np.float32) / 65535
    exp = src[4].astype(np.float32) * a[..., None]
    assert np.abs(img[..., 2::-1].astype(np.float32) - exp).max() <= 1.0


def test_postprocess_ops():
    a = np.zeros((100, 100), np.float32)
    a[30:70, 30:70] = 1
    ms = MatteSettings(erode_dilate=-3)
    assert process_frame(a, None, ms).sum() < a.sum()
    ms = MatteSettings(erode_dilate=3)
    assert process_frame(a, None, ms).sum() > a.sum()
    ms = MatteSettings(feather=2)
    out = process_frame(a, None, ms)
    assert 0 < out[50, 29] < 1
    ms = MatteSettings(invert=True)
    assert np.allclose(process_frame(a, None, ms), 1 - a)
    ms = MatteSettings(black_point=0.2, white_point=0.8)
    g = np.linspace(0, 1, 100, dtype=np.float32)[None, :].repeat(10, 0)
    out = process_frame(g, None, ms)
    assert out[0, 10] == 0 and out[0, 90] == 1
    rgb = (np.random.default_rng(1).random((100, 100, 3)) * 255).astype(np.uint8)
    ms = MatteSettings(edge_refine=True, edge_radius=3, edge_strength=1.0)
    out = process_frame(a, rgb, ms)
    assert out.shape == a.shape and np.isfinite(out).all()
    # plné popředí se dekontaminací nemění
    soft = a.copy()
    soft[28:30, 30:70] = 0.5
    dc = decontaminate_edges(rgb, soft)
    assert np.allclose(dc[40:60, 40:60], rgb[40:60, 40:60] / 255.0)


def test_temporal_smoother_keeps_frame_count():
    sm = TemporalSmoother(0.5)
    out = []
    for i in range(10):
        a = np.full((20, 20), 0.5 + 0.01 * i, np.float32)
        out += sm.push(a)
    out += sm.flush()
    assert len(out) == 10
