from __future__ import annotations

import numpy as np
import pytest

from alphagen_studio.video_io import FrameReader, probe, read_frame, read_frames
from tests.conftest import FFMPEG, FFPROBE, decode_index, make_frame


def test_probe_mp4(media):
    info = probe(media["mp4"], FFPROBE)
    assert (info.width, info.height) == (1280, 720)
    assert info.fps_num == 25 and info.fps_den == 1
    assert info.nb_frames == 150
    assert info.codec == "h264"
    assert info.has_audio
    assert info.bit_depth == 8


def test_probe_prores_high_bit(media):
    info = probe(media["prores"], FFPROBE)
    assert info.codec == "prores"
    assert info.bit_depth == 10 and info.high_bit_depth
    assert abs(info.fps_float - 23.976) < 0.001
    assert info.nb_frames == 60


def test_probe_sequence(media):
    info = probe(media["seq"], FFPROBE)
    assert info.is_sequence and info.seq_start == 1001
    assert info.nb_frames == 30
    assert (info.width, info.height) == (320, 240)


def test_rotation_metadata_applied(media):
    info = probe(media["rotated"], FFPROBE)
    assert info.rotation in (90, 270)
    assert (info.width, info.height) == (720, 1280)
    f = read_frame(info, FFMPEG, 0)
    assert f.shape == (1280, 720, 3)


@pytest.mark.parametrize("key", ["mp4", "prores", "hevc", "seq"])
@pytest.mark.parametrize("start", [0, 1, 7, 24, 29])
def test_frame_accurate_seek(media, key, start):
    """Snímek přečtený od libovolného startu musí být přesně ten snímek (B-snímky, 23.976, sekvence)."""
    info = probe(media[key], FFPROBE)
    if start + 3 > info.nb_frames:
        pytest.skip("krátký zdroj")
    frames = read_frames(info, FFMPEG, start, 3)
    assert [decode_index(f) for f in frames] == [start, start + 1, start + 2]


def test_sequential_read_complete(media):
    info = probe(media["mp4"], FFPROBE)
    idx = [decode_index(f) for f in FrameReader(info, FFMPEG, 100, 50)]
    assert idx == list(range(100, 150))


def test_high_bit_read_returns_uint16(media):
    info = probe(media["prores"], FFPROBE)
    f = read_frame(info, FFMPEG, 5)
    assert f.dtype == np.uint16
    ref = make_frame(5, 640, 360, np.uint16).astype(np.float32)
    # 10bit ProRes 4:2:2 – kruh a pozadí musí sedět barevně (správná matice BT.709, plný rozsah)
    assert np.abs(f.astype(np.float32)[200:300, 400:500] - ref[200:300, 400:500]).mean() < 700


def test_color_matrix_bt709(media):
    """Dekódování musí použít BT.709 (ne tichý BT.601) – barva pozadí se nesmí posunout."""
    info = probe(media["mp4"], FFPROBE)
    f = read_frame(info, FFMPEG, 0).astype(np.float32)
    ref = make_frame(0, 1280, 720).astype(np.float32)
    assert np.abs(f[600:700, 50:150] - ref[600:700, 50:150]).mean() < 2.0
