from __future__ import annotations

import numpy as np
import pytest

from alphagen_studio.segmentation import (
    HARD_MAX_FRAMES,
    PlanError,
    blend_plan,
    ceil_8n1,
    compute_geometry,
    crop_pad,
    is_valid_frame_count,
    pad_frame,
    plan_segments,
)


@pytest.mark.parametrize(("w", "h", "exp"), [
    (1920, 1080, (1920, 1080, 1920, 1088, 3840, 2176)),
    (1280, 720, (1280, 720, 1280, 736, 2560, 1472)),
    (3840, 2160, (1920, 1080, 1920, 1088, 3840, 2176)),
    (1080, 1920, (1080, 1920, 1088, 1920, 2176, 3840)),
    (720, 576, (720, 576, 736, 576, 1472, 1152)),
])
def test_geometry(w, h, exp):
    g = compute_geometry(w, h)
    assert (g.work_w, g.work_h, g.pad_w, g.pad_h, g.cli_w, g.cli_h) == exp
    assert g.pad_w % 32 == 0 and g.pad_h % 32 == 0
    assert g.cli_w == 2 * g.pad_w and g.cli_h == 2 * g.pad_h  # --skip-stage-2 = polovina
    assert max(g.pad_w, g.pad_h) <= 1920 and min(g.pad_w, g.pad_h) <= 1088


def test_geometry_long_side_limit_keeps_aspect():
    g = compute_geometry(1920, 1080, 768)
    assert g.work_w == 768 and g.work_h == 432
    assert g.pad_h == 448 and g.downscaled


def test_pad_and_crop_roundtrip():
    g = compute_geometry(1280, 720)
    img = np.random.default_rng(0).random((720, 1280)).astype(np.float32)
    padded = pad_frame(img, g)
    assert padded.shape == (736, 1280)
    assert np.array_equal(crop_pad(padded, g), img)


def test_8n1():
    assert ceil_8n1(1) == 1 and ceil_8n1(2) == 9 and ceil_8n1(121) == 121 and ceil_8n1(122) == 129
    assert all(is_valid_frame_count(n) for n in (1, 9, 17, 97, 121, 145))


@pytest.mark.parametrize("total", [1, 5, 9, 120, 121, 122, 200, 241, 250, 1000, 7777])
@pytest.mark.parametrize(("seg", "ov"), [(121, 16), (121, 0), (49, 8), (97, 24)])
def test_plan_covers_every_frame(total, seg, ov):
    segs = plan_segments(total, seg, ov)
    for s in segs:
        assert is_valid_frame_count(s.model_frames)
        assert s.model_frames <= HARD_MAX_FRAMES
        assert s.length <= s.model_frames
        assert s.start >= 0 and s.end <= total
    assert segs[0].start == 0 and segs[-1].end == total
    plan = blend_plan(segs, total)
    for f, items in enumerate(plan):
        assert items, f"snímek {f} nepokryt"
        assert abs(sum(w for _, _, w in items) - 1.0) < 1e-5
        for k, i, _ in items:
            assert segs[k].start + i == f  # přesné časové zarovnání
    for a, b in zip(segs, segs[1:]):
        assert a.end - b.start >= ov  # překryv alespoň požadovaný


def test_invalid_segment_lengths():
    with pytest.raises(PlanError):
        plan_segments(100, 120, 8)
    with pytest.raises(PlanError):
        plan_segments(500, 153, 8)


def test_blend_reconstructs_signal_and_is_continuous():
    """Segmenty s vlastní odchylkou (jako by model v každém úseku mírně ujel) – prolnutí nesmí skočit."""
    total = 400
    segs = plan_segments(total, 121, 16)
    truth = np.sin(np.arange(total) / 20.0)
    bias = {k: (0.1 if k % 2 else -0.1) for k in range(len(segs))}
    plan = blend_plan(segs, total)
    out = np.array([sum(w * (truth[segs[k].start + i] + bias[k]) for k, i, w in items) for items in plan])
    # bez odchylky = přesná rekonstrukce
    exact = np.array([sum(w * truth[segs[k].start + i] for k, i, w in items) for items in plan])
    assert np.allclose(exact, truth)
    # s odchylkou: maximální skok mezi snímky ≈ rozdíl odchylek / délka prolnutí (žádný skok 0.2)
    jumps = np.abs(np.diff(out - truth))
    assert jumps.max() < 0.2 / 8
