"""Geometrie a časové dělení pro AlphaGen.

Pravidla z oficiální dokumentace (docs.ltx.io → AlphaGen, model card):
* rozměry dělitelné 32, maximum 1920×1088, doplnit paddingem (ne deformovat),
* počet snímků 8n+1, doporučeno ≤ 121, nad 145 nepodporováno (do matte prosakuje RGB),
* ``--skip-stage-2``: CLI šířka/výška = 2× rozměr zdroje (stage 1 běží v polovině).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

MODEL_MAX_W = 1920
MODEL_MAX_H = 1088
DIM_MULTIPLE = 32
RECOMMENDED_MAX_FRAMES = 121
HARD_MAX_FRAMES = 145


class PlanError(ValueError):
    pass


def is_valid_frame_count(n: int) -> bool:
    return n >= 1 and (n - 1) % 8 == 0


def floor_8n1(n: int) -> int:
    """Největší 8n+1 <= n (min. 1)."""
    if n < 1:
        raise PlanError("Frame count must be positive.")
    return ((n - 1) // 8) * 8 + 1


def ceil_8n1(n: int) -> int:
    """Nejmenší 8n+1 >= n."""
    if n < 1:
        raise PlanError("Frame count must be positive.")
    return math.ceil((n - 1) / 8) * 8 + 1


# ----------------------------------------------------------------------------
# Prostor
# ----------------------------------------------------------------------------
@dataclass
class Geometry:
    src_w: int
    src_h: int
    work_w: int  # obraz po zmenšení (bez paddingu)
    work_h: int
    pad_w: int  # plátno dělitelné 32 (to dostane model)
    pad_h: int
    pad_left: int
    pad_top: int
    cli_w: int  # hodnota --width pro ic_lora se --skip-stage-2 (= 2 × pad_w)
    cli_h: int
    scale: float  # work / src

    @property
    def downscaled(self) -> bool:
        return self.work_w != self.src_w or self.work_h != self.src_h

    @property
    def pad_right(self) -> int:
        return self.pad_w - self.work_w - self.pad_left

    @property
    def pad_bottom(self) -> int:
        return self.pad_h - self.work_h - self.pad_top

    def to_dict(self) -> dict:
        return asdict(self)

    def describe_cs(self) -> str:
        txt = f"Matte working resolution: {self.work_w}×{self.work_h}"
        if self.downscaled:
            txt += f" (downscaled from {self.src_w}×{self.src_h}, {self.scale * 100:.1f}%)"
        if self.pad_w != self.work_w or self.pad_h != self.work_h:
            txt += f", padded to {self.pad_w}×{self.pad_h}"
        return txt


def _round_even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def compute_geometry(src_w: int, src_h: int, long_side_limit: int | None = None) -> Geometry:
    """Spočítá pracovní rozlišení: zachovat poměr stran, vejít se do 1920×1088
    (a volitelně do ``long_side_limit``), pak padding na násobek 32.

    Orientace na výšku (9:16) se vejde do 1088×1920? NE – model má limit 1920×1088
    v ploše; u svislého videa proto platí šířka ≤ 1088 a výška ≤ 1920.
    """
    if src_w <= 0 or src_h <= 0:
        raise PlanError("Invalid source dimensions.")
    landscape = src_w >= src_h
    max_w, max_h = (MODEL_MAX_W, MODEL_MAX_H) if landscape else (MODEL_MAX_H, MODEL_MAX_W)
    scale = min(1.0, max_w / src_w, max_h / src_h)
    if long_side_limit:
        scale = min(scale, long_side_limit / max(src_w, src_h))
    if scale >= 0.9999:
        work_w, work_h, scale = src_w, src_h, 1.0
    else:
        work_w, work_h = _round_even(src_w * scale), _round_even(src_h * scale)
        work_w, work_h = min(work_w, max_w), min(work_h, max_h)
    pad_w = math.ceil(work_w / DIM_MULTIPLE) * DIM_MULTIPLE
    pad_h = math.ceil(work_h / DIM_MULTIPLE) * DIM_MULTIPLE
    if pad_w > max_w or pad_h > max_h:  # nemělo by nastat (max je násobek 32)
        raise PlanError(f"Canvas {pad_w}×{pad_h} exceeds the model limit.")
    pad_left = (pad_w - work_w) // 2
    pad_top = (pad_h - work_h) // 2
    return Geometry(src_w, src_h, work_w, work_h, pad_w, pad_h, pad_left, pad_top,
                    cli_w=pad_w * 2, cli_h=pad_h * 2, scale=work_w / src_w)


def pad_frame(frame: np.ndarray, geo: Geometry, mode: str = "edge") -> np.ndarray:
    """Doplní snímek (work_w×work_h) na plátno pad_w×pad_h."""
    if frame.shape[1] != geo.work_w or frame.shape[0] != geo.work_h:
        raise PlanError(f"Frame {frame.shape[1]}×{frame.shape[0]} does not match {geo.work_w}×{geo.work_h}")
    if geo.pad_w == geo.work_w and geo.pad_h == geo.work_h:
        return frame
    widths = ((geo.pad_top, geo.pad_bottom), (geo.pad_left, geo.pad_right)) + ((0, 0),) * (frame.ndim - 2)
    if mode == "black":
        return np.pad(frame, widths, mode="constant")
    return np.pad(frame, widths, mode="edge")


def crop_pad(matte: np.ndarray, geo: Geometry) -> np.ndarray:
    """Ořízne padding z výstupní masky (pad_h×pad_w → work_h×work_w)."""
    if matte.shape[0] != geo.pad_h or matte.shape[1] != geo.pad_w:
        raise PlanError(f"Matte {matte.shape[1]}×{matte.shape[0]} does not match canvas {geo.pad_w}×{geo.pad_h}")
    return matte[geo.pad_top:geo.pad_top + geo.work_h, geo.pad_left:geo.pad_left + geo.work_w]


# ----------------------------------------------------------------------------
# Čas
# ----------------------------------------------------------------------------
@dataclass
class Segment:
    index: int
    start: int  # první snímek (relativně k IN)
    length: int  # skutečné snímky zdroje v segmentu
    model_frames: int  # 8n+1 předané modelu (>= length; zbytek = opakovaný poslední snímek)
    ramp_in: int = 0  # délka prolnutí na začátku (překryv s předchozím)
    ramp_out: int = 0

    @property
    def end(self) -> int:
        return self.start + self.length

    def to_dict(self) -> dict:
        return asdict(self)


def plan_segments(total: int, seg_len: int = RECOMMENDED_MAX_FRAMES, overlap: int = 8) -> list[Segment]:
    """Rozdělí ``total`` snímků na segmenty délky 8n+1 s překryvem.

    * žádné vymyšlené snímky uprostřed – segmenty jsou rovnoměrně rozložené tak,
      aby poslední končil přesně na konci (překryv >= ``overlap``),
    * kratší klip než ``seg_len``: jeden segment, model dostane ceil(8n+1) snímků
      a chybějící se doplní opakováním posledního (výsledek se pak ořízne).
    """
    if total < 1:
        raise PlanError("Empty frame range.")
    if not is_valid_frame_count(seg_len):
        raise PlanError(f"Segment length {seg_len} is not of the form 8n+1.")
    if seg_len > HARD_MAX_FRAMES:
        raise PlanError(f"Segment length {seg_len} exceeds the model limit of {HARD_MAX_FRAMES} frames.")
    overlap = max(0, min(int(overlap), seg_len // 2))

    if total <= seg_len:
        return [Segment(0, 0, total, ceil_8n1(total))]

    step = seg_len - overlap
    n = math.ceil((total - overlap) / step)
    n = max(n, 2)
    starts = [round(i * (total - seg_len) / (n - 1)) for i in range(n)]
    segs = [Segment(i, s, seg_len, seg_len) for i, s in enumerate(starts)]
    for a, b in zip(segs, segs[1:]):
        ov = a.end - b.start
        if ov < 0:
            raise PlanError("Internal plan error: gap between segments.")
        a.ramp_out = ov
        b.ramp_in = ov
    return segs


def segment_weights(seg: Segment, is_first: bool, is_last: bool) -> np.ndarray:
    """Váhy snímků segmentu pro prolnutí (lineární rampa přes celý překryv)."""
    w = np.ones(seg.length, dtype=np.float32)
    idx = np.arange(seg.length, dtype=np.float32)
    if not is_first and seg.ramp_in > 0:
        w *= np.clip((idx + 0.5) / seg.ramp_in, 0.0, 1.0)
    if not is_last and seg.ramp_out > 0:
        w *= np.clip((seg.length - idx - 0.5) / seg.ramp_out, 0.0, 1.0)
    return w


def blend_plan(segs: list[Segment], total: int) -> list[list[tuple[int, int, float]]]:
    """Pro každý snímek seznam (segment_index, lokální_index, normalizovaná_váha)."""
    acc: list[list[tuple[int, int, float]]] = [[] for _ in range(total)]
    for k, seg in enumerate(segs):
        w = segment_weights(seg, k == 0, k == len(segs) - 1)
        for i in range(seg.length):
            f = seg.start + i
            if f < total and w[i] > 0:
                acc[f].append((k, i, float(w[i])))
    for f, items in enumerate(acc):
        s = sum(x[2] for x in items)
        if s <= 0:
            raise PlanError(f"Frame {f} is not covered by any segment.")
        acc[f] = [(k, i, wt / s) for k, i, wt in items]
    return acc
