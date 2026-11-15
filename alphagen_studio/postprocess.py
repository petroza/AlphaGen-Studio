"""Postprocessing alfa masky – skutečné obrazové operace (ne parametry modelu).

AlphaGen sám žádné parametry hran nemá; vše zde běží až nad hotovou maskou:
levels, erode/dilate, feather, edge refinement (guided filter s originálním RGB),
časové vyhlazení (anti-flicker), inverze a despill/dekontaminace barvy okrajů.

Maska = float32 ``(H, W)`` v rozsahu 0..1 (1 = neprůhledné popředí).
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class MatteSettings:
    # Matte levels
    black_point: float = 0.0  # 0..1 – vše pod = průhledné
    white_point: float = 1.0  # 0..1 – vše nad = plné
    gamma: float = 1.0  # >1 = měkčí poloprůhlednost, <1 = tvrdší
    # Morfologie (px ve výstupním rozlišení, záporné = erode)
    erode_dilate: float = 0.0
    feather: float = 0.0  # sigma gaussovského rozostření v px
    # Edge refinement (guided filter s originálním RGB)
    edge_refine: bool = False
    edge_radius: int = 4
    edge_strength: float = 0.6  # 0..1 míra přimíchání v okrajovém pásmu
    # Časové vyhlazení (anti-flicker) jen v okrajovém pásmu
    temporal: float = 0.0  # 0..1
    # Čištění drobných ostrůvků
    despeckle: int = 0  # min. plocha v px (0 = vypnuto)
    invert: bool = False
    # Barva (jen u RGBA výstupů)
    despill: str = "off"  # off | edge | green | blue
    despill_strength: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MatteSettings:
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})

    @property
    def needs_guide(self) -> bool:
        return self.edge_refine or self.despill != "off"


# ----------------------------------------------------------------------------
# Základní operace
# ----------------------------------------------------------------------------
def to_float_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    if rgb.dtype == np.uint16:
        return rgb.astype(np.float32) / 65535.0
    return rgb.astype(np.float32)


def apply_levels(a: np.ndarray, black: float, white: float, gamma: float) -> np.ndarray:
    black = float(np.clip(black, 0.0, 0.999))
    white = float(np.clip(white, black + 1e-3, 1.0))
    if black == 0.0 and white == 1.0 and abs(gamma - 1.0) < 1e-6:
        return a
    out = np.clip((a - black) / (white - black), 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        out = np.power(out, 1.0 / max(0.05, float(gamma)))
    return out.astype(np.float32)


def erode_dilate(a: np.ndarray, px: float) -> np.ndarray:
    """Šedotónová morfologie elipsou; podporuje i zlomkové px (lineární mix)."""
    if abs(px) < 0.05:
        return a
    r = abs(px)
    r_int = int(np.floor(r))
    frac = r - r_int
    op = cv2.dilate if px > 0 else cv2.erode

    def morph(radius: int) -> np.ndarray:
        if radius <= 0:
            return a
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return op(a, k)

    base = morph(r_int)
    if frac > 0.05:
        nxt = morph(r_int + 1)
        base = base * (1.0 - frac) + nxt * frac
    return base.astype(np.float32)


def feather(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return a
    return cv2.GaussianBlur(a, (0, 0), float(sigma)).astype(np.float32)


def despeckle(a: np.ndarray, min_area: int) -> np.ndarray:
    """Odstraní malé izolované ostrůvky popředí i díry v popředí."""
    if min_area <= 0:
        return a
    out = a.copy()
    hard = (a >= 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            out[labels == i] = 0.0
    holes = (a < 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches = x == 0 or y == 0 or x + w >= a.shape[1] or y + h >= a.shape[0]
        if area < min_area and not touches:
            out[labels == i] = 1.0
    return out


def edge_band(a: np.ndarray, radius: int) -> np.ndarray:
    """Pásmo kolem hrany (poloprůhledné pixely + okolí 50% kontury) jako float maska 0..1."""
    radius = max(1, int(radius))
    soft = ((a > 0.004) & (a < 0.996)).astype(np.uint8)
    hard = (a >= 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    band = (cv2.dilate(hard, k) - cv2.erode(hard, k)) > 0
    band = np.maximum(band.astype(np.uint8), soft)
    return cv2.GaussianBlur(band.astype(np.float32), (0, 0), max(0.8, radius / 2.0))


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Barevný guided filter (He et al.) – hrany masky se přimknou k hranám RGB."""
    r = max(1, int(radius))
    ksize = (2 * r + 1, 2 * r + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)

    I = guide.astype(np.float32)  # noqa: N806
    p = src.astype(np.float32)
    mean_I = box(I)  # noqa: N806
    mean_p = box(p)
    mean_Ip = box(I * p[..., None])  # noqa: N806
    cov_Ip = mean_Ip - mean_I * mean_p[..., None]  # noqa: N806
    # kovarianční matice 3×3 pro každý pixel
    var = np.empty(I.shape[:2] + (3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = box(I[..., i] * I[..., j]) - mean_I[..., i] * mean_I[..., j]
            var[..., i, j] = v
            var[..., j, i] = v
    var += np.eye(3, dtype=np.float32) * eps
    a = np.linalg.solve(var, cov_Ip[..., None])[..., 0]
    b = mean_p - np.einsum("hwc,hwc->hw", a, mean_I)
    mean_a = box(a)
    mean_b = box(b)
    return (np.einsum("hwc,hwc->hw", mean_a, I) + mean_b).astype(np.float32)


def refine_edges(a: np.ndarray, rgb: np.ndarray, radius: int, strength: float) -> np.ndarray:
    """Guided-filter refinement jen v okrajovém pásmu (vnitřek a pozadí zůstávají)."""
    if strength <= 0:
        return a
    guide = to_float_rgb(rgb)
    if guide.shape[:2] != a.shape[:2]:
        guide = cv2.resize(guide, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    # Pro rychlost počítáme v ořezu kolem okrajů
    band = edge_band(a, radius * 2)
    ys, xs = np.nonzero(band > 0.01)
    if len(xs) == 0:
        return a
    pad = radius * 3 + 2
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + pad + 1)
    sub = a[y0:y1, x0:x1]
    gf = np.clip(guided_filter(guide[y0:y1, x0:x1], sub, radius, 1e-4), 0.0, 1.0)
    w = np.clip(band[y0:y1, x0:x1] * float(strength), 0.0, 1.0)
    out = a.copy()
    out[y0:y1, x0:x1] = sub * (1.0 - w) + gf * w
    return out


def upscale_matte(a: np.ndarray, width: int, height: int) -> np.ndarray:
    if a.shape[1] == width and a.shape[0] == height:
        return a
    interp = cv2.INTER_CUBIC if width > a.shape[1] else cv2.INTER_AREA
    return np.clip(cv2.resize(a, (width, height), interpolation=interp), 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Barva okrajů
# ----------------------------------------------------------------------------
def decontaminate_edges(rgb: np.ndarray, a: np.ndarray, strength: float = 1.0, radius: int = 12) -> np.ndarray:
    """Odhad barvy popředí v poloprůhledných pixelech: F = (I - (1-α)·B) / α,
    kde B je lokální barva pozadí (normalizovaná konvoluce přes průhledné pixely).
    Mění pouze pixely s 0 < α < 1; plné popředí zůstává bit-přesně původní.
    Vrací float32 RGB 0..1."""
    I = to_float_rgb(rgb)  # noqa: N806
    soft = (a > 0.002) & (a < 0.998)
    if not soft.any():
        return I
    bg_w = (1.0 - a).astype(np.float32) ** 2
    num = cv2.GaussianBlur(I * bg_w[..., None], (0, 0), radius)
    den = cv2.GaussianBlur(bg_w, (0, 0), radius)[..., None]
    B = num / np.maximum(den, 1e-4)  # noqa: N806
    alpha = np.maximum(a, 0.05)[..., None]
    F = np.clip((I - (1.0 - a[..., None]) * B) / alpha, 0.0, 1.0)  # noqa: N806
    mix = (soft.astype(np.float32) * float(np.clip(strength, 0, 1)))[..., None]
    return I * (1.0 - mix) + F * mix


def chroma_despill(rgb: np.ndarray, a: np.ndarray, channel: str, strength: float = 1.0) -> np.ndarray:
    """Klasický despill (zelená/modrá) – jen kde je maska poloprůhledná nebo u okraje."""
    I = to_float_rgb(rgb).copy()  # noqa: N806
    r, g, b = I[..., 0], I[..., 1], I[..., 2]
    if channel == "green":
        limit = np.maximum(r, b)
        spill = np.clip(g - limit, 0, None)
        ch = 1
    else:
        limit = np.maximum(r, g)
        spill = np.clip(b - limit, 0, None)
        ch = 2
    zone = edge_band(a, 6) * float(np.clip(strength, 0, 1))
    I[..., ch] = I[..., ch] - spill * zone
    return np.clip(I, 0.0, 1.0)


def apply_color(rgb: np.ndarray, a: np.ndarray, s: MatteSettings) -> np.ndarray | None:
    """Vrátí upravené RGB (float 0..1) nebo None, pokud se barva nemění."""
    if s.despill == "edge":
        return decontaminate_edges(rgb, a, s.despill_strength)
    if s.despill in ("green", "blue"):
        return chroma_despill(rgb, a, s.despill, s.despill_strength)
    return None


# ----------------------------------------------------------------------------
# Celý řetězec pro jeden snímek + časové vyhlazení
# ----------------------------------------------------------------------------
def process_frame(a: np.ndarray, rgb: np.ndarray | None, s: MatteSettings) -> np.ndarray:
    """Prostorový postprocess jednoho snímku (pořadí jako v compositingu)."""
    out = np.clip(a.astype(np.float32), 0.0, 1.0)
    out = apply_levels(out, s.black_point, s.white_point, s.gamma)
    if s.despeckle > 0:
        out = despeckle(out, int(s.despeckle))
    if s.edge_refine and rgb is not None:
        out = refine_edges(out, rgb, int(s.edge_radius), float(s.edge_strength))
    out = erode_dilate(out, float(s.erode_dilate))
    out = feather(out, float(s.feather))
    if s.invert:
        out = 1.0 - out
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class TemporalSmoother:
    """Anti-flicker: vážený medián 3 snímků jen v okrajovém pásmu.

    Streamovací – výstup je zpožděný o 1 snímek (``push`` vrací předchozí snímek,
    ``flush`` vrátí poslední). Pevné plochy (0/1) se nemění, takže se nerozmazává
    pohyb celé postavy.
    """

    def __init__(self, amount: float) -> None:
        self.amount = float(np.clip(amount, 0.0, 1.0))
        self.buf: deque[np.ndarray] = deque(maxlen=3)
        self._emitted = 0
        self._pushed = 0

    def _smooth(self, prev: np.ndarray | None, cur: np.ndarray, nxt: np.ndarray | None) -> np.ndarray:
        if self.amount <= 0 or prev is None or nxt is None:
            return cur
        # přesný medián ze 3 hodnot bez třídění: max(min(a,b), min(max(a,b), c)) – ~10× rychlejší než np.median
        med = np.maximum(np.minimum(prev, cur), np.minimum(np.maximum(prev, cur), nxt))
        band = edge_band(cur, 3)
        w = np.clip(band * self.amount, 0.0, 1.0)
        return (cur * (1.0 - w) + med * w).astype(np.float32)

    def push(self, a: np.ndarray) -> list[np.ndarray]:
        self.buf.append(a)
        self._pushed += 1
        out = []
        if self._pushed == 1:
            return out
        if self._pushed == 2:
            out.append(self._smooth(None, self.buf[0], self.buf[1]))
        else:
            out.append(self._smooth(self.buf[0], self.buf[1], self.buf[2]))
        self._emitted += 1
        return out

    def flush(self) -> list[np.ndarray]:
        if self._pushed == 0:
            return []
        self._emitted += 1
        return [self.buf[-1]]
