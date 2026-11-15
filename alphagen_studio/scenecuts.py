"""Detekce střihů (hranic záběrů) v sestříhaném videu.

AlphaGen vidí celý segment najednou – když segment přeskočí přes střih, model míchá dva různé
záběry a maska kolem střihu se rozpadne. Úseky se proto plánují zvlášť pro každý záběr.

Metoda (rychlá, bez GPU): snímky zmenšené na 128 px, pro každý přechod
* rozdíl obrazu (průměrný absolutní rozdíl RGB),
* rozdíl barevných histogramů (HSV 16×4 binů, L1/2 ∈ [0, 1]).
Střih = oba skoky výrazné a zároveň mnohem větší než okolní pohyb (rychlá kamera střih není).
Záblesk (snímek t se liší, ale t−1 a t+1 jsou stejné) se ignoruje. Záběry kratší než
``MIN_SHOT`` se slijí se sousedním.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .video_io import FrameReader, VideoInfo

SMALL_W = 128
MIN_SHOT = 8  # snímků
HIST_MIN = 0.30  # minimální rozdíl histogramů na střihu
DIFF_MIN = 0.10  # minimální rozdíl obrazu (0–1)
LOCAL_RATIO = 3.0  # skok musí být ~3× větší než medián okolí


def _features(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = rgb.astype(np.float32) / 255.0  # RGB – střih se stejným jasem, ale jinou barvou
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256]).ravel()
    return gray, h / max(1.0, float(h.sum()))


def frame_scores(frames: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Pro přechody i-1 → i (i ≥ 1): (rozdíl obrazu, rozdíl histogramů). Index 0 = 0."""
    n = len(frames)
    d = np.zeros(n, np.float32)
    h = np.zeros(n, np.float32)
    for i in range(1, n):
        d[i] = float(np.mean(np.abs(frames[i][0] - frames[i - 1][0])))
        h[i] = 0.5 * float(np.abs(frames[i][1] - frames[i - 1][1]).sum())
    return d, h


def find_cuts(frames: list[tuple[np.ndarray, np.ndarray]]) -> list[int]:
    """Indexy snímků, kterými začíná nový záběr (0 < i < n)."""
    n = len(frames)
    if n < 3:
        return []
    d, h = frame_scores(frames)
    cuts: list[int] = []
    for i in range(1, n):
        if d[i] < DIFF_MIN or h[i] < HIST_MIN:
            continue
        lo, hi = max(1, i - 8), min(n, i + 9)
        around = np.concatenate([d[lo:i], d[i + 1:hi]])
        local = float(np.median(around)) if around.size else 0.0
        if d[i] < LOCAL_RATIO * local + 0.03:
            continue  # rychlý pohyb / švenk – celý úsek se mění
        if i + 1 < n:  # záblesk: snímek i vybočuje, ale i-1 a i+1 jsou si podobné
            dd = float(np.mean(np.abs(frames[i + 1][0] - frames[i - 1][0])))
            if dd < 0.35 * d[i]:
                continue
        # návrat po záblesku: snímek i je zase podobný tomu před 2–3 snímky
        if any(i - k >= 0 and float(np.mean(np.abs(frames[i][0] - frames[i - k][0]))) < 0.35 * d[i]
               for k in (2, 3)):
            continue
        cuts.append(i)
    # sloučit příliš krátké záběry (záblesky, prolínačky) – nech střih s větším skokem
    merged: list[int] = []
    for c in cuts:
        if merged and c - merged[-1] < MIN_SHOT:
            if d[c] > d[merged[-1]]:
                merged[-1] = c
            continue
        merged.append(c)
    return [c for c in merged if 0 < c < n]


def detect(info: VideoInfo, ffmpeg: str, start: int, count: int, cache_dir: str | Path | None = None,
           progress=None, cancel=None) -> list[int]:
    """Střihy v rozsahu [start, start+count) – vrací indexy relativně k ``start``. Výsledek se cachuje."""
    hit = cached(info, start, count, cache_dir)
    if hit is not None:
        return hit
    cache_file = _cache_file(info, start, count, cache_dir)
    w = SMALL_W
    hh = max(2, int(round(info.height * w / max(1, info.width) / 2)) * 2)
    feats = []
    reader = FrameReader(info, ffmpeg, start, count, high_bit=False, size=(w, hh))
    for i, rgb in enumerate(reader):
        if cancel is not None and cancel.is_set():
            reader.close()
            raise InterruptedError
        feats.append(_features(np.ascontiguousarray(rgb)))
        if progress and (i % 50 == 0):
            progress(i, count)
    cuts = find_cuts(feats)
    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"cuts": cuts, "frames": len(feats)}), encoding="utf-8")
        except OSError:
            pass
    return cuts


def _cache_file(info: VideoInfo, start: int, count: int, cache_dir: str | Path | None) -> Path | None:
    if not cache_dir:
        return None
    src = Path(info.seq_pattern).parent if info.is_sequence else Path(info.path)
    try:
        st = src.stat()
    except OSError:
        return None
    sig = f"{src}|{st.st_mtime}|{st.st_size}|{start}|{count}|v1"
    return Path(cache_dir) / f"cuts_{hashlib.sha1(sig.encode()).hexdigest()[:16]}.json"


def cached(info: VideoInfo, start: int, count: int, cache_dir: str | Path | None) -> list[int] | None:
    """Jen z cache (bez dekódování) – None = ještě nezjištěno."""
    f = _cache_file(info, start, count, cache_dir)
    if f is None:
        return None
    try:
        return [int(x) for x in json.loads(f.read_text(encoding="utf-8"))["cuts"]]
    except (OSError, ValueError, KeyError):
        return None
