"""Rychlostní profily (Turbo / Fast / Normal / High quality) – sdílené GUI, CLI a dávkovou frontou.

Profil určuje rozlišení výpočtu masky, překryv úseků a výchozí úpravy; délka úseku se bere
z ověřeného limitu grafické karty (gpu.recommend).
"""

from __future__ import annotations

from dataclasses import dataclass

from .gpu import GpuInfo, recommend
from .postprocess import MatteSettings
from .segmentation import compute_geometry


@dataclass(frozen=True)
class Profile:
    key: str
    name: str
    long_side: int  # 0 = plné rozlišení (do limitu modelu)
    overlap: int
    temporal: float
    description: str


PROFILES = [
    Profile("turbo", "Turbo", 768, 0, 0.3,
            "Matte at 768 px. Fastest – for previews and rough keys."),
    Profile("fast", "Fast", 1280, 8, 0.3, "Recommended. Matte up to 1280 px, smooth transitions."),
    Profile("normal", "Normal", 0, 8, 0.3, "Full resolution (up to Full HD). Finest hair and motion blur."),
    Profile("high", "High quality", 0, 16, 0.5,
            "Full resolution, longer segment blending and stronger edge anti-flicker."),
]
PROFILE_BY_KEY = {p.key: p for p in PROFILES}


def apply_profile(js, key: str, src_w: int, src_h: int, gpu: GpuInfo | None, ram_total_gb: float):
    """Nastaví JobSettings podle profilu, rozměrů videa a karty (stejně jako režim Automatic v GUI)."""
    p = PROFILE_BY_KEY.get(key, PROFILE_BY_KEY["normal"])
    plan = recommend(gpu, ram_total_gb)
    cap = p.long_side
    if plan.recommended_long_side and plan.recommended_long_side < 1920:
        cap = min(cap or plan.recommended_long_side, plan.recommended_long_side)
    js.long_side = cap
    geo = compute_geometry(src_w, src_h, cap or None)
    js.segment_frames = min(121, plan.segment_for(max(geo.work_w, geo.work_h)) or 121)
    js.overlap = p.overlap
    js.quantization = js.offload = js.vae = "auto"
    # Zpřesnění hrany podle originálu přiblížilo Turbo/Fast k Normal jen o ~1 % (tests/tune_edge_refine.py)
    # a stojí ~0,2 s/snímek v exportu → v automatice vypnuto, v režimu Expert dostupné.
    js.matte = MatteSettings(temporal=p.temporal, edge_refine=False, edge_radius=3, edge_strength=0.5)
    return js
