"""Srovnání profilů Turbo / Fast / Normal: metriky proti Normal + zvětšené výřezy hran.

  python tests/compare_profiles.py
Výstup: output/srovnani_profilu/tabulka.md, vyrez_<klip>.png
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "srovnani_profilu"
CLIPS = {"vlasy_obloha_1080": ["turbo", "fast", "normal"], "muz_zed_1080": ["turbo", "fast", "normal"],
         "kudrny_720": ["turbo", "normal"]}
FRAME = 24


def mattes(clip: str, prof: str) -> list[np.ndarray]:
    d = next((OUT / f"{clip}_{prof}").glob("*_matte_png"))
    return [cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535 for p in sorted(d.glob("*.png"))]


def source_frame(clip: str, idx: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(ROOT / "tests" / "data" / "cmp" / f"{clip}.mp4"),
                          "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True, check=True).stdout
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).astype(np.float32) / 255


def edge_band(a: np.ndarray) -> np.ndarray:
    soft = ((a > 0.02) & (a < 0.98)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(soft, k) > 0


def main() -> None:
    rows = ["| Klip | Rychlost | Rozlišení masky | Čas úseku 49 sn. | s / snímek | Odhad 1 min videa (25 fps) "
            "| Shoda tvaru s Normal (IoU) | Odchylka alfy | Odchylka na hranách | Jemné detaily vs Normal |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    times = {}
    for line in (OUT / "casy.csv").read_text(encoding="utf-8").splitlines():
        c, p, total, seg, vram, *_ = line.split(";")
        times[(c, p)] = (float(seg), float(vram))
    res_txt = {"turbo": "768 px", "fast": "1280 px", "normal": "plné"}
    for clip, profs in CLIPS.items():
        ref = mattes(clip, "normal")
        h, w = ref[0].shape
        res = {p: mattes(clip, p) for p in profs}
        ref_soft = np.mean([((a > 0.02) & (a < 0.98)).sum() for a in ref])
        for p in profs:
            m = res[p]
            ious, maes, emaes = [], [], []
            for a, r in zip(m, ref):
                ra, rr = a > 0.5, r > 0.5
                ious.append((ra & rr).sum() / max(1, (ra | rr).sum()))
                maes.append(np.abs(a - r).mean())
                band = edge_band(r)
                emaes.append(np.abs(a - r)[band].mean())
            soft = np.mean([((a > 0.02) & (a < 0.98)).sum() for a in m])
            seg, _vram = times[(clip, p)]
            per = seg / 49
            # 1 min při 25 fps = 1500 snímků → úseky po 49 s překryvem 8 (Turbo bez překryvu)
            ov = 0 if p == "turbo" else 8
            n = int(np.ceil((1500 - ov) / (49 - ov)))
            est = (n * seg + 30) / 60
            work = res_txt[p] if not (clip.endswith("720") and p != "turbo") else "plné (720p)"
            ref_mark = p == "normal"
            rows.append(f"| {clip} | {p.capitalize()} | {work} | {seg:.0f} s | {per:.2f} s | ~{est:.0f} min | "
                        + ("reference |" if ref_mark else f"{np.mean(ious):.3f} |")
                        + ("– |" if ref_mark else f"{np.mean(maes):.4f} |")
                        + ("– |" if ref_mark else f"{np.mean(emaes):.3f} |")
                        + ("100 % |" if ref_mark else f"{soft / max(1, ref_soft) * 100:.0f} % |"))
        # výřez: místo s nejvíc jemnými pixely (vlasy) v referenci
        r = ref[FRAME]
        soft = ((r > 0.02) & (r < 0.98)).astype(np.float32)
        cs = min(320, h // 2)
        dens = cv2.boxFilter(soft, -1, (cs, cs))
        y, x = np.unravel_index(np.argmax(dens), dens.shape)
        y0, x0 = int(np.clip(y - cs // 2, 0, h - cs)), int(np.clip(x - cs // 2, 0, w - cs))
        src = source_frame(clip, FRAME)
        tiles = [src[y0:y0 + cs, x0:x0 + cs]]
        bg = np.array([0.55, 0.25, 0.05], np.float32)  # tmavě modrá (BGR) pro composite
        for p in profs:
            a = res[p][FRAME][y0:y0 + cs, x0:x0 + cs][..., None]
            tiles.append(np.repeat(a, 3, 2))
        for p in profs:
            a = res[p][FRAME][y0:y0 + cs, x0:x0 + cs][..., None]
            tiles.append(tiles[0] * a + bg * (1 - a))
        tiles = [cv2.resize(t, (400, 400), interpolation=cv2.INTER_NEAREST) for t in tiles]
        top = np.concatenate([tiles[0]] + tiles[1:1 + len(profs)], 1)
        bot = np.concatenate([np.zeros_like(tiles[0])] + tiles[1 + len(profs):], 1)
        img = (np.concatenate([top, bot], 0) * 255).clip(0, 255).astype(np.uint8)
        labels = ["zdroj"] + [p.capitalize() for p in profs]
        for i, t in enumerate(labels):
            cv2.putText(img, t, (i * 400 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "composite", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 220, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"vyrez_{clip}.png"), img)
    (OUT / "tabulka.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
