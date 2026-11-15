"""Porovná matte z AlphaGen Studia s referenčním výstupem Lightricks (ukázky *_sbs.mp4 z model card).

Referenční alfa se dopočítá z oficiálního composite přes zelenou:  C = a·F + (1-a)·G  →  a = (C-G)·(F-G) / |F-G|²
(F = vstupní RGB, G = barva zeleného pozadí). Výsledek: MAE alfy, IoU tvrdé masky (α>0.5) a obrázek pro kontrolu.

  python tests/compare_reference.py <input.mp4> <ref_green.mp4> <matte_png_dir> <out.png>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def frames(path: str, w: int, h: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32) / 255


def main() -> None:
    inp, ref, matte_dir, out = sys.argv[1:5]
    files = sorted(Path(matte_dir).glob("*.png"))
    m0 = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    h, w = m0.shape
    F = frames(inp, w, h)  # noqa: N806
    C = frames(ref, w, h)  # noqa: N806
    n = min(len(F), len(C), len(files))
    G = np.median(C[:, :40, :40].reshape(-1, 3), axis=0)  # noqa: N806 - barva zelené

    def ref_alpha(i: int, j: int) -> np.ndarray:
        """Alfa reference ze snímku composite ``j`` pro vstupní snímek ``i``."""
        d = F[i] - G
        den = (d * d).sum(-1)
        a = np.clip(((C[j] - G) * d).sum(-1) / np.maximum(den, 1e-3), 0, 1)
        a[den < 0.02] = np.nan
        return a

    # Ukázková videa Lightricks mohou mít spodní (výsledkovou) půlku časově posunutou –
    # posun reference se určí podle nejlepší shody hran reference se VSTUPEM (ne s naší maskou).
    def grad(x: np.ndarray) -> np.ndarray:
        x = cv2.GaussianBlur(np.nan_to_num(x).astype(np.float32), (0, 0), 2)
        return cv2.magnitude(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / max(1e-9, np.sqrt((a * a).sum() * (b * b).sum())))

    probe = list(range(4, n - 4, max(1, n // 20)))
    scores = {off: np.mean([corr(grad(ref_alpha(i, i + off)), grad(F[i].mean(-1))) for i in probe])
              for off in (-2, -1, 0, 1, 2)}
    best = max(scores, key=scores.get)
    # Korekce jen při jednoznačném posunu (u pomalých záběrů jsou hrany sousedních snímků téměř stejné)
    ref_off = best if scores[best] - scores[0] > 0.035 else 0
    print(f"posun reference Lightricks proti vstupu: {ref_off:+d} snímků "
          f"(shoda hran: {', '.join(f'{k:+d}={v:.3f}' for k, v in scores.items())})")
    maes, ious, rows = [], [], []
    for i in range(max(0, -ref_off), min(n, len(C) - ref_off)):
        a_ref = ref_alpha(i, i + ref_off)
        a = cv2.imread(str(files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
        valid = ~np.isnan(a_ref)
        maes.append(float(np.abs(a - a_ref)[valid].mean()))
        r, o = a_ref[valid] > 0.5, a[valid] > 0.5
        ious.append(float((r & o).sum() / max(1, (r | o).sum())))
        if i in (n // 4, n // 2, 3 * n // 4):
            ar = np.nan_to_num(a_ref)
            row = np.concatenate([F[i], np.repeat(ar[..., None], 3, 2), np.repeat(a[..., None], 3, 2),
                                  np.repeat(np.abs(a - ar)[..., None], 3, 2)], axis=1)
            rows.append(cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2)))
    cv2.imwrite(out, (np.concatenate(rows, 0)[..., ::-1] * 255).astype(np.uint8))
    print(f"snímků {n}: MAE alfy {np.mean(maes):.4f} (max {np.max(maes):.4f}), "
          f"IoU {np.mean(ious):.4f} (min {np.min(ious):.4f})")
    print("obrázek: vstup | reference Lightricks | AlphaGen Studio | rozdíl ->", out)


if __name__ == "__main__":
    main()
