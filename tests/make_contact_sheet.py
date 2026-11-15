"""Náhledový arch z reálných výsledků: originál | alpha matte | composite na šachovnici | composite na barvě.

  python tests/make_contact_sheet.py <vstup.mp4> <matte_png_dir> <out.png> [snímky,odd,ělené,čárkou]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    inp, mdir, out = sys.argv[1:4]
    files = sorted(Path(mdir).glob("*.png"))
    idxs = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [len(files) // 4, len(files) * 3 // 4]
    m0 = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    h, w = m0.shape
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", inp, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         capture_output=True, check=True).stdout
    F = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32) / 255  # noqa: N806
    yy, xx = np.mgrid[0:h, 0:w]
    s = max(16, w // 48)
    checker = (0.4 + 0.2 * (((yy // s) + (xx // s)) % 2)).astype(np.float32)[..., None].repeat(3, 2)
    color = np.broadcast_to(np.array([0.05, 0.22, 0.55], np.float32), (h, w, 3))
    rows = []
    for i in idxs:
        a = cv2.imread(str(files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32)[..., None] / 65535
        row = np.concatenate([F[i], np.repeat(a, 3, 2), F[i] * a + checker * (1 - a), F[i] * a + color * (1 - a)], 1)
        scale = 1600 / row.shape[1]
        rows.append(cv2.resize(row, (1600, int(row.shape[0] * scale)), interpolation=cv2.INTER_AREA))
    sheet = np.concatenate(rows, 0)
    cv2.imwrite(out, (np.clip(sheet, 0, 1)[..., ::-1] * 255).astype(np.uint8))
    print(out)


if __name__ == "__main__":
    main()
