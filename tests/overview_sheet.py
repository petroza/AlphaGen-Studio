"""Přehled výsledků dávkového testu: na každý klip jeden řádek (zdroj | maska | šachovnice | modrá).

  python tests/overview_sheet.py <slozka_vstupu> <vystup_testu> [snimky_na_list]
Čte casy.csv z batch_test_folder.py, bere prostřední snímek úseku. Výstup: prehled_*.jpg
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W = 1600  # šířka řádku


def frame_at(path: Path, idx: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).astype(np.float32) / 255


def main() -> None:
    src_dir, test_dir = Path(sys.argv[1]), Path(sys.argv[2])
    per_sheet = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    rows = []
    for line in (test_dir / "casy.csv").read_text(encoding="utf-8").splitlines():
        name, res, fps, a, n, sec, work, status = (line.split(";") + [""] * 8)[:8]
        if not a:
            continue
        stem = Path(name).stem
        mdir = next((test_dir / stem).glob("*_matte_png"), None)
        if mdir is None:
            continue
        files = sorted(mdir.glob("*.png"))
        k = len(files) // 2
        alpha = cv2.imread(str(files[k]), cv2.IMREAD_UNCHANGED).astype(np.float32)[..., None] / 65535
        src = frame_at(src_dir / name, int(a) + k)
        h, w = alpha.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        s = max(12, w // 60)
        checker = (0.4 + 0.2 * (((yy // s) + (xx // s)) % 2)).astype(np.float32)[..., None].repeat(3, 2)
        blue = np.broadcast_to(np.array([0.55, 0.25, 0.05], np.float32), src.shape)
        row = np.concatenate([src, np.repeat(alpha, 3, 2), src * alpha + checker * (1 - alpha),
                              src * alpha + blue * (1 - alpha)], 1)
        row = cv2.resize(row, (W, int(row.shape[0] * W / row.shape[1])), interpolation=cv2.INTER_AREA)
        row = (np.clip(row, 0, 1) * 255).astype(np.uint8)
        label = f"{stem.replace('GettyImages-', '')}  {res}  {sec}s"
        row[:32, :400] = (row[:32, :400] * 0.25).astype(np.uint8)  # tmavý pruh pod popiskem
        cv2.putText(row, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (232, 184, 43), 1, cv2.LINE_AA)
        rows.append(row)
    for i in range(0, len(rows), per_sheet):
        sheet = np.concatenate(rows[i:i + per_sheet], 0)
        out = test_dir / f"prehled_{i // per_sheet + 1}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(out)


if __name__ == "__main__":
    main()
