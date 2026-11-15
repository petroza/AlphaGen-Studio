"""Otestuje krátký úsek (49 snímků ze středu) každého videa ve složce v daném profilu.

  python tests/batch_test_folder.py <slozka> <vystup> [profil]
Jeden Engine = model se načte jen jednou. Výsledky: <vystup>/<klip>/..., casy.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphagen_studio.config import AppSettings  # noqa: E402
from alphagen_studio.engine import Engine, JobListener, JobSettings  # noqa: E402
from alphagen_studio.gpu import list_gpus, ram_info  # noqa: E402
from alphagen_studio.profiles import apply_profile  # noqa: E402
from alphagen_studio.video_io import probe  # noqa: E402

EXTS = {".mp4", ".mov", ".mkv", ".mxf", ".m4v"}


class L(JobListener):
    def log(self, level, msg):
        if level in ("ERROR", "WARNING"):
            print(f"   [{level}] {msg}", flush=True)


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    prof = sys.argv[3] if len(sys.argv) > 3 else "fast"
    pattern = sys.argv[4] if len(sys.argv) > 4 else "*"
    out.mkdir(parents=True, exist_ok=True)
    s = AppSettings.load()
    eng = Engine(s)
    gpu = list_gpus()[0]
    eng.gpu_total_gb = gpu.total_gb
    csv = out / "casy.csv"
    files = sorted(p for p in src.glob(pattern) if p.suffix.lower() in EXTS)
    print(f"{len(files)} videí, profil {prof}", flush=True)
    for p in files:
        t0 = time.time()
        try:
            info = probe(p, s.ffprobe_path())
            n = min(49, info.nb_frames)
            a = max(0, info.nb_frames // 2 - n // 2)
            js = apply_profile(JobSettings(in_frame=a, out_frame=a + n, output_dir=str(out / p.stem),
                                           base_name=p.stem), prof, info.width, info.height, gpu, ram_info()[1])
            js.export.formats = ["matte_png", "matte_h264", "preview"]
            rep = eng.run(str(p), js, L(), False, gpu.total_gb)
            ok = all(c.get("ok") for c in rep.get("checks", {}).values())
            line = (f"{p.name};{info.width}x{info.height};{info.fps_float:.2f};{a};{n};{time.time() - t0:.0f};"
                    f"{rep['geometry']['work_w']}x{rep['geometry']['work_h']};{'OK' if ok else 'CHECK FAIL'}")
        except Exception as exc:  # noqa: BLE001 - pokračovat dalším videem
            line = f"{p.name};;;;;{time.time() - t0:.0f};;ERROR {type(exc).__name__}: {str(exc)[:150]}"
        print(line, flush=True)
        with csv.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    eng.shutdown()
    print("HOTOVO", flush=True)


if __name__ == "__main__":
    main()
