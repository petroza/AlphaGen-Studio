"""Export engine: matte a RGBA výstupy s původním RGB.

Formáty:
* ``matte_png``   – šedotónová 16bit PNG sekvence (bit-přesná maska)
* ``matte_mov``   – šedotónová maska ProRes 422 HQ MOV (pro NLE jako track matte)
* ``rgba_png``    – PNG sekvence RGBA (8/16 bit podle zdroje), straight nebo premultiplied
* ``rgba_exr``    – OpenEXR RGBA half float, premultiplied (konvence OpenEXR), ZIP
* ``prores4444``  – ProRes 4444 MOV s alfou (yuva444p10le) + zvuk, ověřený dekódováním
* ``preview``     – H.264 MP4 náhled (originál | composite na šachovnici) + zvuk

Původní RGB se do výstupu kopíruje beze změny (kromě volitelného despillu
a premultiplikace, pokud ji uživatel zvolí). Model RGB nikdy negeneruje.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import json  # noqa: E402
import subprocess  # noqa: E402
from concurrent.futures import Future, ThreadPoolExecutor  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .logging_setup import get_logger  # noqa: E402
from .postprocess import to_float_rgb  # noqa: E402
from .video_io import (  # noqa: E402
    CREATE_NO_WINDOW,
    FFmpegWriter,
    VideoError,
    VideoInfo,
    audio_input_args,
    color_tag_args,
    out_color_matrix,
)

log = get_logger("export")

FORMAT_LABELS = {
    "prores4444": "ProRes 4444 MOV with alpha (+ audio)",
    "rgba_png": "PNG sequence RGBA",
    "rgba_exr": "EXR sequence RGBA (half, premultiplied)",
    "matte_png": "Alpha matte – PNG 16-bit (grayscale)",
    "matte_mov": "Alpha matte – ProRes 422 HQ MOV",
    "matte_h264": "Luma matte – H.264 MP4 (After Effects)",
    "preview": "Preview video MP4 (QC)",
}


@dataclass
class ExportSettings:
    formats: list[str] = field(default_factory=lambda: ["matte_h264", "preview"])  # luma pro AE + kontrolní náhled
    alpha_mode: str = "straight"  # straight | premultiplied (PNG, ProRes); EXR vždy premultiplied
    include_audio: bool = True
    exr_linearize: bool = False  # False = hodnoty pixelů beze změny (display-referred)
    preview_bg: str = "checker"  # checker | #RRGGBB
    png_compression: int = 3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExportSettings:
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


# ----------------------------------------------------------------------------
def checkerboard(h: int, w: int, size: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    c = ((yy // size + xx // size) % 2).astype(np.float32)
    v = 0.42 + 0.16 * c
    return np.repeat(v[..., None], 3, axis=2)


def parse_color(hex_color: str) -> np.ndarray:
    s = hex_color.lstrip("#")
    if len(s) != 6:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def composite(rgb: np.ndarray, a: np.ndarray, bg: str | np.ndarray = "checker") -> np.ndarray:
    """out = rgb·α + bg·(1-α); vrací float RGB 0..1."""
    f = to_float_rgb(rgb) if rgb.dtype != np.float32 else rgb
    h, w = a.shape
    if isinstance(bg, np.ndarray) and bg.shape == f.shape:
        back = bg
    elif isinstance(bg, str) and bg.startswith("#"):
        back = np.broadcast_to(parse_color(bg), f.shape)
    else:
        back = checkerboard(h, w)
    return f * a[..., None] + back * (1.0 - a[..., None])


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


# ----------------------------------------------------------------------------
class Exporter:
    def __init__(self, info: VideoInfo, settings: ExportSettings, out_dir: Path, base_name: str,
                 in_frame: int, count: int, ffmpeg: str, ffprobe: str) -> None:
        self.info = info
        self.s = settings
        self.out_dir = Path(out_dir)
        self.base = base_name
        self.in_frame = in_frame
        self.count = count
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.w, self.h = info.width, info.height
        self.hi = info.high_bit_depth
        self.written = 0
        self.outputs: dict[str, str] = {}
        self.writers: dict[str, FFmpegWriter] = {}
        self.pool = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4) - 2)))
        self.pending: list[Future] = []
        self._ref_alpha: dict[int, np.ndarray] = {}
        self._check_frames = sorted({0, count // 2, count - 1})
        self._open()

    # ------------------------------------------------------------------
    def _seq_dir(self, key: str) -> Path:
        d = self.out_dir / f"{self.base}_{key}"
        d.mkdir(parents=True, exist_ok=True)
        self.outputs[key] = str(d)
        return d

    def _frame_number(self, i: int) -> int:
        return (self.info.seq_start if self.info.is_sequence else 0) + self.in_frame + i

    def _open(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        fmts = set(self.s.formats)
        audio = self.s.include_audio and self.info.has_audio
        if "prores4444" in fmts:
            path = self.out_dir / f"{self.base}_prores4444.mov"
            extra = audio_input_args(self.info, self.in_frame, self.count) if audio else None
            maps = ["-map", "0:v:0"] + (["-map", "1:a:0?"] if audio else [])
            args = ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le", "-alpha_bits", "16",
                    "-vendor", "apl0", "-vf", f"scale=out_color_matrix={out_color_matrix(self.info)}:out_range=tv",
                    *color_tag_args(self.info)]
            if audio:
                args += ["-c:a", "pcm_s24le"]
            args += ["-movflags", "+write_colr"]
            self.writers["prores4444"] = FFmpegWriter(self.ffmpeg, path, self.w, self.h, "rgba64le",
                                                      self.info.fps, args, extra, maps)
            self.outputs["prores4444"] = str(path)
        if "matte_mov" in fmts:
            path = self.out_dir / f"{self.base}_matte.mov"
            args = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
                    "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
            self.writers["matte_mov"] = FFmpegWriter(self.ffmpeg, path, self.w, self.h, "gray16le",
                                                     self.info.fps, args)
            self.outputs["matte_mov"] = str(path)
        if "matte_h264" in fmts:
            # Šedotónová maska pro AE (Track Matte → Luma). 8bit ztrátové – na jemné vlasy je přesnější PNG/ProRes.
            path = self.out_dir / f"{self.base}_luma_matte.mp4"
            args = ["-c:v", "libx264", "-crf", "10", "-preset", "medium", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_color_matrix=bt709:out_range=tv",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-color_range", "tv", "-movflags", "+faststart"]
            self.writers["matte_h264"] = FFmpegWriter(self.ffmpeg, path, self.w, self.h, "gray16le",
                                                      self.info.fps, args)
            self.outputs["matte_h264"] = str(path)
        if "preview" in fmts:
            path = self.out_dir / f"{self.base}_preview.mp4"
            pw, ph = self._preview_size()
            extra = audio_input_args(self.info, self.in_frame, self.count) if audio else None
            maps = ["-map", "0:v:0"] + (["-map", "1:a:0?"] if audio else [])
            args = ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
            if audio:
                args += ["-c:a", "aac", "-b:a", "192k"]
            args += ["-movflags", "+faststart"]
            self.writers["preview"] = FFmpegWriter(self.ffmpeg, path, pw * 2, ph, "rgb24", self.info.fps,
                                                   args, extra, maps)
            self.outputs["preview"] = str(path)
        for key in ("matte_png", "rgba_png", "rgba_exr"):
            if key in fmts:
                self._seq_dir(key)

    def _preview_size(self) -> tuple[int, int]:
        scale = min(1.0, 960 / self.w)
        pw = max(2, int(self.w * scale) // 2 * 2)
        ph = max(2, int(self.h * scale) // 2 * 2)
        return pw, ph

    # ------------------------------------------------------------------
    def write(self, rgb: np.ndarray, alpha: np.ndarray, rgb_color: np.ndarray | None = None) -> None:
        """Zapíše jeden snímek. ``rgb`` = originál (uint8/uint16), ``alpha`` float 0..1,
        ``rgb_color`` = volitelně barevně upravené RGB (despill) jako float 0..1."""
        i = self.written
        if alpha.shape != (self.h, self.w):
            raise VideoError(f"Matte {alpha.shape} does not match video {self.h}×{self.w}")
        a16 = np.round(np.clip(alpha, 0, 1) * 65535.0).astype(np.uint16)
        if i in self._check_frames:
            self._ref_alpha[i] = a16
        fmts = self.s.formats

        # Barevná data: originál bez převodu, pokud není despill / premultiplikace
        if rgb_color is not None:
            rgb16 = np.round(np.clip(rgb_color, 0, 1) * 65535.0).astype(np.uint16)
        elif rgb.dtype == np.uint8:
            rgb16 = rgb.astype(np.uint16) * 257  # přesné rozšíření 8→16 bit
        else:
            rgb16 = rgb
        premul = self.s.alpha_mode == "premultiplied"

        def rgba16(premultiply: bool) -> np.ndarray:
            c = rgb16
            if premultiply:
                c = np.round(c.astype(np.float32) * (a16.astype(np.float32)[..., None] / 65535.0)).astype(np.uint16)
            return np.dstack([c, a16])

        if "prores4444" in self.writers:
            self.writers["prores4444"].write(rgba16(premul))
        if "matte_mov" in self.writers:
            self.writers["matte_mov"].write(a16)
        if "matte_h264" in self.writers:
            self.writers["matte_h264"].write(a16)
        if "preview" in self.writers:
            pw, ph = self._preview_size()
            src_small = cv2.resize(to_float_rgb(rgb), (pw, ph), interpolation=cv2.INTER_AREA)
            a_small = cv2.resize(alpha.astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA)
            fg = cv2.resize(to_float_rgb(rgb_color) if rgb_color is not None else to_float_rgb(rgb), (pw, ph),
                            interpolation=cv2.INTER_AREA)
            comp = composite(fg, a_small, self.s.preview_bg)
            frame = np.concatenate([src_small, comp], axis=1)
            self.writers["preview"].write(np.round(np.clip(frame, 0, 1) * 255).astype(np.uint8))

        num = self._frame_number(i)
        if "matte_png" in fmts:
            p = Path(self.outputs["matte_png"]) / f"{self.base}_matte_{num:06d}.png"
            self.pending.append(self.pool.submit(_imwrite, p, a16, [cv2.IMWRITE_PNG_COMPRESSION, self.s.png_compression]))
        if "rgba_png" in fmts:
            p = Path(self.outputs["rgba_png"]) / f"{self.base}_{num:06d}.png"
            img = rgba16(premul)
            if not self.hi and rgb_color is None:  # 8bit zdroj → 8bit PNG (RGB bit-přesné)
                c8 = rgb if not premul else np.round(rgb.astype(np.float32) * alpha[..., None]).astype(np.uint8)
                img = np.dstack([c8, np.round(np.clip(alpha, 0, 1) * 255).astype(np.uint8)])
            bgra = img[..., [2, 1, 0, 3]]
            self.pending.append(self.pool.submit(_imwrite, p, bgra, [cv2.IMWRITE_PNG_COMPRESSION, self.s.png_compression]))
        if "rgba_exr" in fmts:
            p = Path(self.outputs["rgba_exr"]) / f"{self.base}_{num:06d}.exr"
            c = rgb_color if rgb_color is not None else to_float_rgb(rgb)
            if self.s.exr_linearize:
                c = srgb_to_linear(c)
            a = np.clip(alpha, 0, 1).astype(np.float32)
            rgba = np.dstack([c * a[..., None], a]).astype(np.float16)  # premultiplied, half
            self.pending.append(self.pool.submit(_write_exr, p, rgba))
        # Omezení fronty – ať nebobtná RAM
        if len(self.pending) > 32:
            for f in self.pending[:16]:
                f.result()
            self.pending = self.pending[16:]
        self.written += 1

    # ------------------------------------------------------------------
    def close(self) -> dict:
        errors = []
        for f in self.pending:
            try:
                f.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        self.pool.shutdown(wait=True)
        for key, wr in self.writers.items():
            try:
                wr.close()
            except VideoError as exc:
                errors.append(f"{key}: {exc}")
        if errors:
            raise VideoError("Export failed: " + "; ".join(errors[:5]))
        if self.written != self.count:
            raise VideoError(f"Wrote {self.written} frames, expected {self.count}.")
        report = {"outputs": self.outputs, "frames": self.written, "checks": {}}
        if "prores4444" in self.outputs:
            report["checks"]["prores4444"] = verify_prores_alpha(
                self.outputs["prores4444"], self.ffmpeg, self.ffprobe, self.count, self._ref_alpha)
        for key in ("matte_png", "rgba_png", "rgba_exr"):
            if key in self.outputs:
                n = len(list(Path(self.outputs[key]).iterdir()))
                report["checks"][key] = {"ok": n >= self.count, "files": n}
        for key in ("matte_mov", "matte_h264", "preview"):
            if key in self.outputs:
                report["checks"][key] = verify_frame_count(self.outputs[key], self.ffprobe, self.count)
        (self.out_dir / f"{self.base}_export_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    def abort(self) -> None:
        for wr in self.writers.values():
            wr.abort()
        self.pool.shutdown(wait=False, cancel_futures=True)


def _imwrite(path: Path, img: np.ndarray, params: list[int]) -> None:
    ok, buf = cv2.imencode(path.suffix, img, params)
    if not ok:
        raise VideoError(f"Cannot write {path.name}")
    buf.tofile(str(path))  # tofile zvládne i diakritiku v cestě (cv2.imwrite ne)


def _write_exr(path: Path, rgba_half: np.ndarray) -> None:
    """RGBA half float, ZIP, scanline – oficiální knihovna OpenEXR (ASWF)."""
    import OpenEXR

    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    tmp = path.with_suffix(".tmp.exr")
    with OpenEXR.File(header, {"RGBA": np.ascontiguousarray(rgba_half)}) as f:
        f.write(str(tmp))
    tmp.replace(path)


# ----------------------------------------------------------------------------
# Ověření
# ----------------------------------------------------------------------------
def verify_frame_count(path: str, ffprobe: str, expected: int) -> dict:
    res = subprocess.run(  # noqa: S603
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=nb_read_frames,pix_fmt,codec_name", "-of", "json", path],
        capture_output=True, creationflags=CREATE_NO_WINDOW, check=False, timeout=600,
    )
    try:
        st = json.loads(res.stdout.decode())["streams"][0]
        n = int(st.get("nb_read_frames", 0))
    except (ValueError, KeyError, IndexError):
        return {"ok": False, "error": res.stderr.decode(errors="replace")[-300:]}
    return {"ok": n == expected, "frames": n, "expected": expected, "pix_fmt": st.get("pix_fmt"),
            "codec": st.get("codec_name")}


def verify_prores_alpha(path: str, ffmpeg: str, ffprobe: str, expected: int,
                        ref_alpha: dict[int, np.ndarray]) -> dict:
    """Dekóduje ProRes 4444 a ověří: kodek, pix_fmt s alfou, počet snímků a že
    alfa kanál skutečně odpovídá zapsané masce (vybrané snímky)."""
    info = verify_frame_count(path, ffprobe, expected)
    result = {**info, "has_alpha_pix_fmt": str(info.get("pix_fmt", "")).startswith("yuva"),
              "alpha_match": None, "max_alpha_error": None}
    if not result["has_alpha_pix_fmt"]:
        result["ok"] = False
        return result
    max_err = 0.0
    for idx, ref in ref_alpha.items():
        h, w = ref.shape
        res = subprocess.run(  # noqa: S603
            [ffmpeg, "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{idx}),alphaextract,format=gray16le",
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray16le", "pipe:1"],
            capture_output=True, creationflags=CREATE_NO_WINDOW, check=False, timeout=600,
        )
        if len(res.stdout) != w * h * 2:
            result.update(ok=False, alpha_match=False, error=f"Cannot decode alpha of frame {idx}")
            return result
        dec = np.frombuffer(res.stdout, dtype=np.uint16).reshape(h, w).astype(np.float32) / 65535.0
        max_err = max(max_err, float(np.abs(dec - ref.astype(np.float32) / 65535.0).max()))
    # ProRes 4444 ukládá alfu bezztrátově v 16 bit (alpha_bits 16), drobná odchylka = zaokrouhlení
    result["max_alpha_error"] = round(max_err, 6)
    result["alpha_match"] = max_err <= 2.0 / 1023.0
    result["ok"] = bool(info.get("ok")) and result["alpha_match"]
    return result
