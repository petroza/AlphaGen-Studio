"""Video I/O nad FFmpeg: probe, snímkově přesné čtení, zápis přes rawvideo pipe.

Konvence:
* snímky se vrací jako numpy RGB ``(H, W, 3)`` – ``uint8`` pro 8bit zdroje,
  ``uint16`` pro zdroje s vyšší bitovou hloubkou (ProRes 10/12 bit, 16bit PNG),
* rozměry jsou "display" rozměry – rotace z metadat (mobilní video) se aplikuje
  při dekódování (FFmpeg autorotate), výstupy jsou tedy fyzicky narovnané,
* index snímku = pořadí dekódovaného snímku od začátku streamu (CFR).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .logging_setup import get_logger

log = get_logger("video_io")

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mxf", ".webm", ".mts", ".m2ts", ".ts"}
IMAGE_EXTS = {".png", ".tif", ".tiff", ".exr", ".jpg", ".jpeg", ".dpx"}
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class VideoError(RuntimeError):
    """Neplatné nebo nečitelné video."""


# ----------------------------------------------------------------------------
# Info
# ----------------------------------------------------------------------------
@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps_num: int
    fps_den: int
    nb_frames: int
    duration: float
    codec: str
    codec_long: str = ""
    pix_fmt: str = ""
    bit_depth: int = 8
    source_has_alpha: bool = False
    rotation: int = 0
    coded_width: int = 0
    coded_height: int = 0
    color_space: str = ""
    color_transfer: str = ""
    color_primaries: str = ""
    color_range: str = ""
    is_vfr: bool = False
    has_audio: bool = False
    audio_codec: str = ""
    video_start: float = 0.0
    format_start: float = 0.0
    sar: str = "1:1"
    is_sequence: bool = False
    seq_pattern: str = ""
    seq_start: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def fps(self) -> Fraction:
        return Fraction(self.fps_num, self.fps_den)

    @property
    def fps_float(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def high_bit_depth(self) -> bool:
        return self.bit_depth > 8

    def frame_to_time(self, frame: int) -> float:
        return float(Fraction(frame) / self.fps)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> VideoInfo:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def summary_cs(self) -> str:
        """Popis pro GUI (česky)."""
        dur = self.nb_frames / self.fps_float if self.fps_float else self.duration
        lines = [
            f"Resolution: {self.width} × {self.height}" + (f"  (rotation {self.rotation}°)" if self.rotation else ""),
            f"FPS: {self.fps_float:.3f}" + ("  (VFR!)" if self.is_vfr else ""),
            f"Duration: {format_tc(dur)}  ({self.nb_frames} frames)",
            f"Codec: {self.codec}" + (f" / {self.pix_fmt}" if self.pix_fmt else ""),
            f"Bit depth: {self.bit_depth}-bit" + ("  + source alpha" if self.source_has_alpha else ""),
            f"Color: {self.color_space or '?'} / {self.color_transfer or '?'} / {self.color_range or '?'}",
            "Audio: " + (self.audio_codec if self.has_audio else "none"),
        ]
        if self.is_sequence:
            lines.insert(0, f"Sequence: {Path(self.seq_pattern).name} from {self.seq_start}")
        return "\n".join(lines)


def format_tc(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _parse_rate(rate: str | None) -> Fraction | None:
    if not rate or rate in ("0/0", "0"):
        return None
    try:
        fr = Fraction(rate)
    except (ValueError, ZeroDivisionError):
        return None
    return fr if fr > 0 else None


def _bit_depth(pix_fmt: str, bits_raw: str | None) -> int:
    if bits_raw and str(bits_raw).isdigit() and int(bits_raw) > 0:
        return int(bits_raw)
    m = re.search(r"p(\d{2})(le|be)?$", pix_fmt or "")
    if m:
        return int(m.group(1))
    if pix_fmt and ("48" in pix_fmt or "64" in pix_fmt or "16" in pix_fmt):
        return 16
    if pix_fmt and ("f32" in pix_fmt or "f16" in pix_fmt):
        return 16
    return 8


def _run(cmd: list[str], timeout: float | None = 120) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 - příkazy skládáme sami
        cmd, capture_output=True, timeout=timeout, creationflags=CREATE_NO_WINDOW, check=False
    )


# ----------------------------------------------------------------------------
# Sekvence obrázků
# ----------------------------------------------------------------------------
_SEQ_RE = re.compile(r"^(?P<prefix>.*?)(?P<digits>\d+)(?P<ext>\.[A-Za-z0-9]+)$")


def detect_sequence(path: str | Path) -> tuple[str, int, int] | None:
    """Z jednoho souboru (nebo složky) sekvence vrátí (ffmpeg_pattern, start, count)."""
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if not files:
            return None
        p = files[0]
    m = _SEQ_RE.match(p.name)
    if not m or p.suffix.lower() not in IMAGE_EXTS:
        return None
    prefix, digits, ext = m.group("prefix"), m.group("digits"), m.group("ext")
    rx = re.compile(re.escape(prefix) + r"(\d{" + str(len(digits)) + r"})" + re.escape(ext) + "$")
    numbers = sorted(int(mm.group(1)) for f in p.parent.iterdir() if (mm := rx.match(f.name)))
    if not numbers:
        return None
    start = numbers[0]
    count = 0
    for i, n in enumerate(numbers):  # souvislý úsek od prvního snímku
        if n != start + i:
            break
        count += 1
    pattern = str(p.parent / f"{prefix}%0{len(digits)}d{ext}")
    return pattern, start, count


# ----------------------------------------------------------------------------
# Probe
# ----------------------------------------------------------------------------
def probe(path: str | Path, ffprobe: str, *, sequence_fps: float = 25.0) -> VideoInfo:
    path = str(path)
    if not ffprobe:
        raise VideoError("FFprobe not found. Run setup.bat or set the FFmpeg path.")
    p = Path(path)
    if not p.exists():
        raise VideoError(f"File does not exist: {path}")

    seq = None
    if p.is_dir() or p.suffix.lower() in IMAGE_EXTS:
        seq = detect_sequence(p)
        if seq is None:
            raise VideoError("Could not detect an image sequence (expected e.g. frame_0001.png).")

    if seq:
        pattern, start, count = seq
        cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
               "-framerate", str(sequence_fps), "-start_number", str(start), "-i", pattern]
    else:
        cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]
    res = _run(cmd)
    if res.returncode != 0:
        raise VideoError(f"FFprobe failed: {res.stderr.decode(errors='replace').strip()[:500]}")
    data = json.loads(res.stdout.decode("utf-8", errors="replace") or "{}")
    streams = data.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    if not vstreams:
        raise VideoError("File contains no video stream.")
    v = vstreams[0]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]
    fmt = data.get("format", {})

    warnings: list[str] = []
    r_rate = _parse_rate(v.get("r_frame_rate"))
    avg_rate = _parse_rate(v.get("avg_frame_rate"))
    fps = Fraction(sequence_fps).limit_denominator(1001) if seq else (avg_rate or r_rate)
    if fps is None:
        raise VideoError("Could not determine the video FPS.")
    is_vfr = bool(not seq and r_rate and avg_rate and abs(float(r_rate) - float(avg_rate)) > 0.01)
    if is_vfr:
        warnings.append(
            f"Video has variable frame rate (r={float(r_rate):.3f}, avg={float(avg_rate):.3f}). "
            "It will be converted to constant FPS before processing."
        )
        fps = avg_rate or r_rate

    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rotation = int(round(float(sd["rotation"])))
            except (TypeError, ValueError):
                rotation = 0
    if not rotation and v.get("tags", {}).get("rotate"):
        try:
            rotation = -int(v["tags"]["rotate"])
        except ValueError:
            rotation = 0
    rotation = (-rotation) % 360  # displaymatrix -90 = otočit o 90° po směru

    cw, ch = int(v.get("width", 0)), int(v.get("height", 0))
    if cw <= 0 or ch <= 0:
        raise VideoError("Invalid video dimensions.")
    w, h = (ch, cw) if rotation in (90, 270) else (cw, ch)

    pix_fmt = v.get("pix_fmt", "") or ""
    bit_depth = _bit_depth(pix_fmt, v.get("bits_per_raw_sample"))
    source_alpha = any(t in pix_fmt for t in ("yuva", "rgba", "bgra", "argb", "abgr", "gbrap", "ya", "pal8")) or pix_fmt in ("ya8", "ya16le", "ya16be")

    duration = float(v.get("duration") or fmt.get("duration") or 0.0)
    if seq:
        nb = seq[2]
        duration = nb / float(fps)
    else:
        nb = int(v.get("nb_frames") or 0)
        if nb <= 0:
            nb = count_frames(path, ffprobe) or int(round(duration * float(fps)))
    if nb <= 0:
        raise VideoError("Video contains no frames.")

    sar = v.get("sample_aspect_ratio") or "1:1"
    if sar not in ("1:1", "0:1", "N/A"):
        warnings.append(f"Video has non-square pixels (SAR {sar}). The matte matches the stored pixels.")

    info = VideoInfo(
        path=path,
        width=w,
        height=h,
        fps_num=fps.numerator,
        fps_den=fps.denominator,
        nb_frames=nb,
        duration=duration,
        codec=v.get("codec_name", ""),
        codec_long=v.get("codec_long_name", ""),
        pix_fmt=pix_fmt,
        bit_depth=bit_depth,
        source_has_alpha=source_alpha,
        rotation=rotation,
        coded_width=cw,
        coded_height=ch,
        color_space=v.get("color_space", "") or "",
        color_transfer=v.get("color_transfer", "") or "",
        color_primaries=v.get("color_primaries", "") or "",
        color_range=v.get("color_range", "") or "",
        is_vfr=is_vfr,
        has_audio=bool(astreams) and not seq,
        audio_codec=(astreams[0].get("codec_name", "") if astreams else ""),
        video_start=float(v.get("start_time") or 0.0),
        format_start=float(fmt.get("start_time") or 0.0),
        sar=sar,
        is_sequence=bool(seq),
        seq_pattern=seq[0] if seq else "",
        seq_start=seq[1] if seq else 0,
        warnings=warnings,
    )
    if info.color_transfer in ("smpte2084", "arib-std-b67"):
        info.warnings.append("HDR (PQ/HLG) source: the model works on an SDR proxy, output RGB stays original.")
    if w > 1920 or h > 1088:
        info.warnings.append(
            f"Source {w}×{h} exceeds the model limit of 1920×1088 – the matte is computed at a reduced resolution "
            "and upscaled back (original RGB is preserved)."
        )
    return info


def count_frames(path: str, ffprobe: str) -> int:
    """Přesný počet snímků přes počítání paketů (rychlejší než dekódování)."""
    res = _run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
                "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path], timeout=600)
    try:
        return int(res.stdout.decode().strip().split(",")[0])
    except (ValueError, IndexError):
        return 0


# ----------------------------------------------------------------------------
# Dekódování
# ----------------------------------------------------------------------------
def _color_matrix(info: VideoInfo) -> str:
    cs = (info.color_space or "").lower()
    if cs in ("bt709",):
        return "bt709"
    if cs in ("bt470bg", "smpte170m", "bt601", "fcc"):
        return "bt601"
    if cs.startswith("bt2020"):
        return "bt2020"
    if cs == "smpte240m":
        return "smpte240m"
    # Neoznačené video: HD a výš = BT.709, SD = BT.601 (běžná konvence)
    return "bt709" if info.height >= 720 else "bt601"


def _is_rgb_source(pix_fmt: str) -> bool:
    p = pix_fmt.lower()
    return p.startswith(("rgb", "bgr", "gbr", "argb", "abgr", "rgba", "bgra", "gray", "ya")) or p == "pal8"


def rgb_filter(info: VideoInfo, out_pix: str, size: tuple[int, int] | None = None) -> str:
    """Filtr pro převod do RGB se správnou maticí a rozsahem (bez tichého BT.601)."""
    parts = []
    scale_opts = []
    if size:
        scale_opts += [f"w={size[0]}", f"h={size[1]}", "flags=bicubic+accurate_rnd+full_chroma_int"]
    else:
        scale_opts += ["flags=accurate_rnd+full_chroma_int"]
    if not _is_rgb_source(info.pix_fmt) and not info.is_sequence:
        rng = "pc" if info.color_range in ("pc", "jpeg") else "tv"
        if info.pix_fmt.startswith("yuvj"):
            rng = "pc"
        scale_opts += [f"in_color_matrix={_color_matrix(info)}", f"in_range={rng}", "out_range=pc"]
    parts.append("scale=" + ":".join(scale_opts))
    parts.append(f"format={out_pix}")
    return ",".join(parts)


def input_args(info: VideoInfo, start_frame: int = 0) -> list[str]:
    """Vstupní argumenty FFmpeg se snímkově přesným seekem."""
    if info.is_sequence:
        return ["-framerate", f"{info.fps_num}/{info.fps_den}",
                "-start_number", str(info.seq_start + start_frame), "-i", info.seq_pattern]
    args: list[str] = []
    if start_frame > 0:
        # Seek na polovinu snímku před cílovým: první snímek s pts >= t je přesně start_frame.
        # Pozice je relativní k začátku souboru (format start), video může začínat později.
        offset = max(0.0, info.video_start - info.format_start)
        t = offset + (start_frame - 0.5) * info.fps_den / info.fps_num
        args += ["-ss", f"{max(0.0, t):.6f}"]
    args += ["-i", info.path]
    return args


class FrameReader:
    """Iterátor RGB snímků [start, start+count) ze zdroje přes FFmpeg pipe."""

    def __init__(self, info: VideoInfo, ffmpeg: str, start: int = 0, count: int | None = None,
                 high_bit: bool | None = None, size: tuple[int, int] | None = None) -> None:
        if not ffmpeg:
            raise VideoError("FFmpeg not found.")
        self.info = info
        self.start = int(start)
        self.count = info.nb_frames - self.start if count is None else int(count)
        if self.start < 0 or self.count <= 0:
            raise VideoError(f"Invalid frame range: start={start}, count={count}")
        self.high_bit = info.high_bit_depth if high_bit is None else high_bit
        self.w, self.h = size if size else (info.width, info.height)
        out_pix = "rgb48le" if self.high_bit else "rgb24"
        self.dtype = np.uint16 if self.high_bit else np.uint8
        self.frame_bytes = self.w * self.h * 3 * (2 if self.high_bit else 1)
        cmd = [ffmpeg, "-v", "error", "-nostdin", *input_args(info, self.start),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough",
               "-vf", rgb_filter(info, out_pix, size), "-frames:v", str(self.count),
               "-f", "rawvideo", "-pix_fmt", out_pix, "pipe:1"]
        self._stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(  # noqa: S603
            cmd, stdout=subprocess.PIPE, stderr=self._stderr, stdin=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW, bufsize=self.frame_bytes * 2,
        )
        self.read = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        try:
            while self.read < self.count:
                buf = _read_exact(self.proc.stdout, self.frame_bytes)
                if buf is None:
                    break
                self.read += 1
                yield np.frombuffer(buf, dtype=self.dtype).reshape(self.h, self.w, 3)
        finally:
            self.close()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        if self.proc.stdout:
            self.proc.stdout.close()

    def stderr_text(self) -> str:
        try:
            self._stderr.seek(0)
            return self._stderr.read().decode(errors="replace")[-2000:]
        except OSError:
            return ""


def _read_exact(stream, n: int) -> bytes | None:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frames(info: VideoInfo, ffmpeg: str, start: int, count: int, **kw) -> list[np.ndarray]:
    reader = FrameReader(info, ffmpeg, start, count, **kw)
    frames = [f.copy() for f in reader]
    if len(frames) != count:
        raise VideoError(
            f"Read only {len(frames)} of {count} frames from source (from {start}). {reader.stderr_text()}"
        )
    return frames


def read_frame(info: VideoInfo, ffmpeg: str, index: int, size: tuple[int, int] | None = None) -> np.ndarray:
    return read_frames(info, ffmpeg, index, 1, size=size)[0]


# ----------------------------------------------------------------------------
# Zápis
# ----------------------------------------------------------------------------
class FFmpegWriter:
    """Zápis numpy snímků do FFmpeg přes rawvideo stdin."""

    def __init__(self, ffmpeg: str, out_path: str | Path, width: int, height: int, in_pix_fmt: str,
                 fps: Fraction | str, output_args: list[str], extra_inputs: list[str] | None = None,
                 maps: list[str] | None = None) -> None:
        self.out_path = str(out_path)
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        rate = f"{fps.numerator}/{fps.denominator}" if isinstance(fps, Fraction) else str(fps)
        cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", in_pix_fmt,
               "-s", f"{width}x{height}", "-framerate", rate, "-i", "pipe:0"]
        if extra_inputs:
            cmd += extra_inputs
        cmd += maps or ["-map", "0:v:0"]
        cmd += output_args + [self.out_path]
        self.cmd = cmd
        self._stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(  # noqa: S603
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr,
            creationflags=CREATE_NO_WINDOW,
        )
        self.frames = 0
        self._lock = threading.Lock()

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            try:
                self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError) as exc:
                raise VideoError(f"FFmpeg write failed: {self.stderr_text()}") from exc
            self.frames += 1

    def close(self) -> None:
        with self._lock:
            if self.proc.stdin and not self.proc.stdin.closed:
                try:
                    self.proc.stdin.close()
                except OSError:
                    pass
            rc = self.proc.wait()
        if rc != 0:
            raise VideoError(f"FFmpeg exited with error ({rc}): {self.stderr_text()}")

    def abort(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def stderr_text(self) -> str:
        try:
            self._stderr.seek(0)
            return self._stderr.read().decode(errors="replace")[-3000:]
        except OSError:
            return ""


def color_tag_args(info: VideoInfo) -> list[str]:
    """Předá barevné značky zdroje do výstupu (bez nich by NLE mohly zvolit jinou matici)."""
    args = []
    cs = info.color_space if info.color_space and info.color_space != "unknown" else (
        "bt709" if info.height >= 720 else "smpte170m")
    pr = info.color_primaries if info.color_primaries and info.color_primaries != "unknown" else (
        "bt709" if info.height >= 720 else "smpte170m")
    tr = info.color_transfer if info.color_transfer and info.color_transfer != "unknown" else (
        "bt709" if info.height >= 720 else "smpte170m")
    args += ["-colorspace", cs, "-color_primaries", pr, "-color_trc", tr]
    return args


def out_color_matrix(info: VideoInfo) -> str:
    return _color_matrix(info)


def audio_input_args(info: VideoInfo, start_frame: int, count: int) -> list[str]:
    """Vstup zvuku oříznutý přesně na rozsah [start_frame, start_frame+count)."""
    if not info.has_audio:
        return []
    offset = max(0.0, info.video_start - info.format_start)
    t0 = offset + start_frame * info.fps_den / info.fps_num
    dur = count * info.fps_den / info.fps_num
    return ["-ss", f"{t0:.6f}", "-t", f"{dur:.6f}", "-i", info.path]


def normalize_vfr(info: VideoInfo, ffmpeg: str, out_path: Path) -> Path:
    """Převede VFR video na CFR bezztrátový mezisoubor (FFV1 + PCM), aby index = čas."""
    rate = f"{info.fps_num}/{info.fps_den}"
    cmd = [ffmpeg, "-v", "error", "-y", "-i", info.path, "-map", "0:v:0", "-map", "0:a:0?",
           "-fps_mode", "cfr", "-r", rate, "-c:v", "ffv1", "-level", "3",
           "-pix_fmt", "yuv444p16le" if info.high_bit_depth else "yuv444p",
           *color_tag_args(info), "-c:a", "pcm_s24le", str(out_path)]
    res = _run(cmd, timeout=None)
    if res.returncode != 0:
        raise VideoError(f"VFR→CFR conversion failed: {res.stderr.decode(errors='replace')[-800:]}")
    return out_path
