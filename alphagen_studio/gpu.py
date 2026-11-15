"""Zjištění GPU/VRAM/RAM bez nutnosti importovat torch v GUI procesu (nvidia-smi + psutil)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass

from .video_io import CREATE_NO_WINDOW

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mb: int
    free_mb: int
    used_mb: int
    driver: str
    compute_cap: str = ""

    @property
    def total_gb(self) -> float:
        return self.total_mb / 1024

    @property
    def is_blackwell(self) -> bool:
        try:
            return float(self.compute_cap) >= 10.0
        except ValueError:
            return False

    def label(self) -> str:
        return f"[{self.index}] {self.name} – {self.total_gb:.0f} GB ({self.free_mb / 1024:.1f} GB free)"

    def to_dict(self) -> dict:
        return asdict(self)


def list_gpus() -> list[GpuInfo]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        res = subprocess.run(  # noqa: S603
            [smi, "--query-gpu=index,name,memory.total,memory.free,memory.used,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=15, creationflags=CREATE_NO_WINDOW, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in res.stdout.decode(errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(GpuInfo(int(parts[0]), parts[1], int(float(parts[2])), int(float(parts[3])),
                                int(float(parts[4])), parts[5], parts[6] if len(parts) > 6 else ""))
        except ValueError:
            continue
    return gpus


def gpu_memory(index: int) -> tuple[int, int] | None:
    """(used_mb, total_mb) – pro živý ukazatel VRAM."""
    for g in list_gpus():
        if g.index == index:
            return g.used_mb, g.total_mb
    return None


def ram_info() -> tuple[float, float]:
    """(dostupná GB, celková GB)."""
    if psutil is None:
        return 0.0, 0.0
    vm = psutil.virtual_memory()
    return vm.available / 1024**3, vm.total / 1024**3


@dataclass
class MemoryPlan:
    quantization: str | None
    offload: str
    recommended_long_side: int
    recommended_segment: int
    feasible: bool
    notes: list[str]
    vae: str = "diffvae"
    segment_lowres: int = 0  # max. úsek, když se maska počítá do 1280 px (0 = stejné jako recommended_segment)

    def segment_for(self, work_long_side: int) -> int:
        """Nejdelší ověřený úsek pro dané pracovní rozlišení masky."""
        if self.segment_lowres and work_long_side <= 1280:
            return self.segment_lowres
        return self.recommended_segment


BF16_MIN_RAM_GB = 96  # celková RAM pro bf16 streamování (39 GB váhy + Gemma + systém)


def recommend(gpu: GpuInfo | None, ram_total_gb: float) -> MemoryPlan:
    """Konzervativní doporučení. Skutečná spotřeba se ověří až během běhu –
    při OOM nabídne aplikace menší rozlišení / kratší segment.

    Velikosti vah (oficiální soubory Lightricks/LTX-2.5): transformer 22B bf16 39,1 GB,
    Gemma 4 12B text encoder 24,5 GB, video VAE 1,4 GB, AlphaGen LoRA 1,2 GB.
    """
    notes: list[str] = []
    if gpu is None:
        return MemoryPlan(None, "cpu", 0, 0, False,
                          ["No NVIDIA GPU with CUDA found. AlphaGen is not practical on CPU."])
    vram = gpu.total_gb
    if vram >= 80:
        plan = MemoryPlan(None, "none", 1920, 121, True, ["Enough VRAM for bf16 without quantization or offload."])
    elif vram >= 40:
        plan = MemoryPlan("fp8-cast", "none", 1920, 121, True,
                          ["FP8 cast transformer (~22 GB), no offload."])
    elif vram >= 20:
        plan = MemoryPlan("fp8-cast", "cpu", 1920, 49, True,
                          ["Transformer does not fit in VRAM → FP8 + weight streaming from RAM (--offload cpu).",
                           "Lightweight convolutional VAE (same matte quality, less VRAM). Not verified on 24 GB – "
                           "shorten the segment on OOM."], vae="conv")
    else:
        plan = MemoryPlan("fp8-cast", "cpu", 1920, 49, True,
                          ["12 GB card: FP8 + offload to RAM + lightweight convolutional VAE.",
                           "Verified on RTX 4070 Ti: Full HD 49-frame segments (peak 9.2 GB VRAM); up to 1280 px 97-frame segments (8.2 GB).",
                           "On OOM (other apps holding VRAM) shorten the segment to 33.",
                           "The official DiffVAE fits in 12 GB only up to ~1280 px and 33 frames."], vae="conv")
    if plan.recommended_segment < 97:
        plan.segment_lowres = 97  # ověřeno na RTX 4070 Ti: 1280 px, 97 snímků, špička 8,2 GB
    if plan.offload == "cpu" and ram_total_gb and ram_total_gb >= BF16_MIN_RAM_GB:
        # Měřeno na RTX 4070 Ti: při streamování z RAM je bf16 stejně rychlé jako FP8 a maska je čistší
        # (závěs: alfa pozadí 0,05 proti 0,11–0,34 u FP8). Podmínkou je dost RAM pro bf16 váhy (39 GB).
        plan.quantization = None
        plan.notes = plan.notes + [f"{ram_total_gb:.0f} GB RAM → full bf16 precision (no FP8): same speed when "
                                   "streaming from RAM, cleaner matte (measured)."]
    if plan.offload == "cpu" and ram_total_gb and ram_total_gb < 64:
        notes.append(f"Only {ram_total_gb:.0f} GB RAM – offload to RAM may not be enough, try --offload disk.")
        plan.offload = "disk"
    plan.notes = plan.notes + notes
    return plan
