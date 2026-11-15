"""Cesty a uživatelské nastavení (přenosné – vše uvnitř složky aplikace)."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = APP_ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
LOG_DIR = APP_ROOT / "logs"
DEFAULT_LTX_REPO = APP_ROOT / "vendor" / "LTX-2"


@dataclass
class AppSettings:
    models_dir: str = str(APP_ROOT / "models")
    cache_dir: str = str(APP_ROOT / "cache")
    work_dir: str = str(APP_ROOT / "work")
    output_dir: str = str(APP_ROOT / "output")
    ltx_repo: str = str(DEFAULT_LTX_REPO)
    ltx_python: str = ""  # prázdné = <ltx_repo>/.venv/Scripts/python.exe
    ffmpeg: str = ""  # prázdné = automaticky (PATH nebo tools/ffmpeg)
    ffprobe: str = ""
    last_input_dir: str = ""
    # Výchozí volby inference – lze přepsat v GUI
    quantization: str = "auto"  # auto | none | fp8-cast | fp8-scaled-mm | nvfp4-prequant
    offload: str = "auto"  # auto | none | cpu | disk
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls) -> AppSettings:
        data = {}
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(SETTINGS_FILE)

    # ------------------------------------------------------------------
    def ltx_python_path(self) -> Path:
        if self.ltx_python:
            return Path(self.ltx_python)
        return Path(self.ltx_repo) / ".venv" / "Scripts" / "python.exe"

    def ffmpeg_path(self) -> str:
        return self.ffmpeg or find_tool("ffmpeg")

    def ffprobe_path(self) -> str:
        return self.ffprobe or find_tool("ffprobe")

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.cache_dir, self.work_dir, self.output_dir, LOG_DIR):
            Path(d).mkdir(parents=True, exist_ok=True)


def find_tool(name: str) -> str:
    """Najde ffmpeg/ffprobe: nejdřív přibalený v tools/, pak PATH."""
    exe = name + (".exe" if os.name == "nt" else "")
    bundled = APP_ROOT / "tools" / "ffmpeg" / "bin" / exe
    if bundled.exists():
        return str(bundled)
    found = shutil.which(name)
    return found or ""
