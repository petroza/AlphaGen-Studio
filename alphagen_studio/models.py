"""Správce modelových souborů – oficiální zdroje Hugging Face (Lightricks).

Repozitáře jsou "gated": před stažením je nutné na webu přijmout licenci
LTX-2.x Community License a mít HF token s právem čtení.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CONFIG_DIR, AppSettings
from .logging_setup import get_logger

log = get_logger("models")

HF = "https://huggingface.co"
BASE_REPO = "Lightricks/LTX-2.5"
ALPHA_REPO = "Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen"
MANIFEST_CACHE = CONFIG_DIR / "model_manifest.json"


@dataclass
class ModelFile:
    key: str
    repo: str
    path_in_repo: str
    label: str
    size: int = 0  # bajty (z HF API)
    sha256: str = ""  # LFS oid (z HF API)

    @property
    def filename(self) -> str:
        return Path(self.path_in_repo).name

    @property
    def subdir(self) -> str:
        return "ltx-2.5" if self.repo == BASE_REPO else "alpha-gen"

    def url(self) -> str:
        return f"{HF}/{self.repo}/resolve/main/{self.path_in_repo}"

    def page(self) -> str:
        return f"{HF}/{self.repo}"


# Přesně podle oficiální dokumentace AlphaGen (docs.ltx.io) – distilled recept
REQUIRED: list[ModelFile] = [
    ModelFile("transformer", BASE_REPO, "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
              "Transformer LTX-2.5 22B distilled (bf16)", 42_018_190_584),
    ModelFile("text_encoder", BASE_REPO, "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
              "Text encoder Gemma 4 12B + projection (bf16)", 26_263_858_182),
    ModelFile("video_vae", BASE_REPO, "vae/ltx-2.5-video-vae-bf16.safetensors", "Video VAE (bf16)", 1_472_223_346),
    ModelFile("audio_vae", BASE_REPO, "vae/ltx-2.5-audio-vae-bf16.safetensors", "Audio VAE (bf16)", 364_866_540),
    ModelFile("spatial_upsampler", BASE_REPO,
              "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
              "Spatial upsampler x2 (required by the interface, unused with --skip-stage-2)", 995_778_752),
    ModelFile("alpha_lora", ALPHA_REPO, "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors",
              "AlphaGen IC-LoRA 0.9", 1_308_787_472),
]


# Volitelné: oficiální konvoluční video VAE z téhož repozitáře – výrazně méně VRAM při dekódování
# (DiffVAE z dokumentace se na 12–16 GB kartách při Full HD nevejde ani na 17 snímků).
OPTIONAL: list[ModelFile] = [
    ModelFile("video_vae_conv", BASE_REPO, "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
              "Video VAE convolutional (lightweight, optional)", 1_452_269_922),
]


class ModelError(RuntimeError):
    pass


class ModelManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.files = [ModelFile(**asdict(f)) for f in REQUIRED]
        self.optional = [ModelFile(**asdict(f)) for f in OPTIONAL]
        self._load_manifest_cache()

    def has_conv_vae(self) -> bool:
        f = self.optional[0]
        p = self.path_for(f)
        return p.exists() and p.stat().st_size == f.size

    def conv_vae_path(self) -> str:
        return str(self.path_for(self.optional[0]))

    # ------------------------------------------------------------------
    @property
    def overrides(self) -> dict:
        return self.settings.extra.setdefault("model_paths", {})

    def path_for(self, f: ModelFile) -> Path:
        if f.key in self.overrides:
            return Path(self.overrides[f.key])
        return Path(self.settings.models_dir) / f.subdir / f.path_in_repo

    def paths(self) -> dict[str, str]:
        return {f.key: str(self.path_for(f)) for f in self.files}

    def _load_manifest_cache(self) -> None:
        if not MANIFEST_CACHE.exists():
            return
        try:
            data = json.loads(MANIFEST_CACHE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for f in self.files:
            meta = data.get(f"{f.repo}/{f.path_in_repo}")
            if meta:
                f.size = int(meta.get("size", f.size))
                f.sha256 = meta.get("sha256", "")

    def refresh_manifest(self, token: str | None = None) -> None:
        """Načte přesné velikosti a SHA256 z HF API (veřejný seznam souborů)."""
        import requests

        data = {}
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        for repo in {f.repo for f in self.files}:
            r = requests.get(f"{HF}/api/models/{repo}/tree/main", params={"recursive": "true"},
                             headers=headers, timeout=30)
            r.raise_for_status()
            for item in r.json():
                if item.get("type") != "file":
                    continue
                lfs = item.get("lfs") or {}
                data[f"{repo}/{item['path']}"] = {"size": item.get("size", 0), "sha256": lfs.get("oid", "")}
        for f in self.files:
            meta = data.get(f"{f.repo}/{f.path_in_repo}")
            if not meta:
                raise ModelError(f"File {f.path_in_repo} is no longer in the official repository {f.repo} – "
                                 "check the current AlphaGen documentation.")
            f.size, f.sha256 = int(meta["size"]), meta["sha256"]
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST_CACHE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def status(self) -> list[dict]:
        out = []
        for f in self.files:
            p = self.path_for(f)
            exists = p.exists()
            size = p.stat().st_size if exists else 0
            partial = p.with_name(p.name + ".part")
            psize = partial.stat().st_size if partial.exists() else 0
            out.append({
                "key": f.key, "label": f.label, "path": str(p), "exists": exists, "size": size,
                "expected": f.size, "size_ok": exists and (f.size == 0 or size == f.size),
                "partial": psize, "sha256": f.sha256, "repo": f.repo,
            })
        return out

    def missing(self) -> list[ModelFile]:
        st = {s["key"]: s for s in self.status()}
        return [f for f in self.files if not st[f.key]["size_ok"]]

    def all_present(self) -> bool:
        return not self.missing()

    def total_missing_bytes(self) -> int:
        st = {s["key"]: s for s in self.status()}
        return sum(max(0, f.size - st[f.key]["partial"]) for f in self.missing())

    # ------------------------------------------------------------------
    def find_existing(self, folder: str | Path) -> dict[str, str]:
        """Prohledá složku (např. starší stažení) a převezme nalezené soubory."""
        found = {}
        folder = Path(folder)
        names = {f.filename: f for f in self.files}
        for p in folder.rglob("*.safetensors"):
            f = names.get(p.name)
            if f and (f.size == 0 or p.stat().st_size == f.size):
                found[f.key] = str(p)
        self.overrides.update(found)
        self.settings.save()
        return found

    def verify_sha256(self, f: ModelFile, progress: Callable[[int, int], None] | None = None,
                      cancel: threading.Event | None = None) -> bool:
        if not f.sha256:
            raise ModelError("Checksum missing – fetch the manifest from Hugging Face first.")
        p = self.path_for(f)
        h = hashlib.sha256()
        done = 0
        total = p.stat().st_size
        with open(p, "rb") as fh:
            while chunk := fh.read(16 * 1024 * 1024):
                if cancel is not None and cancel.is_set():
                    raise ModelError("Verification cancelled.")
                h.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        return h.hexdigest() == f.sha256

    # ------------------------------------------------------------------
    def download(self, f: ModelFile, token: str | None,
                 progress: Callable[[str, int, int, float], None] | None = None,
                 cancel: threading.Event | None = None, connections: int = 12) -> Path:
        """Paralelní stažení (víc HTTP Range spojení najednou), navazuje po 64MB blocích.

        Jedno spojení na CDN Hugging Face bývá omezené na jednotky MB/s, souběžné bloky to obchází.
        Pokud server Range neumí, spadne na jednoduché stažení."""
        if connections <= 1 or not f.size:
            return self._download_single(f, token, progress, cancel)
        import requests
        from concurrent.futures import ThreadPoolExecutor

        dest = self.path_for(f)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        state_file = dest.with_name(dest.name + ".part.json")
        chunk = 64 * 1024 * 1024
        n_chunks = (f.size + chunk - 1) // chunk
        done: set[int] = set()
        if part.exists() and state_file.exists():
            try:
                done = set(json.loads(state_file.read_text())["done"])
            except (OSError, ValueError, KeyError):
                done = set()
        if not part.exists() or part.stat().st_size != f.size:
            done = set()
            with open(part, "wb") as fh:  # předalokace
                fh.truncate(f.size)
        headers = {"User-Agent": "AlphaGenStudio/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # Podepsaná URL na CDN (redirect) – platí cca hodinu, při 403 se obnoví
        url_lock = threading.Lock()
        url_box = {"url": None, "t": 0.0}

        def resolve() -> str:
            with url_lock:
                if url_box["url"] and time.time() - url_box["t"] < 1800:
                    return url_box["url"]
                r = requests.head(f.url(), headers=headers, allow_redirects=False, timeout=60)
                if r.status_code in (401, 403):
                    raise ModelError(f"Access denied ({r.status_code}). Repository {f.repo} is gated: accept "
                                     f"the license at {f.page()} and enter an HF token (Read).")
                url_box["url"] = r.headers.get("Location") or f.url()
                url_box["t"] = time.time()
                return url_box["url"]

        lock = threading.Lock()
        have = [len(done) * chunk]
        t0, base = time.time(), have[0]

        def fetch(idx: int) -> None:
            start = idx * chunk
            end = min(f.size, start + chunk) - 1
            for attempt in range(6):
                if cancel is not None and cancel.is_set():
                    raise ModelError("Download cancelled (can be resumed).")
                written = 0
                try:
                    url = resolve()
                    h = {"User-Agent": headers["User-Agent"], "Range": f"bytes={start}-{end}"}
                    if url == f.url() and token:
                        h["Authorization"] = headers["Authorization"]
                    with requests.get(url, headers=h, stream=True, timeout=60) as r:
                        if r.status_code == 403:
                            url_box["url"] = None  # vypršel podpis
                            raise OSError("403")
                        if r.status_code != 206:
                            raise OSError(f"HTTP {r.status_code}")
                        pos = start
                        with open(part, "r+b") as fh:
                            fh.seek(start)
                            for data in r.iter_content(1024 * 1024):
                                if cancel is not None and cancel.is_set():
                                    raise ModelError("Download cancelled (can be resumed).")
                                fh.write(data)
                                pos += len(data)
                                written += len(data)
                                with lock:
                                    have[0] += len(data)
                                    if progress:
                                        progress(f.key, min(have[0], f.size), f.size,
                                                 (have[0] - base) / max(1e-3, time.time() - t0))
                        if pos != end + 1:
                            raise OSError("incomplete chunk")
                    with lock:
                        done.add(idx)
                        state_file.write_text(json.dumps({"done": sorted(done)}))
                    return
                except ModelError:
                    raise
                except Exception:  # noqa: BLE001 - síť: zkusit znovu
                    with lock:
                        have[0] -= written
                    time.sleep(2 * (attempt + 1))
            raise ModelError(f"{f.filename}: failed to download chunk {idx}.")

        todo = [i for i in range(n_chunks) if i not in done]
        with ThreadPoolExecutor(max_workers=connections) as ex:
            for fut in [ex.submit(fetch, i) for i in todo]:
                fut.result()
        if part.stat().st_size != f.size or len(done) != n_chunks:
            raise ModelError(f"{f.filename}: download incomplete, try again (it will resume).")
        part.replace(dest)
        state_file.unlink(missing_ok=True)
        log.info("Downloaded: %s", dest)
        return dest

    def _download_single(self, f: ModelFile, token: str | None,
                         progress: Callable[[str, int, int, float], None] | None = None,
                         cancel: threading.Event | None = None) -> Path:
        """Stáhne soubor jedním spojením s navazováním (.part + HTTP Range)."""
        import requests

        dest = self.path_for(f)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "AlphaGenStudio/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if have:
            headers["Range"] = f"bytes={have}-"
        with requests.get(f.url(), headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
            if r.status_code in (401, 403):
                raise ModelError(
                    f"Access denied ({r.status_code}). Repository {f.repo} is gated: sign in at "
                    f"{f.page()}, accept the license and enter an HF token (Read)."
                )
            if r.status_code == 416:  # už staženo celé
                have = part.stat().st_size
            else:
                r.raise_for_status()
                if have and r.status_code != 206:
                    have = 0  # server neumí Range → od začátku
                total = have + int(r.headers.get("Content-Length", 0))
                mode = "ab" if have else "wb"
                t0, done0 = time.time(), have
                with open(part, mode) as fh:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if cancel is not None and cancel.is_set():
                            raise ModelError("Download cancelled (can be resumed).")
                        fh.write(chunk)
                        have += len(chunk)
                        if progress:
                            dt = max(1e-3, time.time() - t0)
                            progress(f.key, have, total or f.size, (have - done0) / dt)
        size = part.stat().st_size
        if f.size and size != f.size:
            raise ModelError(f"{f.filename}: downloaded {size} B, expected {f.size} B. Try downloading again.")
        part.replace(dest)
        log.info("Downloaded: %s", dest)
        return dest
