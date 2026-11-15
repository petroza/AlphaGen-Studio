"""AlphaGen inference worker – běží v prostředí oficiálního LTX-2 (vendor/LTX-2/.venv).

Žádná náhradní segmentace, žádný ComfyUI: model se sestaví přesně stejným kódem
jako oficiální CLI ``python -m ltx_pipelines.ic_lora`` (stejný parser argumentů,
stejné ``ICLoraPipeline``), jen zůstane načtený v paměti mezi segmenty.

Recept podle docs.ltx.io (AlphaGen):
  distilled transformer + Gemma 4 text encoder + video/audio VAE + spatial upsampler,
  --lora ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors 1.0, --prompt "",
  --video-conditioning <zdroj> 1.0, --skip-stage-2, --width/--height = 2× zdroj.

Protokol: příkazy jako JSON řádky na stdin, události jako řádky ``@@AG {json}`` na stdout.
Vše ostatní (logy knihoven, tqdm) jde na stderr.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

PROTO = "@@AG "
_OUT = sys.stdout
sys.stdout = sys.stderr  # cizí print() nesmí rozbít protokol


def emit(ev: str, **kw) -> None:
    _OUT.write(PROTO + json.dumps({"ev": ev, **kw}, ensure_ascii=False) + "\n")
    _OUT.flush()


def _is_oom(exc: BaseException) -> bool:
    import torch

    oom_types = tuple(t for t in (getattr(torch, "OutOfMemoryError", None),
                                  getattr(torch.cuda, "OutOfMemoryError", None)) if t)
    if oom_types and isinstance(exc, oom_types):
        return True
    msg = str(exc).lower()
    return ("out of memory" in msg or "cublas_status_alloc_failed" in msg
            or "under the memory budget" in msg)  # DiffVAE decode se nevešel do VRAM


def _windows_compat() -> None:
    """Oprava jediné POSIX závislosti oficiálního loaderu na Windows (beze změny výsledků).

    ``ltx_core.loader.buffers.host_bytes`` alokuje velké pinned buffery přes
    ``mmap.mmap(-1, n, flags=MAP_PRIVATE | MAP_ANONYMOUS)`` – tyto konstanty i parametr ``flags``
    na Windows neexistují. Ekvivalent na Windows je anonymní mapování ``mmap.mmap(-1, n)``
    (stránkovaný soubor, zarovnané na stránky) a stejné ``cudaHostRegister``.
    """
    if os.name != "nt":
        return
    import mmap
    import weakref

    import torch
    from ltx_core.loader import buffers, registry, state_loader, streaming_builder

    orig = buffers.host_bytes

    def host_bytes(nbytes: int, *, pin: bool = False) -> torch.Tensor:
        if not pin or nbytes < buffers._EXACT_PIN_BYTES:  # noqa: SLF001
            return orig(nbytes, pin=pin)
        region = mmap.mmap(-1, nbytes)
        buffer = torch.frombuffer(region, dtype=torch.uint8)
        cudart = torch.cuda.cudart()
        if int(cudart.cudaHostRegister(buffer.data_ptr(), nbytes, 0)) != 0:
            raise RuntimeError(f"could not page-lock {nbytes} host bytes")
        weakref.finalize(region, cudart.cudaHostUnregister, buffer.data_ptr()).atexit = False
        return buffer

    for mod in (buffers, registry, state_loader, streaming_builder):
        mod.host_bytes = host_bytes


_STREAM_CACHE: dict = {}


def _keep_streaming_transformer() -> None:
    """Drží streamovaný transformer sestavený mezi úseky (výsledek beze změny).

    Oficiální ``DiffusionStage`` při offloadu sestaví streamovaný model (39 GB vah do pinned RAM
    + LoRA) při KAŽDÉM volání a po něm ho zavře – měřeno ~14 s na úsek. Váhy i LoRA jsou pro všechny
    úseky stejné, takže sestavený model lze znovu použít. Klíč cache = váhy + LoRA + kvantizace.
    """
    from contextlib import contextmanager
    from dataclasses import replace

    from ltx_pipelines.utils.blocks import diffusion_stage as ds

    if getattr(ds.DiffusionStage, "_alphagen_keep", False):
        return

    @contextmanager
    def ctx(self):
        builder = self._prepared_builder()
        selection = ds.transformer_checkpoint_selection_ops(builder.model_config())
        if selection is not None:
            builder = replace(builder, base=builder.base.with_sd_ops((*builder.base.sd_ops, selection)))
        base = builder.base
        key = (tuple(str(p) for p in base.paths),
               tuple((str(getattr(lo, "path", lo)), float(getattr(lo, "strength", 0) or 0)) for lo in base.loras),
               type(self._quantization).__name__, str(self._device), str(self._dtype))
        entry = _STREAM_CACHE.get(key)
        if entry is None:
            for old_key, (old_wrapped, _m) in list(_STREAM_CACHE.items()):  # jen jeden model najednou
                old_wrapped.close()
                _STREAM_CACHE.pop(old_key)
            wrapped = builder.build(device=self._device, dtype=self._dtype)
            entry = (wrapped, ds.X0Model(wrapped).eval())
            _STREAM_CACHE[key] = entry
        else:
            emit("log", level="INFO", message="Reusing the streamed transformer from the previous segment.")
        yield entry[1]

    ds.DiffusionStage._streaming_transformer_ctx = ctx
    ds.DiffusionStage._alphagen_keep = True


class _CachedPromptEncoder:
    """Prázdný prompt je pro všechny segmenty stejný → Gemma (24 GB) se spustí jen jednou.

    Výstup kódovače je deterministický (závisí jen na textu), takže cache nemění výsledek.
    """

    def __init__(self, inner, disk_dir: str | None = None, encoder_path: str = "") -> None:
        self.inner = inner
        self.cache: dict = {}
        self.disk_dir = Path(disk_dir) if disk_dir else None
        self.encoder_id = f"{Path(encoder_path).name}_{Path(encoder_path).stat().st_size if encoder_path else 0}"

    def _disk_file(self, key: tuple) -> Path | None:
        if not self.disk_dir:
            return None
        import hashlib

        h = hashlib.sha1(repr((self.encoder_id, key)).encode()).hexdigest()[:16]
        return self.disk_dir / f"prompt_{h}.pt"

    def __call__(self, prompts, **kwargs):
        import torch

        if any(kwargs.get(k) for k in ("enhance_first_prompt", "enhance_prompt_image")):
            return self.inner(prompts, **kwargs)
        key = tuple(prompts)
        if key in self.cache:
            return self.cache[key]
        f = self._disk_file(key)
        if f and f.exists():  # Gemma (24 GB) se vůbec nemusí načítat
            try:
                self.cache[key] = torch.load(f, map_location=getattr(self.inner, "_device", "cpu"), weights_only=False)
                emit("log", level="INFO", message="Prompt embedding loaded from disk cache (text encoder skipped).")
                return self.cache[key]
            except Exception:  # noqa: BLE001 - poškozená cache → přepočítat
                pass
        out = self.inner(prompts, **kwargs)
        self.cache[key] = out
        if f:
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                torch.save(out, f)
            except Exception:  # noqa: BLE001 - cache je jen zrychlení
                pass
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)


class Worker:
    def __init__(self) -> None:
        self.pipeline = None
        self.args = None
        self.load_cfg: dict | None = None

    # ------------------------------------------------------------------
    def _build_argv(self, cfg: dict, ref_video: str, width: int, height: int, frames: int, fps: float,
                    seed: int, out_path: str) -> list[str]:
        m = cfg["models"]
        argv = [
            "--transformer-path", m["transformer"],
            "--text-encoder-path", m["text_encoder"],
            "--video-vae-path", m["video_vae"],
            "--audio-vae-path", m["audio_vae"],
            "--spatial-upsampler-path", m["spatial_upsampler"],
            "--prompt", "",
            "--width", str(width), "--height", str(height),
            "--num-frames", str(frames), "--frame-rate", f"{fps:.6f}",
            "--video-conditioning", ref_video, str(cfg.get("conditioning_strength", 1.0)),
            "--lora", m["alpha_lora"], str(cfg.get("lora_strength", 1.0)),
            "--skip-stage-2",
            "--seed", str(seed),
            "--output-path", out_path,
        ]
        if cfg.get("quantization") and cfg["quantization"] != "none":
            argv += ["--quantization", cfg["quantization"]]
        if cfg.get("offload") and cfg["offload"] != "none":
            argv += ["--offload", cfg["offload"]]
        return argv

    def _parse(self, argv: list[str]):
        from ltx_pipelines import ic_lora as icl

        # resolve_cli_params čte verzi modelu z checkpointu uvedeného v sys.argv
        sys.argv = ["ltx_pipelines.ic_lora", *argv]
        params = icl.resolve_cli_params(distilled=True)
        parser = icl.add_out_of_clip_guides_arg(
            icl.add_chunk_layout_args(
                icl.add_keyframe_decode_arg(
                    icl.add_generated_keyframes_arg(icl.default_2_stage_distilled_arg_parser(params=params))
                )
            )
        )
        parser.add_argument("--video-conditioning", action=icl.VideoConditioningAction, nargs=2,
                            metavar=("PATH", "STRENGTH"), required=True)
        parser.add_argument("--skip-stage-2", action="store_true")
        return parser.parse_args(argv)

    # ------------------------------------------------------------------
    def load(self, cfg: dict) -> None:
        import torch
        from ltx_pipelines import ic_lora as icl

        t0 = time.time()
        for key, path in cfg["models"].items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing model '{key}': {path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the inference environment (torch.cuda.is_available() = False).")
        _windows_compat()
        dev = torch.cuda.get_device_properties(0)
        emit("log", level="INFO", message=f"GPU: {dev.name}, {dev.total_memory / 1024**3:.1f} GB, "
                                          f"torch {torch.__version__}, CUDA {torch.version.cuda}")
        probe = cfg["probe_video"]
        args = self._parse(self._build_argv(cfg, probe, 64, 64, 9, 24.0, 0, str(Path(probe).with_suffix(".x.mp4"))))
        emit("log", level="INFO", message=f"Loading ICLoraPipeline (quantization={cfg.get('quantization')}, "
                                          f"offload={cfg.get('offload')}) …")
        self.pipeline = icl.ICLoraPipeline(
            model_paths=args.model_paths,
            spatial_upsampler_path=args.spatial_upsampler_path,
            loras=tuple(args.lora) if args.lora else (),
            quantization=args.quantization,
            compilation_config=args.compile,
            offload_mode=args.offload_mode,
            prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
            diffvae_optimization=args.diffvae_optimization,
            stages=icl.resolve_ic_lora_stages(height=128, width=128, tile=False, skip_stage_2=True),
        )
        self.pipeline.prompt_encoder = _CachedPromptEncoder(self.pipeline.prompt_encoder, cfg.get("cache_dir"),
                                                            cfg["models"]["text_encoder"])
        if cfg.get("keep_transformer", True):
            _keep_streaming_transformer()
        self.load_cfg = cfg
        emit("loaded", seconds=round(time.time() - t0, 1),
             lora_reference_downscale=getattr(self.pipeline, "reference_downscale_factor", 1))

    # ------------------------------------------------------------------
    def run(self, req: dict) -> None:
        import numpy as np
        import torch
        from ltx_core.model.video_vae import AUTO_TILING
        from PIL import Image

        if self.pipeline is None:
            raise RuntimeError("Model is not loaded.")
        cfg = self.load_cfg
        out_dir = Path(req["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        # Ověř argumenty přes oficiální parser (validace rozměrů, cest, 8n+1…)
        args = self._parse(self._build_argv(cfg, req["video"], req["cli_width"], req["cli_height"],
                                            req["num_frames"], req["fps"], req["seed"],
                                            str(out_dir / "official_cli_equivalent.mp4")))
        (out_dir / "cli_args.json").write_text(json.dumps(sys.argv[1:], ensure_ascii=False, indent=1), "utf-8")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        chunks, num_frames, _ = self.pipeline.stream_chunks(
            prompt=args.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            images=args.images,
            video_conditioning=args.video_conditioning,
            enhance_prompt=False,
            vae_dtype=torch.bfloat16,
            color_space=None,
            tiling_config=AUTO_TILING,
            conditioning_attention_strength=1.0,
            skip_stage_2=True,
            conditioning_attention_mask=None,
            generated_keyframes=args.num_generated_keyframes,
            decode_with_keyframes=args.decode_with_keyframes,
            chunk_config=None,
            out_of_clip_guides=args.out_of_clip_guides,
            audio=None,
        )
        written = 0
        exp_w, exp_h = req["cli_width"] // 2, req["cli_height"] // 2
        for seg in chunks:
            v = seg.video
            if v.dtype == torch.uint8:
                v = v.float() / 255.0
            v = v.float()
            if float(v.min()) < -0.01:  # pojistka pro [-1, 1]
                v = (v + 1.0) / 2.0
            v = v.clamp(0.0, 1.0)
            # Model vrací šedotónové RGB – matte = průměr kanálů (bílá = popředí)
            matte = v.mean(dim=-1).cpu().numpy()
            for frame in matte:
                if frame.shape != (exp_h, exp_w):
                    raise RuntimeError(f"Model output {frame.shape[1]}×{frame.shape[0]} ≠ expected {exp_w}×{exp_h}")
                img = Image.fromarray(np.round(frame * 65535.0).astype(np.uint16))
                img.save(out_dir / f"{written:06d}.png", compress_level=1)
                written += 1
                if written % 8 == 0:
                    emit("frames", id=req["id"], written=written, total=num_frames)
        if written != req["num_frames"]:
            raise RuntimeError(f"Model returned {written} frames, expected {req['num_frames']}.")
        peak = torch.cuda.max_memory_allocated() / 1024**3
        (out_dir / "DONE").write_text(json.dumps({"frames": written, "seconds": time.time() - t0}), "utf-8")
        emit("done", id=req["id"], frames=written, seconds=round(time.time() - t0, 2), peak_vram_gb=round(peak, 2))


def main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    worker = Worker()
    emit("ready", pid=os.getpid(), python=sys.version.split()[0])
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            emit("error", kind="protocol", message="Invalid JSON command.")
            continue
        cmd = req.get("cmd")
        try:
            if cmd in ("load", "run"):
                import torch

                # Stejně jako oficiální main() (@torch.inference_mode) – vše bez autogradu
                with torch.inference_mode():
                    (worker.load if cmd == "load" else worker.run)(req)
            elif cmd == "ping":
                emit("pong")
            elif cmd == "quit":
                emit("bye")
                return
            else:
                emit("error", kind="protocol", message=f"Unknown command {cmd}")
        except Exception as exc:  # noqa: BLE001
            kind = "oom" if _is_oom(exc) else ("missing_model" if isinstance(exc, FileNotFoundError) else "other")
            if kind == "oom":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            emit("error", kind=kind, id=req.get("id"), message=str(exc)[:2000],
                 traceback=traceback.format_exc()[-6000:])


if __name__ == "__main__":
    main()
