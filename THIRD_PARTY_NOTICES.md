# Third-party notices

AlphaGen Studio (© 2026 Petr Závorka, MIT – see [LICENSE](LICENSE)) is a front-end. It does **not** contain
or redistribute the AI models or the LTX-2 code. Everything listed below is owned by its respective
authors and is governed by **its own license**, not by the MIT license of this repository.
This file is a summary for convenience, not legal advice – the original license texts prevail.

## 1. Lightricks LTX-2.5 models (not included)

- **What:** LTX-2.5 22B distilled transformer, video/audio VAE, spatial upsampler, Gemma-based text encoder
  as packaged by Lightricks, and the **AlphaGen IC-LoRA** (`ltx-2.5-22b-ic-lora-alpha-gen-0.9`).
- **Owner:** Lightricks Ltd. – [huggingface.co/Lightricks](https://huggingface.co/Lightricks)
- **License:** **LTX-2.x Community License Agreement** – full text in
  [licenses/LTX-2.x-Community-License.txt](licenses/LTX-2.x-Community-License.txt) and on the model pages.
- **How they get to your PC:** each user downloads them directly from Hugging Face with their own account
  and must accept the license there. This repository never distributes the weights.

**Important points of the LTX-2.x Community License (please read the full text):**

- **Commercial entities** – organizations with **annual revenues of at least USD 10,000,000** need a
  **paid commercial license from Lightricks** for any use other than a Non-Commercial Purpose
  (§2.1, contact: licensing@ltx.io).
- **Non-Commercial Purpose** (§2.2) – personal, non-work use by an individual; or testing / evaluation /
  non-commercial R&D by a company in a non-production environment. Use for revenue-generating activity or
  with impact on end users (e.g. broadcast, client work) is **not** non-commercial.
- **Use restrictions** (§4 and Attachment A) apply to every user of the models. By using AlphaGen Studio
  with these models you agree to comply with them.
- **Output** (§5) – Lightricks claims no rights in the output you generate, but you are responsible for
  your input, the output and its use.
- **AI regulations** (§6) – you must comply with laws that apply to your use, incl. the EU AI Act.

## 2. LTX-2 inference code (not included – installed by `setup.bat`)

- **What:** `ltx-core`, `ltx-pipelines` from [github.com/Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)
  (cloned to `vendor/LTX-2` at commit `f0f2f86`).
- **License:** Apache License 2.0 (the Apache-licensed code only; using it with the models remains subject to
  the LTX-2.x Community License, see its §2.3).
- AlphaGen Studio does not modify these files. Two Windows compatibility adjustments are applied at runtime
  from `alphagen_studio/backend/worker_main.py` (pinned-memory allocation without POSIX `mmap` flags,
  `torch.inference_mode()` like the official CLI).

## 3. Text encoder based on Google Gemma

The LTX-2.5 text encoder (`gemma4-12b-with-proj-ltx-2.5`) is derived from Google's **Gemma** model.
Its use is subject to the terms stated on the Lightricks model page, including the applicable
**Google Gemma license / terms of use**.

## 4. Software installed by `setup.bat` / pip (not included in this repository)

| Component | License |
|---|---|
| PyTorch, torchvision, torchaudio | BSD-3-Clause style |
| Hugging Face `transformers`, `safetensors`, `huggingface_hub` | Apache-2.0 |
| Qt for Python (**PySide6**) | **LGPL-3.0** (installed as an unmodified, dynamically linked package) |
| OpenCV (`opencv-python`) | Apache-2.0 |
| OpenEXR | BSD-3-Clause |
| NumPy, SciPy, psutil | BSD-3-Clause |
| requests | Apache-2.0 |
| PyAV, OpenImageIO, colour-science and other LTX-2 dependencies | see their packages |
| **FFmpeg** (installed separately via `winget install Gyan.FFmpeg`) | LGPL-2.1+ / **GPL** depending on the build (the Gyan "full" build is GPL-3.0) |
| uv | Apache-2.0 / MIT |

AlphaGen Studio calls FFmpeg as a separate executable; it does not link to or redistribute it.

## 5. Example footage and images in `docs/`

The images in `docs/` are **not** covered by the MIT license:

- `docs/hero_before_after.png`, `docs/results_backlit_hair.jpg`, `docs/results_vs_lightricks.jpg` – frames from the
  example videos of the **Lightricks AlphaGen model card** (Complementary Materials of LTX-2.x, governed by the
  LTX-2.x Community License – a copy is included in `licenses/`). © Lightricks.
- `docs/results_hair.jpg` – frames from footage **© Getty Images**, shown for demonstration only under the
  repository owner's license. **Do not reuse.**
- `docs/settings_*.png`, `docs/queue.png` – screenshots of AlphaGen Studio (MIT).

## 6. Trademarks

LTX, LTX-2 and Lightricks are trademarks of Lightricks Ltd. Gemma is a trademark of Google LLC. NVIDIA, CUDA,
GeForce and RTX are trademarks of NVIDIA Corporation. ProRes is a trademark of Apple Inc. Getty Images is a
trademark of Getty Images. AlphaGen Studio is an independent project and is **not affiliated with or
endorsed by** Lightricks, Google, NVIDIA, Apple or Getty Images.
