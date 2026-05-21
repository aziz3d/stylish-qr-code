---
title: AI QR Code Generator
emoji: 🌍
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
tags:
- mcp-server-track
---

# AI QR Code Generator

> **Original repository:** This project is a clone of the [AI QR Code Generator](https://huggingface.co/spaces/Oysiyl/AI-QR-code-generator) Space originally created and maintained by [Oysiyl](https://huggingface.co/Oysiyl) on Hugging Face.

This GitHub mirror is maintained by **Aziz Khan** ([@aziz3d](https://github.com/aziz3d)). Updates, improvements, and fixes are applied here on GitHub while staying in sync with the upstream Hugging Face Space.

---

## Demo

[Watch the demo video](showcase.mp4)

---

## Recent Changes

### Bug Fixes

**URL normalization crash on non-HTTP payloads**
When `Input Type` was set to `URL` but the content was a non-HTTP string (e.g. a WiFi QR payload like `WIFI:T:WPA;S:MyNetwork;P:MyPassword123;;`), `urllib.parse.urlsplit` would misparse the string and `parsed.port` would raise a `ValueError`. The normalizer now wraps `parsed.port` in a `try/except ValueError` and returns the input unchanged if parsing fails — matching the intended behaviour of leaving non-URL payloads untouched.

**Second-pass ksampler device mismatch (CUDA / CPU)**
The artistic pipeline runs two sampling passes. After the first threaded pass and a `soft_empty_cache()` call, ComfyUI's memory manager can partially offload UNet layers back to CPU to make room for the VAE. When the second pass starts, early layers like `time_embed` remain on CPU while input tensors are on CUDA, causing:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```
The fix sets `comfy_cast_weights = True` on every layer of the diffusion model before the second-pass thread runs. This activates the `forward_comfy_cast_weights` path which casts each layer's weights to the input device at forward-time — the same mechanism used by `--lowvram` mode — so the pass works regardless of where weights physically reside. The latent and noise tensors are also explicitly moved to the target device before sampling.

**Generated QR codes not scannable**
The pipeline relied entirely on ControlNet to preserve the QR pattern during diffusion. When ControlNet strength was insufficient, or the second pass (which uses the first-pass AI output as its ControlNet image rather than the original QR) drifted too far, the QR pattern degraded to the point where phone scanners could not read it.

The fix adds `_composite_qr_onto_image()` which blends the original clean QR back onto the final AI image at `blend_strength=0.85` as the very last step in both pipelines:
- Dark QR modules are pushed 85% toward black (AI texture still visible at 15% opacity)
- Light QR modules are pushed 85% toward white (AI texture still visible at 15% opacity)

This guarantees scannability regardless of how well the AI preserved the QR structure, while keeping the artistic look visible through the modules. Applied after all post-processing (upscaling, color quantization) in both the Standard and Artistic pipelines.

**Gradio theme startup crash**
`shadow_drop_dark` is not a valid property in Gradio 5's theme API. Replaced with `block_shadow_dark` which is the correct equivalent.

### UI Improvements

**Dark theme and styled header**
The web UI has been redesigned with a dark theme using Gradio 5's typed theme API (`gr.themes.Base().set(...)`). Key changes:

- Deep dark background (`#0f0f13`) across the full page
- Block and input fills use layered dark blues (`#1e1e2e`, `#1a1a2e`)
- Primary buttons use a purple-to-indigo gradient with a lift-on-hover effect
- Input focus states show a purple glow ring
- Inter font loaded via Google Fonts

The flat markdown header is replaced with a styled HTML hero card featuring:
- Plain white title with a purple-blue gradient accent on the word "Art"
- Subtitle and all body text in high-contrast slate tones (`#f1f5f9`, `#94a3b8`, `#cbd5e1`)
- Four pill badges: Artistic Pipeline, Standard Pipeline, Auto-delete, GPU Accelerated
- A 4-card info grid showing quota limits and tips
- Subtle radial glow blobs (purple top-right, blue bottom-left) on a solid dark background (`#13131a`)

The two separate notice paragraphs are consolidated into a single styled notice bar.

**Dark/light/system mode toggle fixed**
Theme properties were previously set on both the plain and `_dark` variants, locking every mode to dark colors. Now only `_dark`-suffixed properties carry the dark palette; plain (light) properties fall back to Gradio's built-in light defaults. Switching to Light or System mode in the settings panel now works correctly.

### Configuration

**HF_TOKEN support in launch.bat**
`launch.bat` now includes a placeholder `HF_TOKEN` line. Fill in a Read token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) to suppress the unauthenticated rate-limit warning from `huggingface_hub` and get faster model downloads. The token is set as a local environment variable and is never committed to git.

**`.gitignore` updated**
`.env`, `*.env`, and `secrets.bat` are now excluded from git to prevent accidental credential commits.

---

## Running Locally

You can run this app entirely on your own machine, no API keys or cloud GPU required. Models are downloaded automatically from Hugging Face Hub on first launch and cached locally.

### Requirements

- **Python 3.12** (recommended; 3.11 also works)
- **NVIDIA GPU** with CUDA 12.1+ (recommended — RTX 3060 / 8 GB VRAM or better)
- ~10 GB free disk space for models
- ~2.5 GB for PyTorch + dependencies

> CPU-only mode is supported via `launch.bat --cpu` but generation will be very slow.

### Models downloaded on first run

| Model | Size (approx.) |
|---|---|
| Stable Diffusion v1.5 | ~4 GB |
| DreamShaper 3.32 & 6.31 | ~2 GB |
| ControlNet Brightness + Tile | ~1.5 GB |
| VAE (DreamShaper + MSE) | ~0.3 GB |
| RealESRGAN x4 upscaler | ~0.06 GB |

### Setup

**Step 1 — Create the virtual environment** (already done if you cloned this repo and ran the setup):

```bash
python -m venv venv
```

**Step 2 — Install dependencies** (one-time):

Double-click `install.bat` or run manually:

```bash
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Step 3 — (Optional) Set your Hugging Face token:**

Open `launch.bat` and replace `YOUR_NEW_TOKEN_HERE` with a Read token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). This suppresses the rate-limit warning and speeds up model downloads. The token is never committed to git.

**Step 4 — Launch the app:**

Double-click `launch.bat` or run:

```bash
venv\Scripts\activate
python run_local.py
```

Then open `http://127.0.0.1:7860` in your browser.

### Launch options

```bash
launch.bat                  # default — http://127.0.0.1:7860
launch.bat --share          # generate a public Gradio URL
launch.bat --port 8080      # use a custom port
launch.bat --cpu            # force CPU mode (no GPU needed, slow)
launch.bat --highvram       # keep all models in VRAM (needs ≥12 GB free)
launch.bat --normalvram     # use ComfyUI default VRAM mode
```

### How local mode works

The app was originally built for Hugging Face ZeroGPU (the `@spaces.GPU` decorator). The included `run_local.py` script patches this decorator into a transparent no-op so the app runs without any HF infrastructure. By default `run_local.py` injects `--lowvram` into ComfyUI's argument parser, which keeps all model layer weights on CPU and casts them to the GPU on-the-fly. This prevents device-mismatch crashes on GPUs with limited VRAM. Pass `--highvram` or `--normalvram` to override if you have plenty of VRAM. Analytics are also disabled by default locally — no Supabase or PostHog credentials are needed.

---

## Optional analytics (HF Space / cloud deployments)

The Space includes an optional analytics consent toggle for both UI and MCP usage.

- Generated images are not used for analytics.
- Full prompts, QR payload text, and settings are only stored when analytics opt-in is enabled.
- Minimal operational events can still be logged for reliability and product metrics.

## URL normalization

When `Input Type` is set to `URL`, the app normalizes links on CPU before any GPU work starts.

- Common tracking params such as `utm_*`, `fbclid`, and `gclid` are removed automatically.
- Scheme and host casing are normalized to reduce unnecessary QR payload length.
- Non-HTTP payloads (WiFi, vCard, plain text, etc.) are detected and left untouched.

## Temporary short links

When enabled in the UI, URL mode can replace the QR payload with a `qrcut.co` short link.

- The shortener is opt-in and defaults to off.
- Short links expire after 7 days of inactivity.
- If the shortener is unavailable, generation falls back to the normalized original URL.

### Required Space secrets (cloud deployments only)

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `POSTHOG_API_KEY` (or reuse `POSTHOG_SECRET_KEY`)
- `POSTHOG_HOST` (optional, defaults to `https://us.i.posthog.com`)
- `ANALYTICS_ENABLED` (optional, defaults to `true`)
- `ANALYTICS_DEFAULT_OPT_IN` (optional, defaults to `false`)
- `ANALYTICS_PRODUCT` (optional, defaults to `ai_qr_generator`; set Modal/iOS deployments to `arti_qrcode_app` when sharing the same analytics tables)
- `URL_SHORTENER_API_URL` (optional, for example `https://qrcut.co/shorten`)
- `URL_SHORTENER_API_KEY` (optional, private API key for the HF Space)
- `URL_SHORTENER_SOURCE_APP` (optional, defaults to `ai_qr_generator`)
- `URL_SHORTENER_TIMEOUT_SECONDS` (optional, defaults to `10`)

### Supabase schema

Apply `analytics_supabase_schema.sql` to your Supabase project before enabling writes from the Space.

The analytics schema is intended for backend-only writes using `SUPABASE_SERVICE_ROLE_KEY`.
Do not grant `anon` or `authenticated` direct access to the analytics tables or derived views.
