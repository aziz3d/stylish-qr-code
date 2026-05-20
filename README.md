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

## Running Locally

You can run this app entirely on your own machine — no API keys or cloud GPU required. Models are downloaded automatically from Hugging Face Hub on first launch and cached locally.

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

**Step 3 — Launch the app:**

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
```

### How local mode works

The app was originally built for Hugging Face ZeroGPU (the `@spaces.GPU` decorator). The included `run_local.py` script patches this decorator into a transparent no-op so the app runs without any HF infrastructure. Analytics are also disabled by default locally — no Supabase or PostHog credentials are needed.

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
- Plain text payloads are left untouched.

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
