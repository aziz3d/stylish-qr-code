"""
run_local.py — Local launcher for AI QR Code Generator
-------------------------------------------------------
Patches the `spaces` module so the @spaces.GPU decorator
becomes a no-op, allowing the app to run without Hugging Face
ZeroGPU infrastructure.

Usage:
    python run_local.py
    python run_local.py --share          # expose a public Gradio link
    python run_local.py --port 7860      # custom port (default: 7860)
    python run_local.py --cpu            # force CPU (slow but works without GPU)
"""

import sys
import os
import argparse
import types

# ── Parse local args before app.py sees sys.argv ─────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--share", action="store_true", default=False)
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--cpu", action="store_true", default=False)
local_args, remaining = parser.parse_known_args()

# Strip our custom args so ComfyUI's cli_args parser doesn't choke on them
sys.argv = [sys.argv[0]] + remaining

# ── Stub out the `spaces` package (HF ZeroGPU) ───────────────────────────────
# The real `spaces` package requires HF infrastructure.
# Locally we just want @spaces.GPU to be a transparent pass-through decorator.

def _gpu_decorator(fn=None, duration=None):
    """No-op replacement for @spaces.GPU"""
    if fn is None:
        # Called as @spaces.GPU(duration=...) — return a decorator
        def decorator(f):
            return f
        return decorator
    # Called as @spaces.GPU directly
    return fn

fake_spaces = types.ModuleType("spaces")
fake_spaces.GPU = _gpu_decorator
sys.modules["spaces"] = fake_spaces

# ── Force CPU mode if requested ───────────────────────────────────────────────
if local_args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("⚠️  CPU mode enabled — generation will be slow.")

# ── Disable analytics (no Supabase/PostHog needed locally) ───────────────────
os.environ.setdefault("ANALYTICS_ENABLED", "false")

# ── Tell the app it's NOT running on HF Spaces ───────────────────────────────
# This skips the AOT compilation step that can fail locally
os.environ.pop("SPACE_ID", None)
os.environ.pop("SPACE_HOST", None)

# ── Monkey-patch demo.launch to inject our local args ────────────────────────
# We intercept after app.py builds `demo` but before it calls demo.launch()
import importlib, runpy

_original_launch = None

def _patch_gradio_launch():
    try:
        import gradio as gr
        _orig = gr.Blocks.launch

        def _patched_launch(self, *args, **kwargs):
            kwargs.setdefault("share", local_args.share)
            kwargs.setdefault("server_port", local_args.port)
            kwargs.pop("mcp_server", None)   # MCP server not needed locally
            print(f"\n🚀  Launching on http://127.0.0.1:{local_args.port}")
            if local_args.share:
                print("🌐  Public share link will appear below...")
            return _orig(self, *args, **kwargs)

        gr.Blocks.launch = _patched_launch
    except Exception as e:
        print(f"[run_local] Warning: could not patch Gradio launch: {e}")

_patch_gradio_launch()

# ── Run app.py ────────────────────────────────────────────────────────────────
print("=" * 60)
print("  AI QR Code Generator — Local Mode")
print(f"  URL : http://127.0.0.1:{local_args.port}")
print(f"  GPU : {'CPU only' if local_args.cpu else 'auto-detect'}")
print("=" * 60)
print()

# execfile-style run so app.py's __name__ == '__main__' block fires
runpy.run_path(
    os.path.join(os.path.dirname(__file__), "app.py"),
    run_name="__main__",
)
