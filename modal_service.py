from __future__ import annotations

import importlib.util
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import modal


FILE_DIR = Path(__file__).resolve().parent
for candidate in (FILE_DIR, Path("/root/app")):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _load_support_module(module_name: str):
    for base_dir in (FILE_DIR, Path("/root/app")):
        candidate = base_dir / f"{module_name}.py"
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {candidate}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ModuleNotFoundError(module_name)


try:
    from qr_modal_contract import (
        GenerateRequest,
        build_generation_kwargs,
        build_response_payload,
        consume_final_result,
        resolve_request_seed,
    )
    from qr_modal_loader import load_module_from_file
    from qr_modal_requirements import read_modal_requirements
except ModuleNotFoundError:
    qr_modal_contract = _load_support_module("qr_modal_contract")
    qr_modal_loader = _load_support_module("qr_modal_loader")
    qr_modal_requirements = _load_support_module("qr_modal_requirements")

    GenerateRequest = qr_modal_contract.GenerateRequest
    build_generation_kwargs = qr_modal_contract.build_generation_kwargs
    build_response_payload = qr_modal_contract.build_response_payload
    consume_final_result = qr_modal_contract.consume_final_result
    resolve_request_seed = qr_modal_contract.resolve_request_seed
    load_module_from_file = qr_modal_loader.load_module_from_file
    read_modal_requirements = qr_modal_requirements.read_modal_requirements


APP_NAME = os.environ.get("MODAL_APP_NAME", "ai-qr-code-generator-api")
GPU = os.environ.get("MODAL_GPU", "A100-40GB")
TIMEOUT_SECONDS = int(os.environ.get("MODAL_TIMEOUT_SECONDS", "1800"))
SCALEDOWN_WINDOW = int(os.environ.get("MODAL_SCALEDOWN_WINDOW", "300"))
MODEL_CACHE_VOLUME = os.environ.get("MODEL_CACHE_VOLUME", "ai-qr-generator-model-cache")

ROOT = Path(__file__).resolve().parent


def _load_requirements() -> list[str]:
    for requirements_path in (
        ROOT / "modal_requirements.txt",
        Path("/root/app/modal_requirements.txt"),
    ):
        if requirements_path.exists():
            return read_modal_requirements(requirements_path)
    raise FileNotFoundError("Could not locate modal_requirements.txt")


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(MODEL_CACHE_VOLUME, create_if_missing=True)
runtime_secret = modal.Secret.from_name("ai-qr-runtime")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(*_load_requirements())
    .add_local_dir(str(ROOT), remote_path="/root/app")
)


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    timeout=TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={"/root/app/models": volume},
    secrets=[modal.Secret.from_name("huggingface-token"), runtime_secret],
)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.concurrency import run_in_threadpool

    state: dict[str, Any] = {
        "backend": None,
        "import_error": None,
        "ready": False,
    }

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        os.chdir("/root/app")
        os.environ.setdefault("HF_HOME", "/root/app/models/huggingface")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/root/app/models/huggingface")
        os.environ.setdefault("TORCH_HOME", "/root/app/models/torch")
        if "/root/app" not in sys.path:
            sys.path.insert(0, "/root/app")
        try:
            qr_space_app = load_module_from_file("qr_space_entry", "/root/app/app.py")
            state["backend"] = qr_space_app
            state["ready"] = True
            await volume.commit.aio()
        except Exception as exc:  # pragma: no cover
            state["import_error"] = repr(exc)
            state["ready"] = False
        yield

    web_app = FastAPI(title=APP_NAME, lifespan=lifespan)

    @web_app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": state["ready"],
            "app_name": APP_NAME,
            "gpu": GPU,
            "scaledown_window": SCALEDOWN_WINDOW,
            "model_cache_volume": MODEL_CACHE_VOLUME,
            "import_error": state["import_error"],
            "analytics_product": os.environ.get("ANALYTICS_PRODUCT", ""),
            "url_shortener_api_url_present": bool(
                os.environ.get("URL_SHORTENER_API_URL")
            ),
            "url_shortener_source_app": os.environ.get("URL_SHORTENER_SOURCE_APP", ""),
            "url_shortener_api_key_present": bool(
                os.environ.get("URL_SHORTENER_API_KEY")
            ),
        }

    @web_app.post("/generate")
    async def generate(
        payload: GenerateRequest, raw_request: Request
    ) -> dict[str, Any]:
        if not state["ready"] or state["backend"] is None:
            raise HTTPException(
                status_code=503, detail=state["import_error"] or "Backend not ready"
            )

        actual_seed = resolve_request_seed(payload)
        prepared_request = payload.model_copy(
            update={"seed": actual_seed, "use_custom_seed": True}
        )

        def _run_generation() -> tuple[Any, str, dict[str, Any] | None]:
            backend = state["backend"]
            kwargs = build_generation_kwargs(
                prepared_request, runtime_request=raw_request
            )
            if prepared_request.mode == "artistic":
                generator = backend.generate_artistic_qr(**kwargs)
            else:
                generator = backend.generate_standard_qr(**kwargs)
            return consume_final_result(generator)

        started_at = time.perf_counter()
        try:
            image_obj, final_status, settings = await run_in_threadpool(_run_generation)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        elapsed = round(time.perf_counter() - started_at, 3)
        return build_response_payload(
            image_obj,
            final_status,
            payload,
            actual_seed=actual_seed,
            elapsed=elapsed,
            settings=settings,
        )

    return web_app
