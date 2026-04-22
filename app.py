import os
import queue
import sys
import threading
import time
import traceback
import hashlib
import uuid
from datetime import datetime, timezone
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import json
import random
import sys
import warnings
from typing import Any, Mapping, Sequence, Union

import gradio as gr
import numpy as np
import qrcode
import spaces
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
import kornia.color  # For RGB→HSV conversion in Stable Cascade filter

# ── Export helpers (PNG + embedded SVG download) ──────────────────────────────
import base64
import io
import re

from gradio.processing_utils import (
    get_upload_folder,
    save_bytes_to_cache,
    save_pil_to_cache,
)


ANALYTICS_ENABLED = os.getenv("ANALYTICS_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ANALYTICS_DEFAULT_OPT_IN = os.getenv(
    "ANALYTICS_DEFAULT_OPT_IN", "false"
).strip().lower() in {"1", "true", "yes", "on"}
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_GENERATION_TABLE = os.getenv(
    "SUPABASE_GENERATION_TABLE", "analytics_generation_events"
)
SUPABASE_DOWNLOAD_TABLE = os.getenv(
    "SUPABASE_DOWNLOAD_TABLE", "analytics_download_events"
)
SUPABASE_VALIDATION_TABLE = os.getenv(
    "SUPABASE_VALIDATION_TABLE", "analytics_validation_events"
)
POSTHOG_HOST = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com").rstrip("/")
POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY", "") or os.getenv(
    "POSTHOG_SECRET_KEY", ""
)
URL_SHORTENER_API_URL = os.getenv("URL_SHORTENER_API_URL", "").rstrip("/")
URL_SHORTENER_API_KEY = os.getenv("URL_SHORTENER_API_KEY", "")
URL_SHORTENER_SOURCE_APP = os.getenv("URL_SHORTENER_SOURCE_APP", "ai_qr_generator")
URL_SHORTENER_TIMEOUT_SECONDS = float(os.getenv("URL_SHORTENER_TIMEOUT_SECONDS", "10"))

URL_TRACKING_PARAM_PREFIXES = (
    "utm_",
    "mc_",
    "mkt_",
    "pk_",
)
URL_TRACKING_PARAM_NAMES = {
    "fbclid",
    "gclid",
    "dclid",
    "gbraid",
    "wbraid",
    "msclkid",
    "igshid",
    "mibextid",
    "s_cid",
    "si",
    "vero_conv",
    "vero_id",
    "wickedid",
    "yclid",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _request_headers(request: Any) -> Mapping[str, str]:
    headers = getattr(request, "headers", None)
    if isinstance(headers, Mapping):
        return headers
    return {}


def _detect_source(request: Any) -> str:
    if request is None:
        return "unknown"
    headers = _request_headers(request)
    accept = str(headers.get("accept", "")).lower()
    if "text/event-stream" in accept:
        return "mcp"
    url = getattr(request, "url", None)
    path = str(getattr(url, "path", ""))
    if "/gradio_api/mcp" in path:
        return "mcp"
    return "ui"


def _build_anon_id(request: Any, source: str, fallback: str = "anonymous") -> str:
    request_session = getattr(request, "session_hash", None)
    headers = _request_headers(request)
    raw = "|".join(
        [
            source or "unknown",
            str(request_session or ""),
            str(headers.get("x-forwarded-for", "")),
            str(headers.get("user-agent", "")),
            fallback,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _classify_error_bucket(status: str) -> str:
    lowered = (status or "").lower()
    if not lowered:
        return "unknown"
    if any(
        token in lowered for token in ("zero gpu", "quota", "rate limit", "capacity")
    ):
        return "infra_limited"
    if any(
        token in lowered
        for token in ("invalid json", "failed to parse", "please use the correct tab")
    ):
        return "invalid_request"
    return "model_failure"


def _normalize_error_message(status: str) -> str:
    return (status or "").strip()


def _truncate_error_message(message: str, limit: int = 300) -> str:
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _should_strip_tracking_param(name: str) -> bool:
    lowered = (name or "").strip().lower()
    if not lowered:
        return False
    if lowered in URL_TRACKING_PARAM_NAMES:
        return True
    return lowered.startswith(URL_TRACKING_PARAM_PREFIXES)


def _normalize_url_for_qr(text_input: str) -> Mapping[str, Any]:
    raw_input = str(text_input or "")
    stripped_input = raw_input.strip()
    result = {
        "original_input": raw_input,
        "normalized_url": stripped_input,
        "normalized_qr_text": stripped_input,
        "tracking_params_removed": 0,
        "chars_saved": max(0, len(raw_input) - len(stripped_input)),
        "changed": stripped_input != raw_input,
    }
    if not stripped_input:
        return result

    parse_target = stripped_input
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", parse_target):
        parse_target = f"https://{parse_target.lstrip('/')}"

    parsed = urllib_parse.urlsplit(parse_target)
    if parsed.scheme.lower() not in {"http", "https"}:
        return result

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return result

    port = parsed.port
    netloc = hostname
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        netloc = f"{auth}@{netloc}"
    if port and not (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    ):
        netloc = f"{netloc}:{port}"

    filtered_query = []
    tracking_params_removed = 0
    for key, value in urllib_parse.parse_qsl(parsed.query, keep_blank_values=True):
        if _should_strip_tracking_param(key):
            tracking_params_removed += 1
            continue
        filtered_query.append((key, value))

    normalized_path = parsed.path or ""
    if normalized_path == "/":
        normalized_path = ""

    normalized_url = urllib_parse.urlunsplit(
        (
            "https",
            netloc,
            normalized_path,
            urllib_parse.urlencode(filtered_query, doseq=True),
            parsed.fragment,
        )
    )
    normalized_qr_text = normalized_url.replace("https://", "", 1)
    chars_saved = max(0, len(raw_input) - len(normalized_url))

    return {
        "original_input": raw_input,
        "normalized_url": normalized_url,
        "normalized_qr_text": normalized_qr_text,
        "tracking_params_removed": tracking_params_removed,
        "chars_saved": chars_saved,
        "changed": normalized_url != raw_input,
    }


def _build_url_normalization_note(
    normalization: Mapping[str, Any] | None,
) -> str | None:
    if not normalization or not normalization.get("changed"):
        return None

    details = []
    tracking_params_removed = int(normalization.get("tracking_params_removed") or 0)
    chars_saved = int(normalization.get("chars_saved") or 0)
    if tracking_params_removed:
        suffix = "s" if tracking_params_removed != 1 else ""
        details.append(f"removed {tracking_params_removed} tracking param{suffix}")
    if chars_saved:
        suffix = "s" if chars_saved != 1 else ""
        details.append(f"saved {chars_saved} character{suffix}")
    if not details:
        details.append("normalized spacing and host casing")

    return (
        "URL normalized before QR generation: "
        + ", ".join(details)
        + f". Using {normalization.get('normalized_url', '')}"
    )


def _append_status_note(status: str | None, note: str | None) -> str | None:
    if not note:
        return status
    if not status:
        return note
    return f"{status}\n\n{note}"


def _build_shortener_note(shortener_result: Mapping[str, Any] | None) -> str | None:
    if not shortener_result:
        return None
    if shortener_result.get("applied"):
        short_url = str(shortener_result.get("short_url") or "").strip()
        expires_at = str(shortener_result.get("expires_at") or "").strip()
        suffix = ""
        if shortener_result.get("reactivated"):
            suffix = " Existing short link reactivated."
        elif shortener_result.get("existed"):
            suffix = " Existing short link reused."
        if expires_at:
            return (
                f"Temporary short link enabled: using {short_url}. "
                f"Expires after 7 days of inactivity and is currently active until {expires_at}.{suffix}"
            )
        return (
            f"Temporary short link enabled: using {short_url}. "
            f"Expires after 7 days of inactivity.{suffix}"
        )

    error_message = str(shortener_result.get("error") or "").strip()
    if error_message:
        return f"Temporary short link unavailable: {error_message} Using normalized URL instead."
    return None


def _maybe_shorten_url_for_qr(
    *,
    original_input: str,
    input_type: str,
    use_temporary_short_link: bool,
) -> Mapping[str, Any]:
    fallback_qr_text = _normalize_qr_text_for_validation(original_input, input_type)
    result = {
        "applied": False,
        "effective_qr_text": fallback_qr_text,
        "short_url": None,
        "expires_at": None,
        "existed": False,
        "reactivated": False,
        "error": None,
    }
    if input_type != "URL" or not _normalize_bool(use_temporary_short_link):
        return result
    if not URL_SHORTENER_API_URL or not URL_SHORTENER_API_KEY:
        return {
            **result,
            "error": "shortener is not configured on the server",
        }

    request_body = json.dumps({"url": str(original_input or "")}).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "X-API-Key": URL_SHORTENER_API_KEY,
        "X-Source-App": URL_SHORTENER_SOURCE_APP,
    }
    try:
        request_obj = urllib_request.Request(
            URL_SHORTENER_API_URL,
            data=request_body,
            headers=request_headers,
            method="POST",
        )
        with urllib_request.urlopen(
            request_obj, timeout=URL_SHORTENER_TIMEOUT_SECONDS
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        short_url = str(payload.get("short_url") or "").strip()
        if not short_url:
            raise ValueError("shortener returned an empty short URL")
        return {
            "applied": True,
            "effective_qr_text": _normalize_qr_text_for_validation(short_url, "URL"),
            "short_url": short_url,
            "expires_at": payload.get("expires_at"),
            "existed": bool(payload.get("existed")),
            "reactivated": bool(payload.get("reactivated")),
            "error": None,
        }
    except (
        urllib_error.URLError,
        urllib_error.HTTPError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return {
            **result,
            "error": str(exc),
        }


def _build_url_analytics_fields(
    text_input: str, settings: Mapping[str, Any]
) -> Mapping[str, Any]:
    input_type = str(settings.get("input_type") or "")
    effective_qr_text = str(settings.get("effective_qr_text") or text_input or "")
    if input_type != "URL":
        return {
            "original_qr_payload_length": None,
            "effective_qr_payload_length": None,
            "url_normalization_applied": False,
            "url_tracking_params_removed": 0,
            "url_chars_saved": 0,
        }

    return {
        "original_qr_payload_length": len(str(text_input or "")),
        "effective_qr_payload_length": len(effective_qr_text),
        "url_normalization_applied": bool(settings.get("url_normalization_applied")),
        "url_tracking_params_removed": int(
            settings.get("url_tracking_params_removed") or 0
        ),
        "url_chars_saved": int(settings.get("url_chars_saved") or 0),
    }


def _format_exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    exc_type = type(exc).__name__
    if not message:
        return exc_type
    if message == exc_type or message == f"'{exc_type}'":
        return f"{exc_type}: {repr(exc)}"
    return f"{exc_type}: {message}"


def _queue_background_work(fn, *args, **kwargs) -> None:
    thread = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    thread.start()


def _post_json(
    url: str, payload: Mapping[str, Any], headers: Mapping[str, str]
) -> None:
    req = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **dict(headers)},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=10):
            pass
    except urllib_error.URLError as exc:
        print(f"[analytics] remote write failed: {exc}")


def _write_supabase_row(table: str, row: Mapping[str, Any]) -> None:
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
        return
    _post_json(
        f"{SUPABASE_URL}/rest/v1/{table}",
        row,
        {
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Prefer": "return=minimal",
        },
    )


def _capture_posthog_event(event: str, properties: Mapping[str, Any]) -> None:
    if not POSTHOG_API_KEY:
        return
    distinct_id = str(
        properties.get("anonymous_id")
        or properties.get("generation_id")
        or uuid.uuid4()
    )
    _post_json(
        f"{POSTHOG_HOST}/capture/",
        {
            "api_key": POSTHOG_API_KEY,
            "event": event,
            "distinct_id": distinct_id,
            "properties": dict(properties),
        },
        {},
    )


def _emit_analytics_log(kind: str, payload: Mapping[str, Any]) -> None:
    print(f"[analytics] {kind} {json.dumps(payload, ensure_ascii=False, default=str)}")


def _record_generation_event(payload: Mapping[str, Any]) -> None:
    _emit_analytics_log("generation", payload)
    _queue_background_work(_write_supabase_row, SUPABASE_GENERATION_TABLE, payload)
    _queue_background_work(
        _capture_posthog_event,
        "generate_finished",
        {
            "product": "ai_qr_generator",
            "source": payload.get("source"),
            "pipeline": payload.get("pipeline"),
            "tool_name": payload.get("tool_name"),
            "analytics_opt_in": payload.get("analytics_opt_in"),
            "status": payload.get("status"),
            "error_bucket": payload.get("error_bucket"),
            "error_message_hash": payload.get("error_message_hash"),
            "generation_id": payload.get("generation_id"),
            "anonymous_id": payload.get("anonymous_id"),
            "original_qr_payload_length": payload.get("original_qr_payload_length"),
            "effective_qr_payload_length": payload.get("effective_qr_payload_length"),
            "url_normalization_applied": payload.get("url_normalization_applied"),
            "url_tracking_params_removed": payload.get("url_tracking_params_removed"),
            "url_chars_saved": payload.get("url_chars_saved"),
        },
    )


def _record_validation_event(payload: Mapping[str, Any]) -> None:
    _emit_analytics_log("validation", payload)
    _queue_background_work(_write_supabase_row, SUPABASE_VALIDATION_TABLE, payload)
    _queue_background_work(
        _capture_posthog_event,
        "validation_blocked",
        {
            "product": "ai_qr_generator",
            "source": payload.get("source"),
            "pipeline": payload.get("pipeline"),
            "tool_name": payload.get("tool_name"),
            "analytics_opt_in": payload.get("analytics_opt_in"),
            "error_bucket": payload.get("error_bucket"),
            "error_message_hash": payload.get("error_message_hash"),
            "generation_id": payload.get("generation_id"),
            "anonymous_id": payload.get("anonymous_id"),
            "original_qr_payload_length": payload.get("original_qr_payload_length"),
            "effective_qr_payload_length": payload.get("effective_qr_payload_length"),
            "url_normalization_applied": payload.get("url_normalization_applied"),
            "url_tracking_params_removed": payload.get("url_tracking_params_removed"),
            "url_chars_saved": payload.get("url_chars_saved"),
        },
    )


def _record_validation_event(payload: Mapping[str, Any]) -> None:
    _emit_analytics_log("validation", payload)
    _queue_background_work(_write_supabase_row, SUPABASE_VALIDATION_TABLE, payload)
    _queue_background_work(
        _capture_posthog_event,
        "validation_blocked",
        {
            "product": "ai_qr_generator",
            "source": payload.get("source"),
            "pipeline": payload.get("pipeline"),
            "tool_name": payload.get("tool_name"),
            "analytics_opt_in": payload.get("analytics_opt_in"),
            "error_bucket": payload.get("error_bucket"),
            "error_message_hash": payload.get("error_message_hash"),
            "generation_id": payload.get("generation_id"),
            "anonymous_id": payload.get("anonymous_id"),
        },
    )


def _record_download_event(payload: Mapping[str, Any]) -> None:
    _emit_analytics_log("download", payload)
    _queue_background_work(_write_supabase_row, SUPABASE_DOWNLOAD_TABLE, payload)
    _queue_background_work(
        _capture_posthog_event,
        "download_requested",
        {
            "product": "ai_qr_generator",
            "source": payload.get("source"),
            "pipeline": payload.get("pipeline"),
            "tool_name": payload.get("tool_name"),
            "analytics_opt_in": payload.get("analytics_opt_in"),
            "generation_id": payload.get("generation_id"),
            "anonymous_id": payload.get("anonymous_id"),
            "format": payload.get("format"),
        },
    )


def _build_generation_payload(
    *,
    generation_id: str,
    source: str,
    pipeline: str,
    tool_name: str,
    analytics_opt_in: bool,
    prompt: str,
    text_input: str,
    settings: Mapping[str, Any],
    status: str,
    request: Any,
) -> Mapping[str, Any]:
    error_message = _normalize_error_message(status if status != "success" else "")
    error_bucket = (
        "none" if status == "success" else _classify_error_bucket(error_message)
    )
    url_fields = _build_url_analytics_fields(text_input, settings)
    payload = {
        "generation_id": generation_id,
        "timestamp": _utc_now_iso(),
        "product": "ai_qr_generator",
        "source": source,
        "pipeline": pipeline,
        "tool_name": tool_name,
        "analytics_opt_in": analytics_opt_in,
        "status": "success" if status == "success" else "error",
        "error_bucket": error_bucket,
        "anonymous_id": _build_anon_id(request, source, fallback=generation_id),
        **url_fields,
    }
    if error_message:
        payload = {
            **payload,
            "error_message_excerpt": _truncate_error_message(error_message),
            "error_message_hash": hashlib.sha256(
                error_message.encode("utf-8")
            ).hexdigest()[:16],
        }
    if analytics_opt_in:
        payload = {
            **payload,
            "prompt_full": prompt,
            "qr_payload_full": text_input,
            "settings_full": dict(settings),
        }
    return payload


def _build_download_payload(
    *,
    generation_id: str,
    source: str,
    pipeline: str,
    tool_name: str,
    analytics_opt_in: bool,
    text_input: str,
    seed: int,
    fmt: str,
    request: Any,
) -> Mapping[str, Any]:
    payload = {
        "generation_id": generation_id or "",
        "timestamp": _utc_now_iso(),
        "product": "ai_qr_generator",
        "source": source,
        "pipeline": pipeline or "unknown",
        "tool_name": tool_name,
        "analytics_opt_in": analytics_opt_in,
        "format": fmt,
        "anonymous_id": _build_anon_id(
            request, source, fallback=f"{text_input}:{seed}:{fmt}"
        ),
    }
    if analytics_opt_in:
        payload = {
            **payload,
            "qr_payload_full": text_input,
            "seed": seed,
        }
    return payload


def _build_validation_payload(
    *,
    generation_id: str,
    source: str,
    pipeline: str,
    tool_name: str,
    analytics_opt_in: bool,
    prompt: str,
    text_input: str,
    settings: Mapping[str, Any],
    status: str,
    request: Any,
) -> Mapping[str, Any]:
    error_message = _normalize_error_message(status)
    url_fields = _build_url_analytics_fields(text_input, settings)
    payload = {
        "generation_id": generation_id,
        "timestamp": _utc_now_iso(),
        "product": "ai_qr_generator",
        "source": source,
        "pipeline": pipeline,
        "tool_name": tool_name,
        "analytics_opt_in": analytics_opt_in,
        "error_bucket": "validation_blocked",
        "anonymous_id": _build_anon_id(request, source, fallback=generation_id),
        "error_message_excerpt": _truncate_error_message(error_message),
        "error_message_hash": hashlib.sha256(error_message.encode("utf-8")).hexdigest()[
            :16
        ],
        **url_fields,
    }
    if analytics_opt_in:
        payload = {
            **payload,
            "prompt_full": prompt,
            "qr_payload_full": text_input,
            "settings_full": dict(settings),
        }
    return payload


def _make_qr_stem(text_input: str, seed: int) -> str:
    """Build a short human-readable filename stem from QR payload + seed."""
    slug = re.sub(r"[^a-z0-9]+", "-", text_input.lower()).strip("-")[:30] or "export"
    return f"ai-qr-{slug}-seed{seed}"


def _write_png(img: Image.Image, stem: str) -> str:
    """Save PNG into Gradio's cache dir. Returns path.

    Written into /tmp/gradio/ so delete_cache=(3600, 3600) sweeps it after
    1 hour — consistent with the generated image privacy policy in the README.
    """
    return save_pil_to_cache(
        img.convert("RGB"), get_upload_folder(), name=stem, format="png"
    )


def _write_svg(img: Image.Image, stem: str) -> str:
    """Save a PNG-embedded SVG into Gradio's cache dir. Returns path.

    The SVG wraps the raster image as a base64-encoded PNG inside a valid SVG
    container. Confirmed scannable and opens correctly in Figma, Illustrator,
    Safari, and Chrome.

    Written into /tmp/gradio/ so delete_cache=(3600, 3600) sweeps it after
    1 hour — consistent with the generated image privacy policy in the README.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    w, h = img.size
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f"<title>AI QR Code</title>"
        f'<image href="data:image/png;base64,{b64}" '
        f'x="0" y="0" width="{w}" height="{h}" '
        f'preserveAspectRatio="xMidYMid meet"/></svg>'
    )
    return save_bytes_to_cache(svg.encode("utf-8"), f"{stem}.svg", get_upload_folder())


def _download_png(
    image,
    text_input,
    seed,
    analytics_opt_in: bool = False,
    pipeline: str = "unknown",
    generation_id: str = "",
    request: Union[gr.Request, None] = None,
):
    """Called by DownloadButton at click time — generates PNG on demand."""
    if image is None:
        return None
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    source = _detect_source(request)
    if ANALYTICS_ENABLED:
        _record_download_event(
            _build_download_payload(
                generation_id=generation_id,
                source=source,
                pipeline=pipeline,
                tool_name=f"download_png_{pipeline}",
                analytics_opt_in=_normalize_bool(analytics_opt_in),
                text_input=str(text_input),
                seed=_safe_int(seed),
                fmt="png",
                request=request,
            )
        )
    return _write_png(img, _make_qr_stem(str(text_input), int(seed)))


def _download_svg(
    image,
    text_input,
    seed,
    analytics_opt_in: bool = False,
    pipeline: str = "unknown",
    generation_id: str = "",
    request: Union[gr.Request, None] = None,
):
    """Called by DownloadButton at click time — generates embedded SVG on demand."""
    if image is None:
        return None
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    source = _detect_source(request)
    if ANALYTICS_ENABLED:
        _record_download_event(
            _build_download_payload(
                generation_id=generation_id,
                source=source,
                pipeline=pipeline,
                tool_name=f"download_svg_{pipeline}",
                analytics_opt_in=_normalize_bool(analytics_opt_in),
                text_input=str(text_input),
                seed=_safe_int(seed),
                fmt="svg",
                request=request,
            )
        )
    return _write_svg(img, _make_qr_stem(str(text_input), int(seed)))


# ─────────────────────────────────────────────────────────────────────────────

# ComfyUI imports (after HF hub downloads)
from comfy import model_management
from comfy.cli_args import args
from comfy_extras.nodes_freelunch import FreeU_V2

# Suppress torchsde floating-point precision warnings (cosmetic only, no functional impact)
warnings.filterwarnings("ignore", message="Should have tb<=t1 but got")

hf_hub_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    filename="v1-5-pruned-emaonly.ckpt",
    local_dir="models/checkpoints",
)
hf_hub_download(
    repo_id="Lykon/DreamShaper",
    filename="DreamShaper_3.32_baked_vae_clip_fix_half.safetensors",
    local_dir="models/checkpoints",
)
hf_hub_download(
    repo_id="Lykon/DreamShaper",
    filename="DreamShaper_6.31_BakedVae_pruned.safetensors",
    local_dir="models/checkpoints",
)
hf_hub_download(
    repo_id="latentcat/latentcat-controlnet",
    filename="models/control_v1p_sd15_brightness.safetensors",
    local_dir="models/controlnet",
)
hf_hub_download(
    repo_id="comfyanonymous/ControlNet-v1-1_fp16_safetensors",
    filename="control_v11f1e_sd15_tile_fp16.safetensors",
    local_dir="models/controlnet",
)
hf_hub_download(
    repo_id="Lykon/dreamshaper-7",
    filename="vae/diffusion_pytorch_model.fp16.safetensors",
    local_dir="models",
)
hf_hub_download(
    repo_id="stabilityai/sd-vae-ft-mse-original",
    filename="vae-ft-mse-840000-ema-pruned.safetensors",
    local_dir="models/vae",
)
hf_hub_download(
    repo_id="lllyasviel/Annotators",
    filename="RealESRGAN_x4plus.pth",
    local_dir="models/upscale_models",
)


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        try:
            from utils.extra_config import load_extra_path_config
        except (ImportError, ModuleNotFoundError) as e:
            print(
                f"Could not import load_extra_path_config from utils.extra_config either: {e}"
            )
            print(
                "Skipping extra model paths configuration (this is OK for Gradio hot reload)."
            )
            return

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


# Only run initialization on first load, not during Gradio hot reload
if not hasattr(__builtins__, "_comfy_initialized"):
    __builtins__._comfy_initialized = True
    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
else:
    print("Skipping ComfyUI initialization (Gradio hot reload detected)")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio

    import execution
    import server
    from nodes import init_extra_nodes

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS  # noqa: E402

# Initialize common nodes
checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
    ckpt_name="DreamShaper_3.32_baked_vae_clip_fix_half.safetensors"
)
checkpointloadersimple_artistic = checkpointloadersimple.load_checkpoint(
    ckpt_name="DreamShaper_6.31_BakedVae_pruned.safetensors"
)

emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
vaedecodetiled = NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()

import_custom_nodes()
comfy_qr_by_module_size = NODE_CLASS_MAPPINGS["comfy-qr-by-module-size"]()
tilepreprocessor = NODE_CLASS_MAPPINGS["TilePreprocessor"]()

# Load additional nodes for artistic pipeline (upscale model loaded lazily when needed)
imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()
latentupscaleby = NODE_CLASS_MAPPINGS["LatentUpscaleBy"]()

# MPS (Apple Silicon) comprehensive workaround for black QR code bug
# Issue: PyTorch 2.6+ FP16 handling on MPS causes black images in samplers
# Additional issue: MPS tensor operations can produce NaN/inf values (PyTorch bug #84364)
# Solution: Monkey-patch dtype functions to force fp32, enable MPS fallback
# References: https://civitai.com/articles/11106, https://github.com/pytorch/pytorch/issues/84364

# Lazy upscale model loading - only load when needed
# This is safe for ZeroGPU since upscaling happens inside @spaces.GPU function
_upscale_model_cache = None


def get_upscale_model():
    """Load upscale model on-demand and cache it within GPU context"""
    global _upscale_model_cache
    if _upscale_model_cache is None:
        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        _upscale_model_cache = upscalemodelloader.load_model(
            model_name="RealESRGAN_x4plus.pth"
        )
    return _upscale_model_cache


def calculate_vae_tile_size(image_size):
    """
    Calculate optimal VAE tile size based on image dimensions.

    Args:
        image_size: Width/height of square image in pixels

    Returns:
        tuple: (tile_size, overlap) or (None, None) for no tiling
    """
    # No tiling for small images (fits in memory easily)
    if image_size <= 512:
        return None, None

    # Medium images: 512px tiles
    elif image_size <= 1024:
        return 512, 64

    # Large images: 768px tiles (reduces tile count)
    elif image_size <= 2048:
        return 768, 96

    # XL images: 1024px tiles
    else:
        return 1024, 128


def log_progress(message, gr_progress=None, progress_value=None):
    """Helper to log progress to both console and Gradio (simple stage-based updates)"""
    print(f"{message}", flush=True)
    if gr_progress and progress_value is not None:
        gr_progress(progress_value, desc=message)


def _normalize_qr_text_for_validation(qr_text: str, input_type: str) -> str:
    if input_type == "URL":
        return str(_normalize_url_for_qr(qr_text).get("normalized_qr_text", "")).strip()
    return qr_text


def _compute_qr_dimensions(
    *,
    qr_text: str,
    input_type: str,
    border_size: int,
    error_correction: str,
    module_size: int,
) -> int:
    normalized_qr_text = _normalize_qr_text_for_validation(qr_text, input_type)
    qr_protocol = "None" if input_type == "Plain Text" else "Https"
    if qr_protocol == "Https":
        full_text = f"https://{normalized_qr_text}"
    elif qr_protocol == "Http":
        full_text = f"http://{normalized_qr_text}"
    else:
        full_text = normalized_qr_text

    error_level = {
        "Low (7%)": qrcode.constants.ERROR_CORRECT_L,
        "Medium (15%)": qrcode.constants.ERROR_CORRECT_M,
        "Quartile (25%)": qrcode.constants.ERROR_CORRECT_Q,
        "High (30%)": qrcode.constants.ERROR_CORRECT_H,
        "Low": qrcode.constants.ERROR_CORRECT_L,
        "Medium": qrcode.constants.ERROR_CORRECT_M,
        "Quartile": qrcode.constants.ERROR_CORRECT_Q,
        "High": qrcode.constants.ERROR_CORRECT_H,
    }[error_correction]

    qr = qrcode.QRCode(
        error_correction=error_level,
        box_size=module_size,
        border=border_size,
    )
    qr.add_data(full_text)
    qr.make(fit=True)

    return len(qr.get_matrix()) * module_size


def _validate_qr_dimensions(
    *,
    qr_text: str,
    input_type: str,
    image_size: int,
    border_size: int,
    error_correction: str,
    module_size: int,
) -> None:
    size = _compute_qr_dimensions(
        qr_text=qr_text,
        input_type=input_type,
        border_size=border_size,
        error_correction=error_correction,
        module_size=module_size,
    )
    if size > image_size:
        raise RuntimeError(
            f"Error generating QR code: QR dimensions of {size} exceed max size of {image_size}.\n"
            "Try with a shorter text, increase the image size, or decrease the border size, module size, and error correction level under Change Settings Manually."
        )


def _recommended_image_size(required_size: int, current_size: int) -> int | None:
    for candidate in range(512, 1025, 64):
        if candidate >= required_size:
            return candidate
    return None


def _matches_example_settings(
    qr_text: str,
    input_type: str,
    image_size: int,
    border_size: int,
    error_correction: str,
    module_size: int,
    examples,
) -> bool:
    for example in examples:
        if isinstance(example, dict):
            if (
                example["text_input"] == qr_text
                and example["input_type"] == input_type
                and example["image_size"] == image_size
                and example["border_size"] == border_size
                and example["error_correction"] == error_correction
                and example["module_size"] == module_size
            ):
                return True
        else:
            if (
                example[1] == qr_text
                and example[2] == input_type
                and example[3] == image_size
                and example[4] == border_size
                and example[5] == error_correction
                and example[6] == module_size
            ):
                return True
    return False


def _get_artistic_validation_state(
    qr_text: str,
    input_type: str,
    image_size: int,
    border_size: int,
    error_correction: str,
    module_size: int,
):
    normalized_qr_text = _normalize_qr_text_for_validation(qr_text, input_type)
    suppress_reduce = _matches_example_settings(
        qr_text,
        input_type,
        image_size,
        border_size,
        error_correction,
        module_size,
        ARTISTIC_EXAMPLES,
    )
    if not normalized_qr_text.strip():
        return (
            gr.update(value="", visible=False),
            gr.update(interactive=True),
            gr.update(visible=False),
            None,
        )

    try:
        size = _compute_qr_dimensions(
            qr_text=qr_text,
            input_type=input_type,
            border_size=border_size,
            error_correction=error_correction,
            module_size=module_size,
        )
    except Exception:
        return (
            gr.update(value="", visible=False),
            gr.update(interactive=True),
            gr.update(visible=False),
            None,
        )

    if size > image_size:
        suggested_size = _recommended_image_size(size, image_size)
        button_update = (
            gr.update(
                value=f"Increase image size to {suggested_size}",
                visible=True,
            )
            if suggested_size is not None
            else gr.update(visible=False)
        )
        return (
            gr.update(
                value="This QR code does not fit the selected image size. Increase image size or reduce module size, border size, or error correction.",
                visible=True,
            ),
            gr.update(interactive=False),
            button_update,
            suggested_size,
        )

    suggested_size = _recommended_image_size(size, image_size)
    if (
        suggested_size is not None
        and suggested_size < image_size
        and not suppress_reduce
    ):
        return (
            gr.update(value="", visible=False),
            gr.update(interactive=True),
            gr.update(
                value=f"Reduce image size to {suggested_size}",
                visible=True,
            ),
            suggested_size,
        )

    if (
        input_type == "URL"
        and len(normalized_qr_text) >= 38
        and image_size <= 704
        and module_size >= 14
        and border_size >= 6
        and error_correction in {"Medium (15%)", "Quartile (25%)", "High (30%)"}
    ):
        return (
            gr.update(
                value="This link may be too long for the current artistic settings. Short URLs work better, or try image size above 704.",
                visible=True,
            ),
            gr.update(interactive=True),
            (
                gr.update(
                    value=f"Reduce image size to {suggested_size}",
                    visible=True,
                )
                if suggested_size is not None
                and suggested_size < image_size
                and not suppress_reduce
                else gr.update(visible=False)
            ),
            suggested_size
            if suggested_size is not None
            and suggested_size < image_size
            and not suppress_reduce
            else None,
        )

    return (
        gr.update(value="", visible=False),
        gr.update(interactive=True),
        (
            gr.update(
                value=f"Reduce image size to {suggested_size}",
                visible=True,
            )
            if suggested_size is not None
            and suggested_size < image_size
            and not suppress_reduce
            else gr.update(visible=False)
        ),
        suggested_size
        if suggested_size is not None
        and suggested_size < image_size
        and not suppress_reduce
        else None,
    )


def _get_standard_validation_state(
    qr_text: str,
    input_type: str,
    image_size: int,
    border_size: int,
    error_correction: str,
    module_size: int,
):
    normalized_qr_text = _normalize_qr_text_for_validation(qr_text, input_type)
    suppress_reduce = _matches_example_settings(
        qr_text,
        input_type,
        image_size,
        border_size,
        error_correction,
        module_size,
        STANDARD_EXAMPLES,
    )
    if not normalized_qr_text.strip():
        return (
            gr.update(value="", visible=False),
            gr.update(interactive=True),
            gr.update(visible=False),
            None,
        )

    try:
        size = _compute_qr_dimensions(
            qr_text=qr_text,
            input_type=input_type,
            border_size=border_size,
            error_correction=error_correction,
            module_size=module_size,
        )
    except Exception:
        return (
            gr.update(value="", visible=False),
            gr.update(interactive=True),
            gr.update(visible=False),
            None,
        )

    if size > image_size:
        suggested_size = _recommended_image_size(size, image_size)
        button_update = (
            gr.update(
                value=f"Increase image size to {suggested_size}",
                visible=True,
            )
            if suggested_size is not None
            else gr.update(visible=False)
        )
        return (
            gr.update(
                value="This QR code does not fit the selected image size. Increase image size or reduce module size, border size, or error correction.",
                visible=True,
            ),
            gr.update(interactive=False),
            button_update,
            suggested_size,
        )

    suggested_size = _recommended_image_size(size, image_size)
    if (
        suggested_size is not None
        and suggested_size < image_size
        and not suppress_reduce
    ):
        return (
            gr.update(value="", visible=False),
            gr.update(interactive=True),
            gr.update(
                value=f"Reduce image size to {suggested_size}",
                visible=True,
            ),
            suggested_size,
        )

    if (
        input_type == "URL"
        and len(normalized_qr_text) >= 38
        and image_size <= 512
        and module_size >= 12
        and border_size >= 4
        and error_correction in {"Medium (15%)", "Quartile (25%)", "High (30%)"}
    ):
        return (
            gr.update(
                value="This link may be too long for the current standard settings. Short URLs work better, or try image size above 512.",
                visible=True,
            ),
            gr.update(interactive=True),
            (
                gr.update(
                    value=f"Reduce image size to {suggested_size}",
                    visible=True,
                )
                if suggested_size is not None
                and suggested_size < image_size
                and not suppress_reduce
                else gr.update(visible=False)
            ),
            suggested_size
            if suggested_size is not None
            and suggested_size < image_size
            and not suppress_reduce
            else None,
        )

    return (
        gr.update(value="", visible=False),
        gr.update(interactive=True),
        (
            gr.update(
                value=f"Reduce image size to {suggested_size}",
                visible=True,
            )
            if suggested_size is not None
            and suggested_size < image_size
            and not suppress_reduce
            else gr.update(visible=False)
        ),
        suggested_size
        if suggested_size is not None
        and suggested_size < image_size
        and not suppress_reduce
        else None,
    )


def _apply_recommended_image_size(recommended_size: int | None):
    if recommended_size is None:
        return gr.update()
    return gr.update(value=recommended_size)


# Device-specific optimizations
# Note: On ZeroGPU, torch.cuda.is_available() is False at module load time
# CUDA only becomes available inside @spaces.GPU decorated functions
# So we only check for MPS (local development) and apply those workarounds
if torch.backends.mps.is_available():
    # MPS device (Apple Silicon) - force fp32 to avoid black image bug
    print(f"MPS device detected (PyTorch {torch.__version__})")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
        "1"  # Enable MPS fallback for unsupported ops
    )

    # Store original dtype functions
    _original_unet_dtype = model_management.unet_dtype
    _original_vae_dtype = model_management.vae_dtype
    _original_text_encoder_dtype = model_management.text_encoder_dtype

    # Monkey-patch dtype functions to force fp32 for MPS
    def mps_safe_unet_dtype(device=None, *args_inner, **kwargs):
        if device is not None and model_management.is_device_mps(device):
            return torch.float32
        if model_management.mps_mode():
            return torch.float32
        return _original_unet_dtype(device, *args_inner, **kwargs)

    def mps_safe_vae_dtype(device=None, *args_inner, **kwargs):
        if device is not None and model_management.is_device_mps(device):
            return torch.float32
        if model_management.mps_mode():
            return torch.float32
        return _original_vae_dtype(device, *args_inner, **kwargs)

    def mps_safe_text_encoder_dtype(device=None, *args_inner, **kwargs):
        if device is not None and model_management.is_device_mps(device):
            return torch.float32
        if model_management.mps_mode():
            return torch.float32
        return _original_text_encoder_dtype(device, *args_inner, **kwargs)

    # Replace functions in model_management module
    model_management.unet_dtype = mps_safe_unet_dtype
    model_management.vae_dtype = mps_safe_vae_dtype
    model_management.text_encoder_dtype = mps_safe_text_encoder_dtype

    # Set args for additional stability
    args.force_fp32 = True
    args.fp32_vae = True
    args.fp32_unet = True
    args.force_upcast_attention = True

    # Performance settings: Tune these for speed vs stability
    # Try uncommenting these one at a time for better speed:
    args.lowvram = False  # Set to False for FASTER (try this first!)
    args.use_split_cross_attention = (
        False  # Set to False for even FASTER (might cause black images)
    )

    lowvram_status = "enabled" if args.lowvram else "disabled (faster)"
    split_attn_status = (
        "enabled" if args.use_split_cross_attention else "disabled (faster)"
    )
    print("  ✓ Enabled global fp32 dtype enforcement (monkey-patched)")
    print("  ✓ Enabled MPS fallback mode")
    print(f"  ✓ lowvram: {lowvram_status}, split-cross-attention: {split_attn_status}")
else:
    # Not MPS - likely ZeroGPU or other CUDA environment
    # CUDA optimizations (bfloat16) are handled automatically by ComfyUI's model_management
    print(f"PyTorch {torch.__version__} loaded")
    print("  ℹ️  CUDA optimizations will be applied when GPU becomes available")

# Add all the models that load a safetensors file
model_loaders = [checkpointloadersimple_4, checkpointloadersimple_artistic]

# Check which models are valid and how to best load them
valid_models = [
    getattr(loader[0], "patcher", loader[0])
    for loader in model_loaders
    if not isinstance(loader[0], dict)
    and not isinstance(getattr(loader[0], "patcher", None), dict)
]

# Note: Commenting out pre-loading to GPU for ZeroGPU compatibility
# On ZeroGPU, CUDA is not available until inside @spaces.GPU decorator
# Models will be automatically loaded to GPU when first used
# model_management.load_models_gpu(valid_models)


# Apply torch.compile to diffusion models for 1.5-1.7× speedup
# Used as fallback alongside AOT compilation for dynamic sizes
# Compilation happens once at startup (30-60s), then cached for fast inference
def _apply_torch_compile_optimizations():
    """Apply torch.compile to both pipeline models using ComfyUI's infrastructure"""
    try:
        from comfy_api.torch_helpers.torch_compile import set_torch_compile_wrapper

        print("\n🔧 Applying torch.compile optimizations...")

        # Increase cache limit to handle batch size variations (CFG uses batch 1 and 2)
        import torch._dynamo.config

        torch._dynamo.config.cache_size_limit = 64  # Allow more cached graphs

        # Compile standard pipeline model (DreamShaper 3.32)
        standard_model = get_value_at_index(checkpointloadersimple_4, 0)
        set_torch_compile_wrapper(
            model=standard_model,
            backend="inductor",
            mode="max-autotune",  # Maximum runtime speed (longer compile time is OK during warmup)
            fullgraph=False,  # Allow SAG to capture attention maps (disabled in SAG code)
            dynamic=True,  # Handle variable batch sizes during CFG without recompiling
            keys=["diffusion_model"],  # Compile UNet only
        )
        print("  ✓ Compiled standard pipeline diffusion model")

        # Compile artistic pipeline model (DreamShaper 6.31)
        artistic_model = get_value_at_index(checkpointloadersimple_artistic, 0)
        set_torch_compile_wrapper(
            model=artistic_model,
            backend="inductor",
            mode="max-autotune",  # Maximum runtime speed (longer compile time is OK during warmup)
            fullgraph=False,  # Allow SAG to capture attention maps (disabled in SAG code)
            dynamic=True,  # Handle variable batch sizes during CFG without recompiling
            keys=["diffusion_model"],  # Compile UNet only
        )
        print("  ✓ Compiled artistic pipeline diffusion model")
        print("✅ torch.compile optimizations applied successfully!\n")

    except Exception as e:
        print(f"⚠️  torch.compile optimization failed: {e}")
        print("   Continuing without compilation (slower but functional)\n")


# AOT Compilation with ZeroGPU for faster cold starts
# Runs once at startup to pre-compile models
# Falls back to torch.compile with warmup inference if AOTI unavailable
@spaces.GPU(duration=1500)  # Maximum allowed during startup
def compile_models_with_aoti():
    """
    Pre-compile diffusion models using AOT compilation.
    If AOTI fails, falls back to torch.compile with warmup inference.
    Uses the full 1500s GPU allocation to ensure models are compiled.
    """
    print("\n🔧 Starting model compilation warmup...")

    # Test parameters for warmup inference
    TEST_PROMPT = "a beautiful landscape with mountains"
    TEST_TEXT = "test.com"
    TEST_SEED = 12345

    try:
        import torch.export
        from spaces import aoti_apply, aoti_capture, aoti_compile

        print("   Attempting AOT compilation...\n")

        # ============================================================
        # 1. Compile Standard Pipeline @ 512px
        # ============================================================
        print("📦 [1/2] AOT compiling standard pipeline (512px)...")
        standard_model = get_value_at_index(checkpointloadersimple_4, 0)

        # Capture example run
        with aoti_capture(standard_model.model.diffusion_model) as call_standard:
            list(
                _pipeline_standard(
                    prompt=TEST_PROMPT,
                    negative_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
                    qr_text=TEST_TEXT,
                    input_type="URL",
                    image_size=512,
                    border_size=4,
                    error_correction="Medium (15%)",
                    module_size=12,
                    module_drawer="Square",
                    seed=TEST_SEED,
                    enable_upscale=False,
                    enable_animation=False,
                    controlnet_strength_first=1.5,
                    controlnet_strength_final=0.9,
                )
            )

        # Export and compile
        exported_standard = torch.export.export(
            standard_model.model.diffusion_model,
            args=call_standard.args,
            kwargs=call_standard.kwargs,
        )
        compiled_standard = aoti_compile(exported_standard)
        aoti_apply(compiled_standard, standard_model.model.diffusion_model)
        print("   ✓ Standard pipeline compiled")

        # ============================================================
        # 2. Compile Artistic Pipeline @ 640px
        # ============================================================
        print("📦 [2/2] AOT compiling artistic pipeline (640px)...")
        artistic_model = get_value_at_index(checkpointloadersimple_artistic, 0)

        # Capture example run
        with aoti_capture(artistic_model.model.diffusion_model) as call_artistic:
            list(
                _pipeline_artistic(
                    prompt=TEST_PROMPT,
                    negative_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
                    qr_text=TEST_TEXT,
                    input_type="URL",
                    image_size=640,
                    border_size=4,
                    error_correction="Medium (15%)",
                    module_size=12,
                    module_drawer="Square",
                    seed=TEST_SEED,
                    enable_upscale=False,
                    enable_animation=False,
                    controlnet_strength_first=1.5,
                    controlnet_strength_final=0.9,
                    freeu_b1=1.3,
                    freeu_b2=1.4,
                    freeu_s1=0.9,
                    freeu_s2=0.2,
                    enable_sag=True,
                    sag_scale=0.75,
                    sag_blur_sigma=2.0,
                )
            )

        # Export and compile
        exported_artistic = torch.export.export(
            artistic_model.model.diffusion_model,
            args=call_artistic.args,
            kwargs=call_artistic.kwargs,
        )
        compiled_artistic = aoti_compile(exported_artistic)
        aoti_apply(compiled_artistic, artistic_model.model.diffusion_model)
        print("   ✓ Artistic pipeline compiled")

        print("\n✅ AOT compilation complete! Models ready for fast inference.\n")
        return True

    except (ImportError, Exception) as e:
        error_type = "not available" if isinstance(e, ImportError) else f"failed: {e}"
        print(f"\n⚠️  AOT compilation {error_type}")
        print("   Falling back to torch.compile with warmup inference...\n")

        # Apply torch.compile optimizations
        _apply_torch_compile_optimizations()

        # Run warmup inference to trigger torch.compile compilation
        print(
            "🔥 Running warmup inference to compile models (this takes 2-3 minutes)..."
        )

        try:
            # Warmup standard pipeline @ 512px
            print("   [1/2] Warming up standard pipeline...")
            list(
                _pipeline_standard(
                    prompt=TEST_PROMPT,
                    negative_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
                    qr_text=TEST_TEXT,
                    input_type="URL",
                    image_size=512,
                    border_size=4,
                    error_correction="Medium (15%)",
                    module_size=12,
                    module_drawer="Square",
                    seed=TEST_SEED,
                    enable_upscale=False,
                    enable_animation=False,
                    controlnet_strength_first=1.5,
                    controlnet_strength_final=0.9,
                )
            )
            print("   ✓ Standard pipeline compiled")

            # Warmup artistic pipeline @ 640px
            print("   [2/2] Warming up artistic pipeline...")
            list(
                _pipeline_artistic(
                    prompt=TEST_PROMPT,
                    negative_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
                    qr_text=TEST_TEXT,
                    input_type="URL",
                    image_size=640,
                    border_size=4,
                    error_correction="Medium (15%)",
                    module_size=12,
                    module_drawer="Square",
                    seed=TEST_SEED,
                    enable_upscale=False,
                    enable_animation=False,
                    controlnet_strength_first=1.5,
                    controlnet_strength_final=0.9,
                    freeu_b1=1.3,
                    freeu_b2=1.4,
                    freeu_s1=0.9,
                    freeu_s2=0.2,
                    enable_sag=True,
                    sag_scale=0.75,
                    sag_blur_sigma=2.0,
                )
            )
            print("   ✓ Artistic pipeline compiled")

            print(
                "\n✅ torch.compile warmup complete! Models ready for fast inference.\n"
            )
            return True

        except Exception as warmup_error:
            print(f"\n⚠️  Warmup inference failed: {warmup_error}")
            print("   Models will compile on first real inference (slower first run)\n")
            return False


def get_dynamic_duration(*args, **kwargs):
    """
    Calculate GPU duration based on benchmarks with 32-37.5% safety margin (+10% buffer).
    Max duration capped at 120s (unauthenticated user limit).

    Benchmarks (actual measured times):
    Standard: 512+anim=10s, 512-anim=7s, 832+anim=20s, 1024=40s
    Artistic: 640+anim=23s, 832+anim=45s, 832+anim+upscale=57s, 1024+anim+upscale=124s
    """
    # Debug logging
    print(
        f"[GPU DURATION DEBUG] Called with args length={len(args)}, kwargs keys={list(kwargs.keys()) if kwargs else 'None'}"
    )

    # Extract parameters from correct source (args vs kwargs)
    # Function signature: generate_qr_code_unified(prompt, negative_prompt, text_input, input_type, image_size, ...)
    # ZeroGPU passes some as positional args, some as kwargs
    image_size = args[4] if len(args) > 4 else kwargs.get("image_size", 512)
    pipeline = kwargs.get("pipeline", "standard")
    enable_animation = kwargs.get("enable_animation", True)
    enable_upscale = kwargs.get("enable_upscale", False)

    print(
        f"[GPU DURATION DEBUG] Extracted: pipeline={pipeline}, image_size={image_size}, enable_animation={enable_animation}, enable_upscale={enable_upscale}"
    )

    if pipeline == "standard":
        # Standard pipeline benchmarks (with 32% safety margin = 20% + 10% buffer)
        if image_size <= 512:
            duration = 13 if enable_animation else 10
        elif image_size <= 640:
            duration = 20 if enable_animation else 14
        elif image_size <= 768:
            duration = 24 if enable_animation else 18
        elif image_size <= 832:
            duration = 26 if enable_animation else 19
        else:  # 1024
            duration = 53 if enable_animation else 37
    else:  # artistic
        # Artistic pipeline benchmarks (with 37.5% safety margin = 25% + 10% buffer)
        if image_size <= 512:
            # Extrapolated from 640 benchmark (~18s base)
            duration = 24 if not enable_upscale else 42
        elif image_size <= 640:
            duration = 31 if not enable_upscale else 55
        elif image_size <= 768:
            # Interpolated between 640 and 832 (~35s base)
            duration = 48 if not enable_upscale else 72
        elif image_size <= 832:
            duration = 62 if not enable_upscale else 79
        else:  # 1024
            # Extrapolated from 832 (~75s base)
            duration = 103 if not enable_upscale else 132  # Worst case measured at 124s

    # Cap at 120 seconds (unauthenticated user limit)
    final_duration = min(duration, 120)
    print(
        f"[GPU DURATION DEBUG] Calculated duration={duration}, final_duration={final_duration}"
    )
    return final_duration


@spaces.GPU(duration=get_dynamic_duration)  # Dynamic duration based on settings
def generate_qr_code_unified(
    prompt: str,
    negative_prompt: str = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
    text_input: str = "",
    input_type: str = "URL",
    image_size: int = 512,
    border_size: int = 4,
    error_correction: str = "Medium (15%)",
    module_size: int = 12,
    module_drawer: str = "Square",
    use_custom_seed: bool = False,
    seed: int = 0,
    pipeline: str = "standard",
    enable_upscale: bool = False,
    freeu_b1: float = 1.4,
    freeu_b2: float = 1.3,
    freeu_s1: float = 0.0,
    freeu_s2: float = 1.3,
    enable_sag: bool = True,
    sag_scale: float = 0.5,
    sag_blur_sigma: float = 1.5,
    controlnet_strength_first: float = 0.45,
    controlnet_strength_final: float = 0.7,
    controlnet_strength_standard_first: float = 0.45,
    controlnet_strength_standard_final: float = 1.0,
    enable_color_quantization: bool = False,
    num_colors: int = 4,
    color_1: str = "#000000",
    color_2: str = "#FFFFFF",
    color_3: str = "#FF0000",
    color_4: str = "#00FF00",
    apply_gradient_filter: bool = False,
    gradient_strength: float = 0.3,
    variation_steps: int = 5,
    enable_animation: bool = True,
    enable_cascade_filter: bool = False,
    cascade_blur_kernel: int = 15,
    cascade_threshold_ratio: float = 0.33,
    enable_detail_sharpening: bool = False,
    sharpening_radius: float = 2.0,
    sharpening_amount: float = 1.5,
    sharpening_threshold: int = 0,
    customize_tile_preprocessing: bool = False,
    tile_pyrup_iters: int = 3,
    gr_progress=None,
):
    # Track actual GPU time spent
    start_time = time.time()
    print(
        f"[GPU TIMING] Started generation: pipeline={pipeline}, image_size={image_size}, animation={enable_animation}, upscale={enable_upscale}"
    )

    # URL inputs are normalized on CPU before any GPU work begins.
    qr_text = _normalize_qr_text_for_validation(text_input, input_type)

    # Use custom seed or random
    actual_seed = seed if use_custom_seed else random.randint(1, 2**32 - 1)

    _validate_qr_dimensions(
        qr_text=qr_text,
        input_type=input_type,
        image_size=image_size,
        border_size=border_size,
        error_correction=error_correction,
        module_size=module_size,
    )

    with torch.no_grad():
        try:
            if pipeline == "standard":
                for result in _pipeline_standard(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    qr_text=qr_text,
                    input_type=input_type,
                    image_size=image_size,
                    border_size=border_size,
                    error_correction=error_correction,
                    module_size=module_size,
                    module_drawer=module_drawer,
                    seed=actual_seed,
                    enable_upscale=enable_upscale,
                    controlnet_strength_first=controlnet_strength_standard_first,
                    controlnet_strength_final=controlnet_strength_standard_final,
                    enable_color_quantization=enable_color_quantization,
                    num_colors=num_colors,
                    color_1=color_1,
                    color_2=color_2,
                    color_3=color_3,
                    color_4=color_4,
                    apply_gradient_filter=apply_gradient_filter,
                    gradient_strength=gradient_strength,
                    variation_steps=variation_steps,
                    enable_animation=enable_animation,
                    gr_progress=gr_progress,
                ):
                    yield result
            else:  # artistic
                for result in _pipeline_artistic(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    qr_text=qr_text,
                    input_type=input_type,
                    image_size=image_size,
                    border_size=border_size,
                    error_correction=error_correction,
                    module_size=module_size,
                    module_drawer=module_drawer,
                    seed=actual_seed,
                    enable_upscale=enable_upscale,
                    freeu_b1=freeu_b1,
                    freeu_b2=freeu_b2,
                    freeu_s1=freeu_s1,
                    freeu_s2=freeu_s2,
                    enable_sag=enable_sag,
                    sag_scale=sag_scale,
                    sag_blur_sigma=sag_blur_sigma,
                    controlnet_strength_first=controlnet_strength_first,
                    controlnet_strength_final=controlnet_strength_final,
                    enable_color_quantization=enable_color_quantization,
                    num_colors=num_colors,
                    color_1=color_1,
                    color_2=color_2,
                    color_3=color_3,
                    color_4=color_4,
                    apply_gradient_filter=apply_gradient_filter,
                    gradient_strength=gradient_strength,
                    variation_steps=variation_steps,
                    enable_animation=enable_animation,
                    enable_cascade_filter=enable_cascade_filter,
                    cascade_blur_kernel=cascade_blur_kernel,
                    cascade_threshold_ratio=cascade_threshold_ratio,
                    enable_detail_sharpening=enable_detail_sharpening,
                    sharpening_radius=sharpening_radius,
                    sharpening_amount=sharpening_amount,
                    sharpening_threshold=sharpening_threshold,
                    customize_tile_preprocessing=customize_tile_preprocessing,
                    tile_pyrup_iters=tile_pyrup_iters,
                    gr_progress=gr_progress,
                ):
                    yield result
        except Exception:
            print(f"[generation-error][unified][{pipeline}]", traceback.format_exc())
            raise

    # Log actual time spent after generation completes
    elapsed_time = time.time() - start_time
    print(
        f"[GPU TIMING] Completed generation in {elapsed_time:.2f}s (pipeline={pipeline}, image_size={image_size})"
    )


class AnimationHandler:
    """Handler for managing KSampler animation callbacks"""

    def __init__(self, preview_size=512):
        self.intermediate_images = []
        self.image_queue = queue.Queue()
        self.enabled = False
        self.preview_size = (
            preview_size  # Consistent preview size for all intermediate images
        )

    def create_callback(self, vae, interval=5):
        """Create a callback that stores intermediate decoded images"""
        last_step = [0]

        def callback(step, x0, x, total_steps):
            if not self.enabled:
                return

            # Only decode every 'interval' steps, but skip the very last step to avoid contaminating main pipeline
            if (step - last_step[0]) >= interval and step < total_steps:
                last_step[0] = step
                try:
                    # Use torch.no_grad() instead of inference_mode to avoid tensor contamination
                    # Key insight: inference_mode tensors cannot be used in backward pass
                    # Source: https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
                    import torch

                    with torch.no_grad():
                        # Create a detached clone and ensure contiguous memory layout
                        # .contiguous() ensures proper memory layout for VAE decoder
                        x0_copy = x0.detach().clone().contiguous()

                        # CRITICAL: Scale the latent for VAE decoding
                        # SD1.5 uses scale_factor = 0.18215, so we divide by it
                        # This converts from model sampling space to VAE decoding space
                        x0_scaled = x0_copy / 0.18215

                        # Decode - create full latent dict like KSampler output
                        latent_dict = {"samples": x0_scaled}
                        decoded = vaedecode.decode(samples=latent_dict, vae=vae)
                        image_tensor = get_value_at_index(decoded, 0)

                        # Convert EXACTLY like final images (lines 1915-1918) - no transpose, no mode
                        image_np = (image_tensor.detach().cpu().numpy() * 255).astype(
                            np.uint8
                        )
                        image_np = image_np[0]
                        pil_image = Image.fromarray(image_np)

                        # Resize to consistent preview size to avoid size inconsistencies in UI
                        if (
                            pil_image.size[0] > self.preview_size
                            or pil_image.size[1] > self.preview_size
                        ):
                            pil_image.thumbnail(
                                (self.preview_size, self.preview_size), Image.LANCZOS
                            )

                        # Store with message (step is already the correct value at interval points)
                        msg = f"Sampling progress: step {step}/{total_steps}"
                        self.intermediate_images.append((pil_image, msg))
                        self.image_queue.put((pil_image, msg))
                except Exception as e:
                    print(f"Animation decode error: {e}")

        return callback

    def get_and_clear_images(self):
        """Get all intermediate images and clear the buffer"""
        images = self.intermediate_images.copy()
        self.intermediate_images.clear()
        return images


def ksampler_with_animation(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise=1.0,
    animation_handler=None,
    vae=None,
):
    """
    Custom KSampler that supports animation callbacks.
    Based on ComfyUI's common_ksampler but with animation support.
    """
    import comfy.sample
    import comfy.utils

    # Prepare noise
    latent = latent_image
    latent_image_data = latent["samples"]
    latent_image_data = comfy.sample.fix_empty_latent_channels(model, latent_image_data)

    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(latent_image_data, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    # Create animation callback once before sampling (not on every step!)
    callback_fn = None
    if animation_handler and animation_handler.enabled and vae:
        callback_fn = animation_handler.create_callback(vae)

    def animation_callback(step, x0, x, total_steps):
        # Call animation callback if enabled
        if callback_fn:
            callback_fn(step, x0, x, total_steps)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    # Sample
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image_data,
        denoise=denoise,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        noise_mask=noise_mask,
        callback=animation_callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )

    out = latent.copy()
    out["samples"] = samples
    return (out,)


def apply_color_quantization(
    image: Image.Image,
    colors: list[str],
    num_colors: int = 4,
    apply_gradients: bool = False,
    gradient_strength: float = 0.3,
    variation_steps: int = 5,
) -> Image.Image:
    """
    Apply color quantization to an image using nearest-color mapping.
    Optionally apply gradient filter for artistic effect while preserving QR scannability.

    Args:
        image: PIL Image to quantize
        colors: List of hex color strings (e.g., ["#FF0000", "#00FF00", "#0000FF", "#FFFFFF"])
        num_colors: Number of colors to use from the colors list (2-4)
        apply_gradients: If True, create gradient variations around base colors
        gradient_strength: How much brightness variation to allow (0.0-1.0), e.g. 0.3 = ±30%
        variation_steps: Number of gradient steps for each color (1-10)

    Returns:
        Quantized PIL Image (with optional gradient effect)

    Note:
        When gradients are enabled, first 2 colors are always preserved (no gradients)
        to maintain QR code scannability. Only colors 3-4 get gradient variations.
    """
    # Validate num_colors
    if num_colors < 2:
        num_colors = 2
    if num_colors > len(colors):
        num_colors = len(colors)

    # Parse colors with error handling (supports both hex and rgba formats)
    palette = []
    for color_str in colors[:num_colors]:
        try:
            # Check if it's an rgba string (from Gradio ColorPicker)
            if color_str.startswith("rgba("):
                # Extract RGB values from "rgba(r, g, b, a)" format
                rgb_part = color_str[5:-1]  # Remove "rgba(" and ")"
                values = [float(v.strip()) for v in rgb_part.split(",")]
                r = int(values[0])
                g = int(values[1])
                b = int(values[2])
                palette.append((r, g, b))
            else:
                # Assume hex format
                color_hex = color_str.lstrip("#")
                r = int(color_hex[0:2], 16)
                g = int(color_hex[2:4], 16)
                b = int(color_hex[4:6], 16)
                palette.append((r, g, b))
        except (ValueError, IndexError, AttributeError):
            # Fallback to black for invalid colors
            palette.append((0, 0, 0))

    # Ensure at least 2 colors
    if len(palette) < 2:
        palette = [(0, 0, 0), (255, 255, 255)]  # Default to black & white

    # Ensure image is in RGB mode (fixes MPS grayscale conversion bug)
    # On MPS devices, PIL might incorrectly interpret the image as grayscale
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert PIL Image to numpy array
    img_array = np.array(image)

    # Handle RGBA images by converting to RGB (though we already converted above)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    # Handle grayscale images that slipped through
    elif len(img_array.shape) == 2:
        # Convert grayscale to RGB by repeating the channel
        img_array = np.stack([img_array, img_array, img_array], axis=2)

    h, w, c = img_array.shape
    pixels = img_array.reshape(h * w, c).astype(np.float32)

    # ============================================================
    # GRADIENT FILTER MODE: Create gradient variations
    # ============================================================
    if apply_gradients:
        # Always preserve first 2 colors (black/white for QR scannability)
        preserve_colors = [0, 1]

        # Create gradient palette
        palette_with_gradients = []
        color_family_map = []  # Track which base color each gradient belongs to

        for base_idx, base_color in enumerate(palette):
            r, g, b = base_color

            # Check if this color should be preserved (no gradients)
            if base_idx in preserve_colors:
                # Keep this color pure - only add the base color once
                palette_with_gradients.append((r, g, b))
                color_family_map.append(base_idx)
            else:
                # Create variations from dark to light
                for step in range(variation_steps):
                    # Calculate brightness multiplier
                    if variation_steps == 1:
                        multiplier = 1.0  # Only use base color when steps=1
                    else:
                        multiplier = 1.0 + gradient_strength * (
                            2 * step / (variation_steps - 1) - 1
                        )

                    # Apply multiplier and clamp to valid range
                    varied_r = int(np.clip(r * multiplier, 0, 255))
                    varied_g = int(np.clip(g * multiplier, 0, 255))
                    varied_b = int(np.clip(b * multiplier, 0, 255))

                    palette_with_gradients.append((varied_r, varied_g, varied_b))
                    color_family_map.append(base_idx)

        gradient_palette_array = np.array(palette_with_gradients, dtype=np.float32)
        base_palette_array = np.array(palette, dtype=np.float32)

        # Calculate original pixel brightness for gradient selection
        pixel_brightness = np.mean(pixels, axis=1)

        # Step 1: Find nearest BASE color for each pixel
        distances_to_base = np.sqrt(
            np.sum((pixels[:, None, :] - base_palette_array[None, :, :]) ** 2, axis=2)
        )
        nearest_base_idx = np.argmin(distances_to_base, axis=1)

        # Step 2: Fully vectorized gradient assignment
        # Create mapping from base color index to gradient range
        gradient_ranges = {}
        for base_idx in range(len(palette)):
            family_indices = [
                i for i, fam in enumerate(color_family_map) if fam == base_idx
            ]
            gradient_ranges[base_idx] = np.array(family_indices)

        # Initialize result
        result_indices = np.zeros(len(pixels), dtype=int)

        # For each base color family, compute gradient indices
        for base_idx in range(len(palette)):
            mask = nearest_base_idx == base_idx
            if not np.any(mask):
                continue

            family_indices = gradient_ranges[base_idx]
            masked_brightness = pixel_brightness[mask]

            # Normalize brightness within this family
            min_b, max_b = masked_brightness.min(), masked_brightness.max()
            if max_b > min_b:
                norm_bright = (masked_brightness - min_b) / (max_b - min_b)
            else:
                norm_bright = np.full(len(masked_brightness), 0.5)

            # Map to gradient steps
            steps = (norm_bright * (len(family_indices) - 1)).astype(int)
            steps = np.clip(steps, 0, len(family_indices) - 1)

            # Assign palette indices
            result_indices[mask] = family_indices[steps]

        # Final color assignment
        result_pixels = gradient_palette_array[result_indices].astype(np.uint8)
        quantized_image = result_pixels.reshape(h, w, c)

    # ============================================================
    # STRICT QUANTIZATION MODE: No gradients
    # ============================================================
    else:
        # Convert palette to numpy array
        palette_array = np.array(palette, dtype=np.uint8)

        # Calculate Euclidean distance from each pixel to each palette color
        distances = np.sqrt(
            np.sum(
                (pixels[:, None, :] - palette_array[None, :, :].astype(np.float32))
                ** 2,
                axis=2,
            )
        )

        # Find index of nearest color for each pixel
        nearest_indices = np.argmin(distances, axis=1)

        # Map each pixel to its nearest palette color
        quantized = palette_array[nearest_indices]

        # Reshape back to image dimensions
        quantized_image = quantized.reshape(h, w, c).astype(np.uint8)

    # Convert back to PIL Image
    return Image.fromarray(quantized_image)


def apply_stable_cascade_qr_filter(
    image_tensor: torch.Tensor,
    blur_kernel: int = 15,
    threshold_ratio: float = 0.33,
    device: str = None,
) -> torch.Tensor:
    """
    Apply Stable Cascade QR filter for better brightness-based control.

    This filter improves ControlNet perception by:
    1. Converting to HSV (perceptually accurate brightness)
    2. Applying Gaussian blur (reduces blockiness)
    3. Adaptive thresholding (creates 3 brightness levels)

    Based on: https://github.com/Stability-AI/StableCascade

    Args:
        image_tensor: Input QR tensor [B, H, W, C] or [B, C, H, W] in range [0, 1]
        blur_kernel: Gaussian blur kernel size (default: 15)
        threshold_ratio: Brightness threshold ratio (default: 0.33)
        device: Target device (auto-detect if None)

    Returns:
        Filtered tensor in same format as input with three brightness levels (0.0, 0.5, 1.0)
    """
    if device is None:
        device = image_tensor.device

    # Ensure tensor is on correct device and in float32
    x = image_tensor.to(device).float()

    # ComfyUI images are [B, H, W, C], but kornia expects [B, C, H, W]
    needs_permute = x.ndim == 4 and x.shape[-1] == 3
    if needs_permute:
        x = x.permute(0, 3, 1, 2)

    # 1. Convert RGB to HSV and extract Value (brightness) channel
    x_hsv = kornia.color.rgb_to_hsv(x)
    brightness = x_hsv[:, -1:, :, :]  # Shape: [B, 1, H, W]

    # 2. Apply Gaussian blur
    if blur_kernel > 0:
        kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        import torchvision

        brightness = torchvision.transforms.GaussianBlur(kernel)(brightness)

    # 3. Adaptive thresholding
    vmax = brightness.amax(dim=[2, 3], keepdim=True)
    vmin = brightness.amin(dim=[2, 3], keepdim=True)
    threshold = (vmax - vmin) * threshold_ratio

    # 4. Create three-level mask (0.0=dark, 0.5=mid, 1.0=bright)
    high_brightness = (brightness > (vmax - threshold)).float()
    low_brightness = (brightness < (vmin + threshold)).float()
    mask = (torch.ones_like(brightness) - low_brightness + high_brightness) * 0.5

    # 5. Convert to 3-channel RGB
    filtered = mask.repeat(1, 3, 1, 1)

    # 6. Permute back to ComfyUI format if needed
    if needs_permute:
        filtered = filtered.permute(0, 2, 3, 1)

    return filtered


def apply_detail_sharpening(
    image: Image.Image, radius: float = 2.0, amount: float = 1.5, threshold: int = 0
) -> Image.Image:
    """
    Apply unsharp mask sharpening to preserve QR details between passes.

    This filter is applied to the first-pass output before the second pass,
    helping maintain sharp QR code edges when using lower ControlNet strengths.

    Args:
        image: PIL Image to sharpen
        radius: Sharpening radius in pixels (1.0-5.0)
                Higher values = wider sharpening effect
        amount: Sharpening strength (0.5-3.0)
                Higher values = stronger sharpening
        threshold: Minimum brightness change to sharpen (0-10)
                   0 = sharpen all pixels, higher = sharpen only high-contrast edges

    Returns:
        Sharpened PIL Image
    """
    from PIL import ImageFilter

    # Ensure radius is valid for UnsharpMask (must be positive)
    radius = max(0.1, radius)

    # Apply unsharp mask filter
    # PIL's UnsharpMask expects percent as integer (0-100+)
    sharpened = image.filter(
        ImageFilter.UnsharpMask(
            radius=radius, percent=int(amount * 100), threshold=threshold
        )
    )

    return sharpened


def generate_standard_qr(
    prompt: str,
    negative_prompt: str = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
    text_input: str = "",
    input_type: str = "URL",
    use_temporary_short_link: bool = False,
    image_size: int = 512,
    border_size: int = 4,
    error_correction: str = "Medium (15%)",
    module_size: int = 12,
    module_drawer: str = "Square",
    use_custom_seed: bool = False,
    seed: int = 0,
    enable_upscale: bool = False,
    enable_animation: bool = True,
    controlnet_strength_standard_first: float = 0.45,
    controlnet_strength_standard_final: float = 1.0,
    enable_color_quantization: bool = False,
    num_colors: int = 4,
    color_1: str = "#000000",
    color_2: str = "#FFFFFF",
    color_3: str = "#FF0000",
    color_4: str = "#00FF00",
    apply_gradient_filter: bool = False,
    gradient_strength: float = 0.3,
    variation_steps: int = 5,
    analytics_opt_in: bool = False,
    progress=gr.Progress(),
    request: Union[gr.Request, None] = None,
):
    """Wrapper function for standard QR generation.

    Set analytics_opt_in=True to share prompt, QR payload, and settings for product improvement.
    Generated images are never stored for analytics.
    """
    generation_id = str(uuid.uuid4())
    analytics_opt_in = _normalize_bool(analytics_opt_in)
    source = _detect_source(request)
    url_normalization = (
        _normalize_url_for_qr(text_input) if input_type == "URL" else None
    )
    qr_text_input = (
        str(url_normalization.get("normalized_qr_text", text_input))
        if url_normalization is not None
        else text_input
    )
    url_normalization_note = _build_url_normalization_note(url_normalization)
    shortener_result = _maybe_shorten_url_for_qr(
        original_input=text_input,
        input_type=input_type,
        use_temporary_short_link=use_temporary_short_link,
    )
    qr_text_input = str(shortener_result.get("effective_qr_text") or qr_text_input)
    normalization_applied_for_qr = bool(
        url_normalization
        and url_normalization.get("changed")
        and not shortener_result.get("applied")
    )
    normalization_tracking_params_removed = int(
        (url_normalization or {}).get("tracking_params_removed") or 0
    )
    normalization_chars_saved = int((url_normalization or {}).get("chars_saved") or 0)
    if shortener_result.get("applied"):
        normalization_tracking_params_removed = 0
        normalization_chars_saved = 0
        url_normalization_note = None
    shortener_note = _build_shortener_note(shortener_result)
    # Get actual seed used (custom or random)
    actual_seed = seed if use_custom_seed else random.randint(1, 2**32 - 1)

    # Create settings JSON once
    settings_dict = {
        "pipeline": "standard",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "text_input": text_input,
        "effective_qr_text": qr_text_input,
        "use_temporary_short_link": _normalize_bool(use_temporary_short_link),
        "shortener_applied": bool(shortener_result.get("applied")),
        "short_url": shortener_result.get("short_url"),
        "shortener_expires_at": shortener_result.get("expires_at"),
        "shortener_error": shortener_result.get("error"),
        "input_type": input_type,
        "image_size": image_size,
        "border_size": border_size,
        "error_correction": error_correction,
        "module_size": module_size,
        "module_drawer": module_drawer,
        "seed": actual_seed,
        "use_custom_seed": True,
        "enable_upscale": enable_upscale,
        "enable_animation": enable_animation,
        "controlnet_strength_standard_first": controlnet_strength_standard_first,
        "controlnet_strength_standard_final": controlnet_strength_standard_final,
        "enable_color_quantization": enable_color_quantization,
        "num_colors": num_colors,
        "color_1": color_1,
        "color_2": color_2,
        "color_3": color_3,
        "color_4": color_4,
        "apply_gradient_filter": apply_gradient_filter,
        "gradient_strength": gradient_strength,
        "variation_steps": variation_steps,
        "url_normalization_applied": normalization_applied_for_qr,
        "url_tracking_params_removed": normalization_tracking_params_removed,
        "url_chars_saved": normalization_chars_saved,
    }
    settings_json = generate_settings_json(settings_dict)

    try:
        _validate_qr_dimensions(
            qr_text=qr_text_input,
            input_type=input_type,
            image_size=image_size,
            border_size=border_size,
            error_correction=error_correction,
            module_size=module_size,
        )
    except RuntimeError as exc:
        final_status = str(exc)
        if ANALYTICS_ENABLED:
            _record_validation_event(
                _build_validation_payload(
                    generation_id=generation_id,
                    source=source,
                    pipeline="standard",
                    tool_name="generate_standard_qr",
                    analytics_opt_in=analytics_opt_in,
                    prompt=prompt,
                    text_input=text_input,
                    settings=settings_dict,
                    status=final_status,
                    request=request,
                )
            )
        if source == "mcp":
            raise RuntimeError(final_status)
        yield (
            None,
            _append_status_note(
                _append_status_note(final_status, url_normalization_note),
                shortener_note,
            ),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
        )
        return

    # Generate QR and yield progressive results
    generator = generate_qr_code_unified(
        prompt,
        negative_prompt,
        qr_text_input,
        input_type,
        image_size,
        border_size,
        error_correction,
        module_size,
        module_drawer,
        use_custom_seed,
        seed,
        pipeline="standard",
        enable_upscale=enable_upscale,
        enable_animation=enable_animation,
        controlnet_strength_standard_first=controlnet_strength_standard_first,
        controlnet_strength_standard_final=controlnet_strength_standard_final,
        enable_color_quantization=enable_color_quantization,
        num_colors=num_colors,
        color_1=color_1,
        color_2=color_2,
        color_3=color_3,
        color_4=color_4,
        apply_gradient_filter=apply_gradient_filter,
        gradient_strength=gradient_strength,
        variation_steps=variation_steps,
        gr_progress=progress,
    )

    final_image = None
    final_status = None

    try:
        for image, status in generator:
            final_image = image
            final_status = status
            # Show progressive updates but don't show accordion or export buttons yet
            yield (
                image,
                _append_status_note(
                    _append_status_note(status, url_normalization_note),
                    shortener_note,
                ),
                gr.update(),
                gr.update(),
                gr.update(visible=False),
            )
    except Exception as exc:
        print("[generation-error][standard-wrapper]", traceback.format_exc())
        final_status = _format_exception_message(exc)

    # After all steps complete, show the accordion with JSON and export buttons
    if final_image is not None:
        if ANALYTICS_ENABLED:
            _record_generation_event(
                _build_generation_payload(
                    generation_id=generation_id,
                    source=source,
                    pipeline="standard",
                    tool_name="generate_standard_qr",
                    analytics_opt_in=analytics_opt_in,
                    prompt=prompt,
                    text_input=text_input,
                    settings=settings_dict,
                    status="success",
                    request=request,
                )
            )
        yield (
            final_image,
            _append_status_note(
                _append_status_note(final_status, url_normalization_note),
                shortener_note,
            ),
            gr.update(value=settings_json),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    elif ANALYTICS_ENABLED:
        _record_generation_event(
            _build_generation_payload(
                generation_id=generation_id,
                source=source,
                pipeline="standard",
                tool_name="generate_standard_qr",
                analytics_opt_in=analytics_opt_in,
                prompt=prompt,
                text_input=text_input,
                settings=settings_dict,
                status=final_status or "Generation failed",
                request=request,
            )
        )
    if final_image is None and final_status:
        if source == "mcp":
            raise RuntimeError(final_status)
        yield (
            None,
            _append_status_note(
                _append_status_note(final_status, url_normalization_note),
                shortener_note,
            ),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
        )


def generate_artistic_qr(
    prompt: str,
    negative_prompt: str = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
    text_input: str = "",
    input_type: str = "URL",
    use_temporary_short_link: bool = False,
    image_size: int = 512,
    border_size: int = 4,
    error_correction: str = "Medium (15%)",
    module_size: int = 12,
    module_drawer: str = "Square",
    use_custom_seed: bool = False,
    seed: int = 0,
    enable_upscale: bool = False,
    enable_animation: bool = True,
    enable_cascade_filter: bool = False,
    cascade_blur_kernel: int = 15,
    cascade_threshold_ratio: float = 0.33,
    enable_detail_sharpening: bool = False,
    sharpening_radius: float = 2.0,
    sharpening_amount: float = 1.5,
    sharpening_threshold: int = 0,
    customize_tile_preprocessing: bool = False,
    tile_pyrup_iters: int = 3,
    enable_freeu: bool = True,
    freeu_b1: float = 1.4,
    freeu_b2: float = 1.3,
    freeu_s1: float = 0.0,
    freeu_s2: float = 1.3,
    enable_sag: bool = True,
    sag_scale: float = 0.5,
    sag_blur_sigma: float = 0.5,
    controlnet_strength_first: float = 0.45,
    controlnet_strength_final: float = 0.70,
    enable_color_quantization: bool = False,
    num_colors: int = 4,
    color_1: str = "#000000",
    color_2: str = "#FFFFFF",
    color_3: str = "#FF0000",
    color_4: str = "#00FF00",
    apply_gradient_filter: bool = False,
    gradient_strength: float = 0.3,
    variation_steps: int = 5,
    analytics_opt_in: bool = False,
    progress=gr.Progress(),
    request: Union[gr.Request, None] = None,
):
    """Wrapper function for artistic QR generation with FreeU and SAG parameters.

    Set analytics_opt_in=True to share prompt, QR payload, and settings for product improvement.
    Generated images are never stored for analytics.
    """
    generation_id = str(uuid.uuid4())
    analytics_opt_in = _normalize_bool(analytics_opt_in)
    source = _detect_source(request)
    url_normalization = (
        _normalize_url_for_qr(text_input) if input_type == "URL" else None
    )
    qr_text_input = (
        str(url_normalization.get("normalized_qr_text", text_input))
        if url_normalization is not None
        else text_input
    )
    url_normalization_note = _build_url_normalization_note(url_normalization)
    shortener_result = _maybe_shorten_url_for_qr(
        original_input=text_input,
        input_type=input_type,
        use_temporary_short_link=use_temporary_short_link,
    )
    qr_text_input = str(shortener_result.get("effective_qr_text") or qr_text_input)
    normalization_applied_for_qr = bool(
        url_normalization
        and url_normalization.get("changed")
        and not shortener_result.get("applied")
    )
    normalization_tracking_params_removed = int(
        (url_normalization or {}).get("tracking_params_removed") or 0
    )
    normalization_chars_saved = int((url_normalization or {}).get("chars_saved") or 0)
    if shortener_result.get("applied"):
        normalization_tracking_params_removed = 0
        normalization_chars_saved = 0
        url_normalization_note = None
    shortener_note = _build_shortener_note(shortener_result)
    # Get actual seed used (custom or random)
    actual_seed = seed if use_custom_seed else random.randint(1, 2**32 - 1)

    # Create settings JSON once
    settings_dict = {
        "pipeline": "artistic",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "text_input": text_input,
        "effective_qr_text": qr_text_input,
        "use_temporary_short_link": _normalize_bool(use_temporary_short_link),
        "shortener_applied": bool(shortener_result.get("applied")),
        "short_url": shortener_result.get("short_url"),
        "shortener_expires_at": shortener_result.get("expires_at"),
        "shortener_error": shortener_result.get("error"),
        "input_type": input_type,
        "image_size": image_size,
        "border_size": border_size,
        "error_correction": error_correction,
        "module_size": module_size,
        "module_drawer": module_drawer,
        "seed": actual_seed,
        "use_custom_seed": True,
        "enable_upscale": enable_upscale,
        "enable_animation": enable_animation,
        "enable_freeu": enable_freeu,
        "freeu_b1": freeu_b1,
        "freeu_b2": freeu_b2,
        "freeu_s1": freeu_s1,
        "freeu_s2": freeu_s2,
        "enable_sag": enable_sag,
        "sag_scale": sag_scale,
        "sag_blur_sigma": sag_blur_sigma,
        "controlnet_strength_first": controlnet_strength_first,
        "controlnet_strength_final": controlnet_strength_final,
        "enable_color_quantization": enable_color_quantization,
        "num_colors": num_colors,
        "color_1": color_1,
        "color_2": color_2,
        "color_3": color_3,
        "color_4": color_4,
        "apply_gradient_filter": apply_gradient_filter,
        "gradient_strength": gradient_strength,
        "variation_steps": variation_steps,
        "url_normalization_applied": normalization_applied_for_qr,
        "url_tracking_params_removed": normalization_tracking_params_removed,
        "url_chars_saved": normalization_chars_saved,
    }
    settings_json = generate_settings_json(settings_dict)

    try:
        _validate_qr_dimensions(
            qr_text=qr_text_input,
            input_type=input_type,
            image_size=image_size,
            border_size=border_size,
            error_correction=error_correction,
            module_size=module_size,
        )
    except RuntimeError as exc:
        final_status = str(exc)
        if ANALYTICS_ENABLED:
            _record_validation_event(
                _build_validation_payload(
                    generation_id=generation_id,
                    source=source,
                    pipeline="artistic",
                    tool_name="generate_artistic_qr",
                    analytics_opt_in=analytics_opt_in,
                    prompt=prompt,
                    text_input=text_input,
                    settings=settings_dict,
                    status=final_status,
                    request=request,
                )
            )
        if source == "mcp":
            raise RuntimeError(final_status)
        yield (
            None,
            _append_status_note(
                _append_status_note(final_status, url_normalization_note),
                shortener_note,
            ),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )
        return

    # Generate QR and yield progressive results
    generator = generate_qr_code_unified(
        prompt,
        negative_prompt,
        qr_text_input,
        input_type,
        image_size,
        border_size,
        error_correction,
        module_size,
        module_drawer,
        use_custom_seed,
        seed,
        pipeline="artistic",
        enable_upscale=enable_upscale,
        freeu_b1=freeu_b1,
        freeu_b2=freeu_b2,
        freeu_s1=freeu_s1,
        freeu_s2=freeu_s2,
        enable_sag=enable_sag,
        sag_scale=sag_scale,
        sag_blur_sigma=sag_blur_sigma,
        controlnet_strength_first=controlnet_strength_first,
        controlnet_strength_final=controlnet_strength_final,
        enable_color_quantization=enable_color_quantization,
        num_colors=num_colors,
        color_1=color_1,
        color_2=color_2,
        color_3=color_3,
        color_4=color_4,
        apply_gradient_filter=apply_gradient_filter,
        gradient_strength=gradient_strength,
        variation_steps=variation_steps,
        enable_animation=enable_animation,
        enable_cascade_filter=enable_cascade_filter,
        cascade_blur_kernel=cascade_blur_kernel,
        cascade_threshold_ratio=cascade_threshold_ratio,
        enable_detail_sharpening=enable_detail_sharpening,
        sharpening_radius=sharpening_radius,
        sharpening_amount=sharpening_amount,
        sharpening_threshold=sharpening_threshold,
        customize_tile_preprocessing=customize_tile_preprocessing,
        tile_pyrup_iters=tile_pyrup_iters,
        gr_progress=progress,
    )

    final_image = None
    final_status = None
    first_yield = True

    try:
        for image, status in generator:
            final_image = image
            final_status = status
            # Show progressive updates but don't show accordion yet
            # On first yield, hide gallery and show output components
            if first_yield:
                yield (
                    gr.update(visible=True, value=image),
                    _append_status_note(
                        _append_status_note(status, url_normalization_note),
                        shortener_note,
                    ),
                    gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
                first_yield = False
            else:
                yield (
                    image,
                    _append_status_note(
                        _append_status_note(status, url_normalization_note),
                        shortener_note,
                    ),
                    gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
    except Exception as exc:
        print("[generation-error][artistic-wrapper]", traceback.format_exc())
        final_status = _format_exception_message(exc)

    # After all steps complete, show the accordion with JSON and the "Try Another Example" button
    if final_image is not None:
        if ANALYTICS_ENABLED:
            _record_generation_event(
                _build_generation_payload(
                    generation_id=generation_id,
                    source=source,
                    pipeline="artistic",
                    tool_name="generate_artistic_qr",
                    analytics_opt_in=analytics_opt_in,
                    prompt=prompt,
                    text_input=text_input,
                    settings=settings_dict,
                    status="success",
                    request=request,
                )
            )
        yield (
            final_image,
            _append_status_note(
                _append_status_note(final_status, url_normalization_note),
                shortener_note,
            ),
            gr.update(value=settings_json),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    elif ANALYTICS_ENABLED:
        _record_generation_event(
            _build_generation_payload(
                generation_id=generation_id,
                source=source,
                pipeline="artistic",
                tool_name="generate_artistic_qr",
                analytics_opt_in=analytics_opt_in,
                prompt=prompt,
                text_input=text_input,
                settings=settings_dict,
                status=final_status or "Generation failed",
                request=request,
            )
        )
    if final_image is None and final_status:
        if source == "mcp":
            raise RuntimeError(final_status)
        yield (
            None,
            _append_status_note(
                _append_status_note(final_status, url_normalization_note),
                shortener_note,
            ),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )


# Helper functions for shareable settings JSON


def generate_settings_json(params_dict: dict) -> str:
    """Generate a formatted JSON string from parameters dictionary"""
    try:
        return json.dumps(params_dict, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to generate JSON: {str(e)}"}, indent=2)


def parse_settings_json(json_string: str) -> dict:
    """Parse JSON string and return parameters dictionary with validation"""
    try:
        if not json_string or not json_string.strip():
            return {}

        params = json.loads(json_string)
        if not isinstance(params, dict):
            return {}

        return params
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to parse JSON: {str(e)}"}


def load_settings_from_json_standard(json_string: str):
    """Load settings from JSON for Standard pipeline"""
    try:
        params = json.loads(json_string)

        # Validate pipeline type
        pipeline = params.get(
            "pipeline", "standard"
        )  # Default to standard for backward compatibility
        if pipeline != "standard":
            error_msg = f"❌ Error: You're trying to load {pipeline.upper()} pipeline settings into the STANDARD pipeline. Please use the correct tab."
            # Return empty updates for all fields + error message + make status visible
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(value=error_msg, visible=True),
            )

        # Extract parameters with defaults
        prompt = params.get("prompt", "")
        negative_prompt = params.get(
            "negative_prompt",
            "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
        )
        text_input = params.get("text_input", "")
        input_type = params.get("input_type", "URL")
        use_temporary_short_link = params.get("use_temporary_short_link", False)
        image_size = params.get("image_size", 512)
        border_size = params.get("border_size", 4)
        error_correction = params.get("error_correction", "Medium (15%)")
        module_size = params.get("module_size", 12)
        module_drawer = params.get("module_drawer", "Square")
        use_custom_seed = params.get("use_custom_seed", True)
        seed = params.get("seed", 718313)
        enable_upscale = params.get("enable_upscale", False)
        enable_animation = params.get("enable_animation", True)
        controlnet_strength_standard_first = params.get(
            "controlnet_strength_standard_first", 0.45
        )
        controlnet_strength_standard_final = params.get(
            "controlnet_strength_standard_final", 1.0
        )
        enable_color_quantization = params.get("enable_color_quantization", False)
        num_colors = params.get("num_colors", 4)
        color_1 = params.get("color_1", "#000000")
        color_2 = params.get("color_2", "#FFFFFF")
        color_3 = params.get("color_3", "#FF0000")
        color_4 = params.get("color_4", "#00FF00")
        apply_gradient_filter = params.get("apply_gradient_filter", False)
        gradient_strength = params.get("gradient_strength", 0.3)
        variation_steps = params.get("variation_steps", 5)

        success_msg = "✅ Settings loaded successfully!"
        return (
            prompt,
            negative_prompt,
            text_input,
            input_type,
            use_temporary_short_link,
            image_size,
            border_size,
            error_correction,
            module_size,
            module_drawer,
            use_custom_seed,
            seed,
            enable_upscale,
            enable_animation,
            controlnet_strength_standard_first,
            controlnet_strength_standard_final,
            enable_color_quantization,
            num_colors,
            color_1,
            color_2,
            color_3,
            color_4,
            apply_gradient_filter,
            gradient_strength,
            variation_steps,
            gr.update(value=success_msg, visible=True),
        )

    except json.JSONDecodeError as e:
        error_msg = f"❌ Invalid JSON format: {str(e)}"
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=error_msg, visible=True),
        )
    except Exception as e:
        error_msg = f"❌ Error loading settings: {str(e)}"
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=error_msg, visible=True),
        )


def load_settings_from_json_artistic(json_string: str):
    """Load settings from JSON for Artistic pipeline"""
    try:
        params = json.loads(json_string)

        # Validate pipeline type
        pipeline = params.get(
            "pipeline", "artistic"
        )  # Default to artistic for backward compatibility
        if pipeline != "artistic":
            error_msg = f"❌ Error: You're trying to load {pipeline.upper()} pipeline settings into the ARTISTIC pipeline. Please use the correct tab."
            # Return empty updates for all fields + error message + make status visible
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(value=error_msg, visible=True),
            )

        # Extract parameters with defaults
        prompt = params.get("prompt", "")
        negative_prompt = params.get(
            "negative_prompt",
            "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
        )
        text_input = params.get("text_input", "")
        input_type = params.get("input_type", "URL")
        use_temporary_short_link = params.get("use_temporary_short_link", False)
        image_size = params.get("image_size", 704)
        border_size = params.get("border_size", 6)
        error_correction = params.get("error_correction", "High (30%)")
        module_size = params.get("module_size", 16)
        module_drawer = params.get("module_drawer", "Square")
        use_custom_seed = params.get("use_custom_seed", True)
        seed = params.get("seed", 718313)
        enable_upscale = params.get("enable_upscale", False)
        enable_animation = params.get("enable_animation", True)
        enable_freeu = params.get("enable_freeu", True)
        freeu_b1 = params.get("freeu_b1", 1.4)
        freeu_b2 = params.get("freeu_b2", 1.3)
        freeu_s1 = params.get("freeu_s1", 0.0)
        freeu_s2 = params.get("freeu_s2", 1.3)
        enable_sag = params.get("enable_sag", True)
        sag_scale = params.get("sag_scale", 0.5)
        sag_blur_sigma = params.get("sag_blur_sigma", 0.5)
        controlnet_strength_first = params.get("controlnet_strength_first", 0.45)
        controlnet_strength_final = params.get("controlnet_strength_final", 0.7)
        enable_color_quantization = params.get("enable_color_quantization", False)
        num_colors = params.get("num_colors", 4)
        color_1 = params.get("color_1", "#000000")
        color_2 = params.get("color_2", "#FFFFFF")
        color_3 = params.get("color_3", "#FF0000")
        color_4 = params.get("color_4", "#00FF00")
        apply_gradient_filter = params.get("apply_gradient_filter", False)
        gradient_strength = params.get("gradient_strength", 0.3)
        variation_steps = params.get("variation_steps", 5)

        success_msg = "✅ Settings loaded successfully!"
        return (
            prompt,
            negative_prompt,
            text_input,
            input_type,
            use_temporary_short_link,
            image_size,
            border_size,
            error_correction,
            module_size,
            module_drawer,
            use_custom_seed,
            seed,
            enable_upscale,
            enable_animation,
            enable_freeu,
            freeu_b1,
            freeu_b2,
            freeu_s1,
            freeu_s2,
            enable_sag,
            sag_scale,
            sag_blur_sigma,
            controlnet_strength_first,
            controlnet_strength_final,
            enable_color_quantization,
            num_colors,
            color_1,
            color_2,
            color_3,
            color_4,
            apply_gradient_filter,
            gradient_strength,
            variation_steps,
            gr.update(value=success_msg, visible=True),
        )

    except json.JSONDecodeError as e:
        error_msg = f"❌ Invalid JSON format: {str(e)}"
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=error_msg, visible=True),
        )
    except Exception as e:
        error_msg = f"❌ Error loading settings: {str(e)}"
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=error_msg, visible=True),
        )


def add_noise_to_border_only(
    image_tensor, seed: int, border_size: int, image_size: int, module_size: int = 12
):
    """
    Add QR-like cubic patterns ONLY to the border region of a QR code image.
    Creates black squares that resemble QR modules for a smooth transition.
    The density of border cubics automatically matches the QR code interior density.

    Args:
        image_tensor: ComfyUI image tensor (batch, height, width, channels) with values 0-1
        seed: Random seed for reproducible noise
        border_size: Border size in QR modules (from QR generation settings)
        image_size: Image size in pixels
        module_size: Size of QR modules in pixels (for cubic pattern)

    Returns:
        Modified tensor with QR-like cubic patterns in border region
    """
    # Early return if no border
    if border_size == 0:
        return image_tensor

    # Convert to numpy for manipulation
    img_np = image_tensor.detach().cpu().numpy()

    # Set random seed for reproducibility (ensure it's within numpy's valid range)
    np.random.seed(seed % (2**32))

    # Work with first image in batch
    img = img_np[0]  # (height, width, channels)
    height, width, channels = img.shape

    # Calculate border region in pixels using exact QR parameters
    border_thickness = border_size * module_size  # Exact border size in pixels

    # Create border mask (1 for border region, 0 for QR code interior)
    border_mask = np.zeros((height, width), dtype=bool)

    # Top border
    border_mask[0:border_thickness, :] = True
    # Bottom border
    border_mask[height - border_thickness : height, :] = True
    # Left border
    border_mask[:, 0:border_thickness] = True
    # Right border
    border_mask[:, width - border_thickness : width] = True

    # Only apply to white/light areas in the border (threshold > 240)
    img_255 = (img * 255).astype(np.uint8)
    white_mask = np.all(img_255 > 240, axis=-1)

    # Combine: only border AND white areas
    final_mask = border_mask & white_mask

    # Calculate QR code interior density to determine border cubic density
    interior_mask = ~border_mask  # Inverse of border = QR interior
    interior_pixels = img_255[interior_mask][:, 0]  # Get first channel (grayscale)
    black_count = (interior_pixels < 128).sum()  # Count black pixels (< 128)
    total_count = len(interior_pixels)
    qr_density = float(black_count) / float(total_count) if total_count > 0 else 0.5

    # Use QR interior density as probability for placing border cubics
    # This creates a natural transition matching the QR pattern density

    # Generate QR-like cubic pattern noise
    # Create a grid based on module_size
    for y in range(0, height, module_size):
        for x in range(0, width, module_size):
            # Check if this module position is mostly in the border area
            y_end = min(y + module_size, height)
            x_end = min(x + module_size, width)

            # Count how many pixels in this module are in the final_mask
            module_region = final_mask[y:y_end, x:x_end]

            # If at least 50% of the module is in the border, we can place a cubic here
            if module_region.sum() > (module_size * module_size * 0.5):
                # Randomly decide to place a black cubic based on QR interior density
                if np.random.random() < qr_density:
                    # Place a black square (cubic) - set all channels to 0 (black)
                    for c in range(channels):
                        img[y:y_end, x:x_end, c] = 0

    # Put modified image back into batch array
    img_np[0] = img

    # Convert back to tensor
    return torch.from_numpy(img_np).to(image_tensor.device)


def _pipeline_standard(
    prompt: str,
    negative_prompt: str = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
    qr_text: str = "",
    input_type: str = "URL",
    image_size: int = 512,
    border_size: int = 4,
    error_correction: str = "Medium (15%)",
    module_size: int = 12,
    module_drawer: str = "Square",
    seed: int = 0,
    enable_upscale: bool = False,
    controlnet_strength_first: float = 0.45,
    controlnet_strength_final: float = 1.0,
    enable_color_quantization: bool = False,
    num_colors: int = 4,
    color_1: str = "#000000",
    color_2: str = "#FFFFFF",
    color_3: str = "#FF0000",
    color_4: str = "#00FF00",
    apply_gradient_filter: bool = False,
    gradient_strength: float = 0.3,
    variation_steps: int = 5,
    enable_animation: bool = True,
    gr_progress=None,
):
    # Initialize animation handler if enabled
    animation_handler = (
        AnimationHandler(preview_size=image_size) if enable_animation else None
    )
    if animation_handler:
        animation_handler.enabled = True

    emptylatentimage_5 = emptylatentimage.generate(
        width=image_size, height=image_size, batch_size=1
    )

    cliptextencode_6 = cliptextencode.encode(
        text=prompt,
        clip=get_value_at_index(checkpointloadersimple_4, 1),
    )

    cliptextencode_7 = cliptextencode.encode(
        text=negative_prompt,
        clip=get_value_at_index(checkpointloadersimple_4, 1),
    )

    controlnetloader_10 = controlnetloader.load_controlnet(
        control_net_name="models/control_v1p_sd15_brightness.safetensors"
    )

    controlnetloader_12 = controlnetloader.load_controlnet(
        control_net_name="control_v11f1e_sd15_tile_fp16.safetensors"
    )

    # Set protocol based on input type: None for plain text, Https for URLs
    qr_protocol = "None" if input_type == "Plain Text" else "Https"

    # Test progress bar at the very beginning
    print(f"DEBUG: gr_progress type: {type(gr_progress)}")
    print(f"DEBUG: gr_progress value: {gr_progress}")
    if gr_progress:
        print("DEBUG: Calling gr_progress(0.0)")
        gr_progress(0.0, desc="Starting QR generation...")
        print("DEBUG: Called gr_progress(0.0) successfully")

    try:
        comfy_qr_by_module_size_15 = comfy_qr_by_module_size.generate_qr(
            protocol=qr_protocol,
            text=qr_text,
            module_size=module_size,
            max_image_size=image_size,
            fill_hexcolor="#000000",
            back_hexcolor="#FFFFFF",
            error_correction=error_correction,
            border=border_size,
            module_drawer=module_drawer,
        )
    except RuntimeError as e:
        error_msg = (
            f"Error generating QR code: {str(e)}\n"
            "Try with a shorter text, increase the image size, or decrease the border size, module size, and error correction level under Change Settings Manually."
        )
        yield None, error_msg
        return

    # Calculate total steps based on enabled features
    total_steps = (
        3 + (1 if enable_upscale else 0) + (1 if enable_color_quantization else 0)
    )

    # 1) Yield the base QR image as the first intermediate result
    base_qr_tensor = get_value_at_index(comfy_qr_by_module_size_15, 0)
    base_qr_np = (base_qr_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    base_qr_np = base_qr_np[0]
    base_qr_pil = Image.fromarray(base_qr_np)
    msg = f"Generated base QR pattern… enhancing with AI (step 1/{total_steps})"
    log_progress(msg, gr_progress, 0.05)
    yield base_qr_pil, msg

    emptylatentimage_17 = emptylatentimage.generate(
        width=image_size * 2, height=image_size * 2, batch_size=1
    )

    controlnetloader_19 = controlnetloader.load_controlnet(
        control_net_name="control_v11f1e_sd15_tile_fp16.safetensors"
    )

    # Simple stage update for first pass
    log_progress("First pass - preparing controlnets...", gr_progress, 0.1)

    for q in range(1):
        controlnetapplyadvanced_11 = controlnetapplyadvanced.apply_controlnet(
            strength=controlnet_strength_first,
            start_percent=0,
            end_percent=1,
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            control_net=get_value_at_index(controlnetloader_10, 0),
            image=get_value_at_index(comfy_qr_by_module_size_15, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        tilepreprocessor_14 = tilepreprocessor.execute(
            pyrUp_iters=3,
            resolution=image_size,
            image=get_value_at_index(comfy_qr_by_module_size_15, 0),
        )

        controlnetapplyadvanced_13 = controlnetapplyadvanced.apply_controlnet(
            strength=controlnet_strength_first,
            start_percent=0,
            end_percent=1,
            positive=get_value_at_index(controlnetapplyadvanced_11, 0),
            negative=get_value_at_index(controlnetapplyadvanced_11, 1),
            control_net=get_value_at_index(controlnetloader_12, 0),
            image=get_value_at_index(tilepreprocessor_14, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        # Use animation-enabled sampler if requested
        if animation_handler and enable_animation:
            # Run ksampler in thread to allow real-time image yielding
            result_container = [None]

            def run_ksampler():
                result_container[0] = ksampler_with_animation(
                    model=get_value_at_index(checkpointloadersimple_4, 0),
                    seed=seed,
                    steps=20,
                    cfg=7,
                    sampler_name="dpmpp_2m",
                    scheduler="karras",
                    positive=get_value_at_index(controlnetapplyadvanced_13, 0),
                    negative=get_value_at_index(controlnetapplyadvanced_13, 1),
                    latent_image=get_value_at_index(emptylatentimage_5, 0),
                    denoise=1,
                    animation_handler=animation_handler,
                    vae=get_value_at_index(checkpointloadersimple_4, 2),
                )

            ksampler_thread = threading.Thread(target=run_ksampler)
            ksampler_thread.start()

            # Yield intermediate images as they're captured
            while (
                ksampler_thread.is_alive() or not animation_handler.image_queue.empty()
            ):
                try:
                    img, msg = animation_handler.image_queue.get(timeout=0.1)
                    yield img, msg
                except queue.Empty:
                    pass

            ksampler_thread.join()
            ksampler_3 = result_container[0]
        else:
            ksampler_3 = ksampler.sample(
                seed=seed,
                steps=20,
                cfg=7,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(checkpointloadersimple_4, 0),
                positive=get_value_at_index(controlnetapplyadvanced_13, 0),
                negative=get_value_at_index(controlnetapplyadvanced_13, 1),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

        # Progress update after first sampling completes (no yield to avoid showing base QR)
        msg = "First pass sampling complete... decoding image"
        log_progress(msg, gr_progress, 0.4)
        # Removed yield here - caused flash of black/white QR before decoded image

        # Calculate optimal tile size for this image - disable for now
        # tile_size, overlap = calculate_vae_tile_size(image_size)

        # Small image, use standard decode (faster)
        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(ksampler_3, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        # 2) Yield the first decoded image as a second intermediate result
        mid_tensor = get_value_at_index(vaedecode_8, 0)
        mid_np = (mid_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        mid_np = mid_np[0]
        mid_pil = Image.fromarray(mid_np)
        msg = (
            f"First enhancement pass complete (step 2/{total_steps})… refining details"
        )
        log_progress(msg, gr_progress, 0.5)
        yield mid_pil, msg

        # Clear cache before second pass to free memory
        model_management.soft_empty_cache()

        # Simple stage update for second pass
        log_progress("Second pass (refinement)...", gr_progress, 0.5)

        controlnetapplyadvanced_20 = controlnetapplyadvanced.apply_controlnet(
            strength=controlnet_strength_final,
            start_percent=0,
            end_percent=1,
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            control_net=get_value_at_index(controlnetloader_19, 0),
            image=get_value_at_index(vaedecode_8, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        # Use animation-enabled sampler if requested
        if animation_handler and enable_animation:
            # Run ksampler in thread to allow real-time image yielding
            result_container = [None]

            def run_ksampler():
                result_container[0] = ksampler_with_animation(
                    model=get_value_at_index(checkpointloadersimple_4, 0),
                    seed=seed + 1,
                    steps=20,
                    cfg=7,
                    sampler_name="dpmpp_2m",
                    scheduler="karras",
                    positive=get_value_at_index(controlnetapplyadvanced_20, 0),
                    negative=get_value_at_index(controlnetapplyadvanced_20, 1),
                    latent_image=get_value_at_index(emptylatentimage_17, 0),
                    denoise=1,
                    animation_handler=animation_handler,
                    vae=get_value_at_index(checkpointloadersimple_4, 2),
                )

            ksampler_thread = threading.Thread(target=run_ksampler)
            ksampler_thread.start()

            # Yield intermediate images as they're captured
            while (
                ksampler_thread.is_alive() or not animation_handler.image_queue.empty()
            ):
                try:
                    img, msg = animation_handler.image_queue.get(timeout=0.1)
                    yield img, msg
                except queue.Empty:
                    pass

            ksampler_thread.join()
            ksampler_18 = result_container[0]
        else:
            ksampler_18 = ksampler.sample(
                seed=seed + 1,
                steps=20,
                cfg=7,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(checkpointloadersimple_4, 0),
                positive=get_value_at_index(controlnetapplyadvanced_20, 0),
                negative=get_value_at_index(controlnetapplyadvanced_20, 1),
                latent_image=get_value_at_index(emptylatentimage_17, 0),
            )

        # Progress update after second sampling completes (no yield to avoid showing first pass)
        msg = "Second pass sampling complete... decoding final image"
        log_progress(msg, gr_progress, 0.8)
        # Removed yield here - caused flash of first pass image before final decoded

        # Second pass is always 2x original, calculate based on doubled size
        tile_size_2x, overlap_2x = calculate_vae_tile_size(image_size * 2)

        if tile_size_2x is not None:
            vaedecode_21 = vaedecodetiled.decode(
                samples=get_value_at_index(ksampler_18, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
                tile_size=tile_size_2x,
                overlap=overlap_2x,
            )
        else:
            vaedecode_21 = vaedecode.decode(
                samples=get_value_at_index(ksampler_18, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

        # 3) Optionally upscale if enabled
        if enable_upscale:
            # Show pre-upscale result
            pre_upscale_tensor = get_value_at_index(vaedecode_21, 0)
            pre_upscale_np = (pre_upscale_tensor.detach().cpu().numpy() * 255).astype(
                np.uint8
            )
            pre_upscale_np = pre_upscale_np[0]
            pre_upscale_pil = Image.fromarray(pre_upscale_np)
            current_step = 3
            msg = f"Enhancement complete (step {current_step}/{total_steps})... upscaling image"
            log_progress(msg, gr_progress, 0.9)
            yield pre_upscale_pil, msg

            # Upscale the final image (load model on-demand)
            upscale_model = get_upscale_model()
            upscaled = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscale_model, 0),
                image=get_value_at_index(vaedecode_21, 0),
            )

            image_tensor = get_value_at_index(upscaled, 0)
            image_np = (image_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
            image_np = image_np[0]
            # Ensure RGB array shape to prevent MPS grayscale conversion bug
            if len(image_np.shape) == 2:
                # Convert grayscale (H, W) to RGB (H, W, 3)
                image_np = np.stack([image_np, image_np, image_np], axis=2)
            elif image_np.shape[2] == 1:
                # Convert (H, W, 1) to (H, W, 3)
                image_np = np.repeat(image_np, 3, axis=2)
            pil_image = Image.fromarray(image_np)
            current_step += 1

            # Apply color quantization if enabled
            if enable_color_quantization:
                msg = f"Upscaling complete (step {current_step}/{total_steps})... applying color quantization"
                log_progress(msg, gr_progress, 0.95)
                yield pil_image, msg

                pil_image = apply_color_quantization(
                    pil_image,
                    colors=[color_1, color_2, color_3, color_4],
                    num_colors=num_colors,
                    apply_gradients=apply_gradient_filter,
                    gradient_strength=gradient_strength,
                    variation_steps=variation_steps,
                )
                current_step += 1

            msg = f"No errors, all good! Final QR art generated and upscaled. (step {current_step}/{total_steps})"
            log_progress(msg, gr_progress, 1.0)
            yield (pil_image, msg)
            return  # Explicit return to cleanly exit generator
        else:
            # No upscaling
            image_tensor = get_value_at_index(vaedecode_21, 0)
            image_np = (image_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
            image_np = image_np[0]
            # Ensure RGB array shape to prevent MPS grayscale conversion bug
            if len(image_np.shape) == 2:
                # Convert grayscale (H, W) to RGB (H, W, 3)
                image_np = np.stack([image_np, image_np, image_np], axis=2)
            elif image_np.shape[2] == 1:
                # Convert (H, W, 1) to (H, W, 3)
                image_np = np.repeat(image_np, 3, axis=2)
            pil_image = Image.fromarray(image_np)
            current_step = 3

            # Apply color quantization if enabled
            if enable_color_quantization:
                msg = f"Enhancement complete (step {current_step}/{total_steps})... applying color quantization"
                log_progress(msg, gr_progress, 0.95)
                yield pil_image, msg

                pil_image = apply_color_quantization(
                    pil_image,
                    colors=[color_1, color_2, color_3, color_4],
                    num_colors=num_colors,
                    apply_gradients=apply_gradient_filter,
                    gradient_strength=gradient_strength,
                    variation_steps=variation_steps,
                )
                current_step += 1

            msg = f"No errors, all good! Final QR art generated. (step {current_step}/{total_steps})"
            log_progress(msg, gr_progress, 1.0)
            yield pil_image, msg
            return  # Explicit return to cleanly exit generator


def _pipeline_artistic(
    prompt: str,
    negative_prompt: str = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
    qr_text: str = "",
    input_type: str = "URL",
    image_size: int = 640,
    border_size: int = 4,
    error_correction: str = "Medium (15%)",
    module_size: int = 12,
    module_drawer: str = "Square",
    seed: int = 0,
    enable_upscale: bool = True,
    freeu_b1: float = 1.4,
    freeu_b2: float = 1.3,
    freeu_s1: float = 0.0,
    freeu_s2: float = 1.3,
    enable_sag: bool = True,
    sag_scale: float = 0.5,
    sag_blur_sigma: float = 0.5,
    controlnet_strength_first: float = 0.45,
    controlnet_strength_final: float = 0.7,
    enable_color_quantization: bool = False,
    num_colors: int = 4,
    color_1: str = "#000000",
    color_2: str = "#FFFFFF",
    color_3: str = "#FF0000",
    color_4: str = "#00FF00",
    apply_gradient_filter: bool = False,
    gradient_strength: float = 0.3,
    variation_steps: int = 5,
    enable_animation: bool = True,
    enable_cascade_filter: bool = False,
    cascade_blur_kernel: int = 15,
    cascade_threshold_ratio: float = 0.33,
    enable_detail_sharpening: bool = False,
    sharpening_radius: float = 2.0,
    sharpening_amount: float = 1.5,
    sharpening_threshold: int = 0,
    customize_tile_preprocessing: bool = False,
    tile_pyrup_iters: int = 3,
    gr_progress=None,
):
    # Initialize animation handler if enabled
    animation_handler = (
        AnimationHandler(preview_size=image_size) if enable_animation else None
    )
    if animation_handler:
        animation_handler.enabled = True

    # Generate QR code
    qr_protocol = "None" if input_type == "Plain Text" else "Https"

    try:
        comfy_qr = comfy_qr_by_module_size.generate_qr(
            protocol=qr_protocol,
            text=qr_text,
            module_size=module_size,
            max_image_size=image_size,
            fill_hexcolor="#000000",
            back_hexcolor="#FFFFFF",
            error_correction=error_correction,
            border=border_size,
            module_drawer=module_drawer,
        )
    except RuntimeError as e:
        error_msg = (
            f"Error generating QR code: {str(e)}\n"
            "Try with a shorter text, increase the image size, or decrease the border size, module size, and error correction level under Change Settings Manually."
        )
        yield None, error_msg
        return

    # Show the base QR code
    base_qr_tensor = get_value_at_index(comfy_qr, 0)
    base_qr_np = (base_qr_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    base_qr_np = base_qr_np[0]
    base_qr_pil = Image.fromarray(base_qr_np)

    # Calculate total steps based on border and upscale
    total_steps = 3  # Base: first pass, final refinement, final result
    if border_size > 0:
        total_steps += 1  # Add border noise step
    if enable_upscale:
        total_steps += 1  # Add upscale step

    current_step = 1

    # Only add noise if there's a border (border_size > 0)
    if border_size > 0:
        msg = f"Generated base QR pattern... adding QR-like cubics to border (step {current_step}/{total_steps})"
        log_progress(msg, gr_progress, 0.05)
        yield (base_qr_pil, msg)
        current_step += 1

        # Add QR-like cubic patterns ONLY to border region (extends QR structure into border)
        # Density automatically matches QR code interior density for natural transition
        qr_with_border_noise = add_noise_to_border_only(
            get_value_at_index(comfy_qr, 0),
            seed=seed + 100,
            border_size=border_size,
            image_size=image_size,
            module_size=module_size,  # Use same module size as QR code
        )

        # Show the noisy QR so you can see the border cubic pattern effect
        noisy_qr_np = (qr_with_border_noise.detach().cpu().numpy() * 255).astype(
            np.uint8
        )
        noisy_qr_np = noisy_qr_np[0]
        noisy_qr_pil = Image.fromarray(noisy_qr_np)
        msg = f"Added QR-like cubics to border... enhancing with AI (step {current_step}/{total_steps})"
        log_progress(msg, gr_progress, 0.1)
        yield (noisy_qr_pil, msg)
        current_step += 1
    else:
        # No border, skip noise
        qr_with_border_noise = get_value_at_index(comfy_qr, 0)
        msg = f"Generated base QR pattern (no border)... enhancing with AI (step {current_step}/{total_steps})"
        log_progress(msg, gr_progress, 0.1)
        yield (base_qr_pil, msg)
        current_step += 1

    # Generate latent image
    latent_image = emptylatentimage.generate(
        width=image_size, height=image_size, batch_size=1
    )

    # Encode text prompts
    positive_prompt = cliptextencode.encode(
        text=prompt,
        clip=get_value_at_index(checkpointloadersimple_artistic, 1),
    )

    negative_prompt_encoded = cliptextencode.encode(
        text=negative_prompt,
        clip=get_value_at_index(checkpointloadersimple_artistic, 1),
    )

    # Load controlnets
    brightness_controlnet = controlnetloader.load_controlnet(
        control_net_name="models/control_v1p_sd15_brightness.safetensors"
    )

    tile_controlnet = controlnetloader.load_controlnet(
        control_net_name="control_v11f1e_sd15_tile_fp16.safetensors"
    )

    # Apply Stable Cascade filter if enabled
    if enable_cascade_filter:
        qr_for_brightness = apply_stable_cascade_qr_filter(
            qr_with_border_noise,
            blur_kernel=cascade_blur_kernel,
            threshold_ratio=cascade_threshold_ratio,
        )
    else:
        qr_for_brightness = qr_with_border_noise

    # First ControlNet pass (using filtered or raw QR with border cubics)
    controlnet_apply = controlnetapplyadvanced.apply_controlnet(
        strength=controlnet_strength_first,
        start_percent=0,
        end_percent=1,
        positive=get_value_at_index(positive_prompt, 0),
        negative=get_value_at_index(negative_prompt_encoded, 0),
        control_net=get_value_at_index(brightness_controlnet, 0),
        image=qr_for_brightness,
        vae=get_value_at_index(checkpointloadersimple_artistic, 2),
    )

    # Tile preprocessor (using filtered or raw QR with border cubics)
    # Use custom pyrUp_iters if enabled, otherwise default to 3
    actual_pyrup_iters = tile_pyrup_iters if customize_tile_preprocessing else 3

    tile_processed = tilepreprocessor.execute(
        pyrUp_iters=actual_pyrup_iters,
        resolution=image_size,
        image=qr_for_brightness,
    )

    # Second ControlNet pass (using tile processed from filtered/raw QR)
    controlnet_apply = controlnetapplyadvanced.apply_controlnet(
        strength=controlnet_strength_first,
        start_percent=0,
        end_percent=1,
        positive=get_value_at_index(controlnet_apply, 0),
        negative=get_value_at_index(controlnet_apply, 1),
        control_net=get_value_at_index(tile_controlnet, 0),
        image=get_value_at_index(tile_processed, 0),
        vae=get_value_at_index(checkpointloadersimple_artistic, 2),
    )

    # Apply FreeU_V2 for enhanced quality (better detail, texture, and cleaner output)
    base_model = get_value_at_index(checkpointloadersimple_artistic, 0)

    freeu = FreeU_V2()
    freeu_model = freeu.patch(
        model=base_model,
        b1=freeu_b1,  # Backbone feature enhancement - customizable
        b2=freeu_b2,  # Backbone feature enhancement (layer 2) - customizable
        s1=freeu_s1,  # Skip connection dampening - customizable structure hiding
        s2=freeu_s2,  # Skip connection dampening (layer 2) - customizable scannability balance
    )[0]

    # Apply SAG (Self-Attention Guidance) for improved structural coherence (if enabled)
    if enable_sag:
        smoothed_energy = NODE_CLASS_MAPPINGS["SelfAttentionGuidance"]()
        enhanced_model = smoothed_energy.patch(
            model=freeu_model,
            scale=sag_scale,  # SAG guidance scale - customizable
            blur_sigma=sag_blur_sigma,  # Blur amount - customizable artistic blending
        )[0]
    else:
        enhanced_model = freeu_model

    # First sampling pass
    log_progress("First pass - artistic sampling...", gr_progress, 0.2)

    # Use animation-enabled sampler if requested
    if animation_handler and enable_animation:
        # Run ksampler in thread to allow real-time image yielding
        result_container = [None]

        def run_ksampler():
            result_container[0] = ksampler_with_animation(
                model=enhanced_model,  # Using FreeU + SAG enhanced model
                seed=seed,
                steps=30,
                cfg=7,
                sampler_name="dpmpp_3m_sde",
                scheduler="karras",
                positive=get_value_at_index(controlnet_apply, 0),
                negative=get_value_at_index(controlnet_apply, 1),
                latent_image=get_value_at_index(latent_image, 0),
                denoise=1,
                animation_handler=animation_handler,
                vae=get_value_at_index(checkpointloadersimple_artistic, 2),
            )

        ksampler_thread = threading.Thread(target=run_ksampler)
        ksampler_thread.start()

        # Yield intermediate images as they're captured
        while ksampler_thread.is_alive() or not animation_handler.image_queue.empty():
            try:
                img, msg = animation_handler.image_queue.get(timeout=0.1)
                yield (img, msg)
            except queue.Empty:
                pass

        ksampler_thread.join()
        samples = result_container[0]
    else:
        samples = ksampler.sample(
            seed=seed,
            steps=30,
            cfg=7,
            sampler_name="dpmpp_3m_sde",
            scheduler="karras",
            denoise=1,
            model=enhanced_model,  # Using FreeU + SAG enhanced model
            positive=get_value_at_index(controlnet_apply, 0),
            negative=get_value_at_index(controlnet_apply, 1),
            latent_image=get_value_at_index(latent_image, 0),
        )

    # Progress update after first sampling completes (no yield to avoid showing old QR)
    msg = f"First pass sampling complete... decoding image (step {current_step}/{total_steps})"
    log_progress(msg, gr_progress, 0.4)
    # Removed yield here - caused flash of old QR before decoded image

    # First decode with dynamic tiling - disable for now
    # tile_size, overlap = calculate_vae_tile_size(image_size)

    decoded = vaedecode.decode(
        samples=get_value_at_index(samples, 0),
        vae=get_value_at_index(checkpointloadersimple_artistic, 2),
    )

    # Show first pass result
    first_pass_tensor = get_value_at_index(decoded, 0)
    first_pass_np = (first_pass_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    first_pass_np = first_pass_np[0]
    first_pass_pil = Image.fromarray(first_pass_np)

    # Apply detail sharpening if enabled (experimental feature)
    # This preserves QR code edge sharpness for the second pass
    if enable_detail_sharpening:
        first_pass_pil = apply_detail_sharpening(
            first_pass_pil,
            radius=sharpening_radius,
            amount=sharpening_amount,
            threshold=sharpening_threshold,
        )

    msg = f"First enhancement pass complete (step {current_step}/{total_steps})... final refinement pass"
    log_progress(msg, gr_progress, 0.5)
    yield (first_pass_pil, msg)
    current_step += 1

    # Clear cache before second pass to free memory
    model_management.soft_empty_cache()

    # Final ControlNet pass (second pass - refinement)
    controlnet_apply_final = controlnetapplyadvanced.apply_controlnet(
        strength=controlnet_strength_final,
        start_percent=0,
        end_percent=1,
        positive=get_value_at_index(positive_prompt, 0),
        negative=get_value_at_index(negative_prompt_encoded, 0),
        control_net=get_value_at_index(tile_controlnet, 0),
        image=get_value_at_index(decoded, 0),
        vae=get_value_at_index(checkpointloadersimple_artistic, 2),
    )

    # Upscale latent
    upscaled_latent = latentupscaleby.upscale(
        upscale_method="area",
        scale_by=2.0,
        samples=get_value_at_index(samples, 0),
    )

    # Final sampling pass
    log_progress("Second pass (refinement)...", gr_progress, 0.6)

    # MPS device workaround: Recreate enhanced model for second pass to avoid device placement issues
    # After the first threaded sampling pass, some model weights can end up on CPU instead of MPS
    # This happens due to threading interaction with MPS backend + SAG making additional model calls
    if torch.backends.mps.is_available():
        # Recreate FreeU enhanced model from base model
        freeu_model_second = freeu.patch(
            model=base_model,
            b1=freeu_b1,
            b2=freeu_b2,
            s1=freeu_s1,
            s2=freeu_s2,
        )[0]

        # Reapply SAG if enabled
        if enable_sag:
            smoothed_energy_second = NODE_CLASS_MAPPINGS["SelfAttentionGuidance"]()
            enhanced_model_second = smoothed_energy_second.patch(
                model=freeu_model_second,
                scale=sag_scale,
                blur_sigma=sag_blur_sigma,
            )[0]
        else:
            enhanced_model_second = freeu_model_second
    else:
        # On non-MPS devices, reuse the same enhanced model
        enhanced_model_second = enhanced_model

    # Use animation-enabled sampler if requested
    if animation_handler and enable_animation:
        # Run ksampler in thread to allow real-time image yielding
        result_container = [None]

        def run_ksampler():
            result_container[0] = ksampler_with_animation(
                model=enhanced_model_second,  # Using recreated FreeU + SAG enhanced model (MPS fix)
                seed=seed + 1,
                steps=30,
                cfg=7,
                sampler_name="dpmpp_3m_sde",
                scheduler="karras",
                positive=get_value_at_index(controlnet_apply_final, 0),
                negative=get_value_at_index(controlnet_apply_final, 1),
                latent_image=get_value_at_index(upscaled_latent, 0),
                denoise=0.8,
                animation_handler=animation_handler,
                vae=get_value_at_index(checkpointloadersimple_artistic, 2),
            )

        ksampler_thread = threading.Thread(target=run_ksampler)
        ksampler_thread.start()

        # Yield intermediate images as they're captured
        while ksampler_thread.is_alive() or not animation_handler.image_queue.empty():
            try:
                img, msg = animation_handler.image_queue.get(timeout=0.1)
                yield (img, msg)
            except queue.Empty:
                pass

        ksampler_thread.join()
        final_samples = result_container[0]
    else:
        final_samples = ksampler.sample(
            seed=seed + 1,
            steps=30,
            cfg=7,
            sampler_name="dpmpp_3m_sde",
            scheduler="karras",
            denoise=0.8,
            model=enhanced_model_second,  # Using recreated FreeU + SAG enhanced model (MPS fix)
            positive=get_value_at_index(controlnet_apply_final, 0),
            negative=get_value_at_index(controlnet_apply_final, 1),
            latent_image=get_value_at_index(upscaled_latent, 0),
        )

    # Progress update after second sampling completes (no yield to avoid showing first pass)
    msg = f"Second pass sampling complete... decoding final image (step {current_step}/{total_steps})"
    log_progress(msg, gr_progress, 0.8)
    # Removed yield here - caused flash of first pass image before final decoded

    # Final decode with dynamic tiling
    tile_size, overlap = calculate_vae_tile_size(image_size)

    if tile_size is not None:
        final_decoded = vaedecodetiled.decode(
            samples=get_value_at_index(final_samples, 0),
            vae=get_value_at_index(checkpointloadersimple_artistic, 2),
            tile_size=tile_size,
            overlap=overlap,
        )
    else:
        final_decoded = vaedecode.decode(
            samples=get_value_at_index(final_samples, 0),
            vae=get_value_at_index(checkpointloadersimple_artistic, 2),
        )

    # Optionally upscale if enabled
    if enable_upscale:
        # Show result before upscaling
        pre_upscale_tensor = get_value_at_index(final_decoded, 0)
        pre_upscale_np = (pre_upscale_tensor.detach().cpu().numpy() * 255).astype(
            np.uint8
        )
        pre_upscale_np = pre_upscale_np[0]
        pre_upscale_pil = Image.fromarray(pre_upscale_np)
        msg = f"Final refinement complete (step {current_step}/{total_steps})... upscaling image"
        log_progress(msg, gr_progress, 0.9)
        yield (pre_upscale_pil, msg)
        current_step += 1

        # Upscale image with model (load model on-demand)
        upscale_model = get_upscale_model()
        upscaled = imageupscalewithmodel.upscale(
            upscale_model=get_value_at_index(upscale_model, 0),
            image=get_value_at_index(final_decoded, 0),
        )

        # Convert upscaled image to PIL Image and return
        image_tensor = get_value_at_index(upscaled, 0)
        image_np = (image_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        image_np = image_np[0]
        # Ensure RGB array shape to prevent MPS grayscale conversion bug
        if len(image_np.shape) == 2:
            # Convert grayscale (H, W) to RGB (H, W, 3)
            image_np = np.stack([image_np, image_np, image_np], axis=2)
        elif image_np.shape[2] == 1:
            # Convert (H, W, 1) to (H, W, 3)
            image_np = np.repeat(image_np, 3, axis=2)
        final_image = Image.fromarray(image_np)

        # Apply color quantization if enabled
        if enable_color_quantization:
            final_image = apply_color_quantization(
                final_image,
                colors=[color_1, color_2, color_3, color_4],
                num_colors=num_colors,
                apply_gradients=apply_gradient_filter,
                gradient_strength=gradient_strength,
                variation_steps=variation_steps,
            )

        msg = f"No errors, all good! Final artistic QR code generated and upscaled. (step {current_step}/{total_steps})"
        log_progress(msg, gr_progress, 1.0)
        yield (final_image, msg)
        return  # Explicit return to cleanly exit generator
    else:
        # No upscaling
        image_tensor = get_value_at_index(final_decoded, 0)
        image_np = (image_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        image_np = image_np[0]
        # Ensure RGB array shape to prevent MPS grayscale conversion bug
        if len(image_np.shape) == 2:
            # Convert grayscale (H, W) to RGB (H, W, 3)
            image_np = np.stack([image_np, image_np, image_np], axis=2)
        elif image_np.shape[2] == 1:
            # Convert (H, W, 1) to (H, W, 3)
            image_np = np.repeat(image_np, 3, axis=2)
        final_image = Image.fromarray(image_np)

        # Apply color quantization if enabled
        if enable_color_quantization:
            final_image = apply_color_quantization(
                final_image,
                colors=[color_1, color_2, color_3, color_4],
                num_colors=num_colors,
                apply_gradients=apply_gradient_filter,
                gradient_strength=gradient_strength,
                variation_steps=variation_steps,
            )

        msg = f"No errors, all good! Final artistic QR code generated. (step {current_step}/{total_steps})"
        log_progress(msg, gr_progress, 1.0)
        yield (final_image, msg)
        return  # Explicit return to cleanly exit generator


# Define artistic examples data (at module level for hot reload)
ARTISTIC_EXAMPLES = [
    {
        "image": "examples/artistic/sunset_mountains.jpg",
        "label": "Sunset Mountains",
        "prompt": "a beautiful sunset over mountains, photorealistic, detailed landscape, golden hour, dramatic lighting, 8k, ultra detailed",
        "text_input": "https://github.com",
        "input_type": "URL",
        "image_size": 704,
        "border_size": 6,
        "error_correction": "Medium (15%)",
        "module_size": 14,
        "module_drawer": "Square",
        "use_custom_seed": True,
        "seed": 718313,
        "sag_blur_sigma": 0.5,
    },
    {
        "image": "examples/artistic/japanese_temple.jpg",
        "label": "Japanese Temple",
        "prompt": "some clothes spread on ropes, Japanese girl sits inside in the middle of the image, few sakura flowers, realistic, great details, out in the open air sunny day realistic, great details, absence of people, Detailed and Intricate, CGI, Photoshoot, rim light, 8k, 16k, ultra detail",
        "text_input": "https://www.google.com",
        "input_type": "URL",
        "image_size": 640,
        "border_size": 6,
        "error_correction": "Medium (15%)",
        "module_size": 14,
        "module_drawer": "Square",
        "use_custom_seed": True,
        "seed": 718313,
        "sag_blur_sigma": 0.5,
    },
    {
        "image": "examples/artistic/roman_city.jpg",
        "label": "Roman City",
        "prompt": "aerial bird view of ancient Roman city, cobblestone streets and pathways forming intricate patterns, vintage illustration style, sepia tones, aged parchment look, detailed architecture, 8k, ultra detailed",
        "text_input": "WIFI:T:WPA;S:MyNetwork;P:MyPassword123;;",
        "input_type": "Plain Text",
        "image_size": 832,
        "border_size": 6,
        "error_correction": "High (30%)",
        "module_size": 16,
        "module_drawer": "Square",
        "use_custom_seed": True,
        "seed": 718313,
        "sag_blur_sigma": 0.5,
    },
    {
        "image": "examples/artistic/neapolitan_pizza.webp",
        "label": "Restaurant Brand",
        "prompt": "artisan boutique restaurant branding, neapolitan pizza on rustic wooden board, premium food editorial, warm ambient lighting, refined composition, photorealistic, high detail",
        "text_input": "https://www.instagram.com",
        "input_type": "URL",
        "image_size": 704,
        "border_size": 6,
        "error_correction": "Medium (15%)",
        "module_size": 14,
        "module_drawer": "Square",
        "use_custom_seed": True,
        "seed": 856749,
        "sag_blur_sigma": 2.0,
    },
    {
        "image": "examples/artistic/poker_chips.webp",
        "label": "Poker Chips",
        "prompt": "some cards on poker tale, realistic, great details, realistic, great details,absence of people, Detailed and Intricate, CGI, Photoshoot,rim light, 8k, 16k, ultra detail",
        "text_input": "https://store.steampowered.com",
        "input_type": "URL",
        "image_size": 768,
        "border_size": 6,
        "error_correction": "High (30%)",
        "module_size": 16,
        "module_drawer": "Square",
        "use_custom_seed": True,
        "seed": 718313,
        "sag_blur_sigma": 1.5,
    },
    {
        "image": "examples/artistic/underwater_fish.webp",
        "label": "Underwater Fish",
        "prompt": "underwater scene with tropical fish, coral reef, rays of sunlight penetrating water, vibrant colors, detailed marine life, photorealistic, 8k, ultra detailed",
        "text_input": "https://www.reddit.com",
        "input_type": "URL",
        "image_size": 704,
        "border_size": 6,
        "error_correction": "High (30%)",
        "module_size": 16,
        "module_drawer": "Square",
        "use_custom_seed": True,
        "seed": 3048334933,
        "sag_blur_sigma": 1.5,
    },
    {
        "image": "examples/artistic/mediterranean_garden.jpg",
        "label": "Boutique Venue",
        "prompt": "luxury boutique venue courtyard, mediterranean garden, olive trees, elegant stone textures, warm afternoon light, refined lifestyle branding, photorealistic, high detail",
        "text_input": "https://www.linkedin.com",
        "input_type": "URL",
        "image_size": 704,
        "border_size": 6,
        "error_correction": "Medium (15%)",
        "module_size": 14,
        "module_drawer": "Square",
        "use_custom_seed": True,
        "seed": 413468,
        "sag_blur_sigma": 0.5,
    },
    {
        "image": "examples/artistic/rice_fields.jpg",
        "label": "Rice Fields",
        "prompt": "aerial view of terraced rice fields on mountainside, winding pathways between green paddies, Asian countryside, bird's eye perspective, detailed landscape, golden hour lighting, photorealistic, 8k, ultra detailed",
        "text_input": "geo:37.7749,-122.4194",
        "input_type": "Plain Text",
        "image_size": 704,
        "border_size": 6,
        "error_correction": "High (30%)",
        "module_size": 16,
        "module_drawer": "Square",
        "use_custom_seed": True,
        "seed": 962359,
        "sag_blur_sigma": 0.5,
    },
    {
        "image": "examples/artistic/cyberpunk_city.webp",
        "label": "Cyberpunk City",
        "prompt": "futuristic cityscape with flying cars and neon lights, cyberpunk style, detailed architecture, night scene, 8k, ultra detailed",
        "text_input": "https://linkedin.com",
        "input_type": "URL",
        "image_size": 704,
        "border_size": 6,
        "error_correction": "High (30%)",
        "module_size": 16,
        "module_drawer": "Square",
        "use_custom_seed": True,
        "seed": 718313,
        "sag_blur_sigma": 1.5,
    },
]

STANDARD_EXAMPLES = [
    [
        "some clothes spread on ropes, realistic, great details, out in the open air sunny day realistic, great details,absence of people, Detailed and Intricate, CGI, Photoshoot,rim light, 8k, 16k, ultra detail",
        "https://www.google.com",
        "URL",
        512,
        4,
        "Medium (15%)",
        12,
        "Square",
    ],
    [
        "some cards on poker tale, realistic, great details, realistic, great details,absence of people, Detailed and Intricate, CGI, Photoshoot,rim light, 8k, 16k, ultra detail",
        "https://store.steampowered.com",
        "URL",
        512,
        4,
        "Medium (15%)",
        12,
        "Square",
    ],
    [
        "a beautiful sunset over mountains, photorealistic, detailed landscape, golden hour, dramatic lighting, 8k, ultra detailed",
        "https://github.com",
        "URL",
        512,
        4,
        "Medium (15%)",
        12,
        "Square",
    ],
    [
        "underwater scene with coral reef and tropical fish, photorealistic, detailed, crystal clear water, sunlight rays, 8k, ultra detailed",
        "https://twitter.com",
        "URL",
        512,
        4,
        "Medium (15%)",
        12,
        "Square",
    ],
    [
        "futuristic cityscape with flying cars and neon lights, cyberpunk style, detailed architecture, night scene, 8k, ultra detailed",
        "https://linkedin.com",
        "URL",
        512,
        4,
        "Medium (15%)",
        12,
        "Square",
    ],
    [
        "vintage camera on wooden table, photorealistic, detailed textures, soft lighting, bokeh background, 8k, ultra detailed",
        "https://instagram.com",
        "URL",
        512,
        4,
        "Medium (15%)",
        12,
        "Square",
    ],
    [
        "business card design, professional, modern, clean layout, corporate style, detailed, 8k, ultra detailed",
        "BEGIN:VCARD\nVERSION:3.0\nFN:John Doe\nORG:Acme Corporation\nTITLE:Software Engineer\nTEL:+1-555-123-4567\nEMAIL:john.doe@example.com\nEND:VCARD",
        "Plain Text",
        832,
        4,
        "Medium (15%)",
        12,
        "Square",
    ],
    [
        "wifi network symbol, modern tech, digital art, glowing blue, detailed, 8k, ultra detailed",
        "WIFI:T:WPA;S:MyNetwork;P:MyPassword123;;",
        "Plain Text",
        576,
        4,
        "Medium (15%)",
        12,
        "Square",
    ],
    [
        "calendar appointment reminder, organized planner, professional office, detailed, 8k, ultra detailed",
        "BEGIN:VEVENT\nSUMMARY:Team Meeting\nDTSTART:20251115T140000Z\nDTEND:20251115T150000Z\nLOCATION:Conference Room A\nEND:VEVENT",
        "Plain Text",
        832,
        4,
        "Medium (15%)",
        12,
        "Square",
    ],
    [
        "location pin on map, travel destination, scenic view, detailed cartography, 8k, ultra detailed",
        "geo:37.7749,-122.4194",
        "Plain Text",
        512,
        4,
        "Medium (15%)",
        12,
        "Square",
    ],
]

# Start your Gradio app with automatic cache cleanup (at module level for hot reload)
# delete_cache=(3600, 3600) means: check every hour and delete files older than 1 hour
with gr.Blocks(delete_cache=(3600, 3600)) as demo:
    # Add a title and description
    gr.Markdown("# QR Code Art Generator")
    gr.Markdown("""
        AI-powered QR code generator with two pipelines: **Artistic** (creative, photorealistic) and **Standard** (fast, reliable).

        **Privacy:** Generated images auto-delete after 1 hour. Download promptly!

        **GPU Quota:**
        - **Unauthenticated**: 120s daily (~1 generation at 1024px or ~6 at 512px)
        - **Authenticated**: 210s daily (~10 artistic generations at 512px)
        - **Tip**: Use Standard pipeline (2x faster) to save quota

        **Zero GPU Error?** If you see "Zero GPU" error, you've run out of quota. Options:
        - Wait until tomorrow for quota reset
        - Register a Hugging Face account for more generations
        - Subscribe to PRO for even more generations

        **Tip:** URL shortener is the best way to keep QR codes scannable without burning extra GPU quota on larger image sizes. If your link is longer than ~10 characters, consider enabling the URL shortener. If you keep the original URL and it grows beyond ~38 characters, use an image size above 704 for better results.

        Choose a tab below to get started!
        """)

    gr.Markdown(
        "Note: You can opt in to help improve the product by sharing anonymous usage data. Generated images are not used for analytics and are automatically deleted after 1 hour."
    )
    gr.Markdown(
        "Note: You can also opt in to a temporary `qrcut.co` URL shortener when using URL mode. This is usually the best way to get cleaner QR codes for longer links. Short links expire if nobody opens the QR code for 7 days."
    )
    analytics_opt_in_global = gr.Checkbox(
        label="Share anonymous usage data to improve QR quality",
        value=ANALYTICS_DEFAULT_OPT_IN,
    )

    # Add tabs for different generation methods
    with gr.Tabs():
        # ARTISTIC QR TAB
        with gr.TabItem("Artistic QR"):
            # Short description
            gr.Markdown("""
                🎨 **Create artistic QR codes that blend seamlessly with your creative vision**

                ⚡ **Advanced controls** for perfect balance between scannability and aesthetics

                💡 **More creative and photorealistic** than Standard pipeline
                """)

            # Full documentation in collapsed accordion
            with gr.Accordion("📖 Full Documentation & Tips", open=False):
                gr.Markdown("""
                    ### About Artistic QR Pipeline
                    The Artistic pipeline creates highly creative, photorealistic QR codes. This pipeline offers:
                    - More artistic freedom and creative results
                    - Optional upscaling with RealESRGAN
                    - FreeU and SAG (Self-Attention Guidance) for enhanced quality
                    - Customizable ControlNet strength for balancing art vs scannability

                    ### Tips for Best Results:
                    - **Prompts**: Use detailed descriptions with style keywords ('photorealistic', 'detailed', '8k', '16k')
                    - **Input Mode**: Choose **URL** for web links or **Plain Text** for VCARD, WiFi, calendars, etc.
                    - **Animation** (enabled by default): Shows intermediate steps. Disable to save ~20% GPU time
                    - **Color Quantization**: Apply custom brand colors (2-4 color palette with optional gradients)
                    - **Upscaling**: Enhances output quality but uses more GPU quota - disabled by default

                    ### GPU Usage:
                    - Default settings (512px): ~20 seconds per generation
                    - With upscaling: ~40-60 seconds
                    - Large images (832px+): Always disable upscaling to conserve quota

                    ### Sharing Settings:
                    After generation, copy the JSON settings that appear below your image to reproduce exact results or share with others using "Import Settings from JSON"
                    """)

            with gr.Row():
                with gr.Column():
                    # Add input type selector for artistic QR
                    artistic_input_type = gr.Radio(
                        choices=["URL", "Plain Text"],
                        value="URL",
                        label="Input Type",
                        info="URL: For web links (auto-removes https://). Plain Text: For VCARD, WiFi, calendar, location, etc. (no manipulation)",
                    )

                    # Add inputs for artistic QR
                    artistic_prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate (check examples below for inspiration)",
                        value="a beautiful sunset over mountains, photorealistic, detailed landscape, golden hour, dramatic lighting, 8k, ultra detailed",
                        lines=3,
                    )
                    artistic_text_input = gr.Textbox(
                        label="QR Code Content",
                        placeholder="Enter URL or plain text",
                        value="https://github.com",
                        lines=3,
                        info="URL mode automatically removes common tracking params like utm_*, fbclid, and gclid before QR generation.",
                    )
                    artistic_use_temporary_short_link = gr.Checkbox(
                        label="Use URL shortener",
                        value=False,
                        info="URL mode only. Best option for longer links when you want cleaner QR codes without raising image size too much. The short link expires if nobody opens the QR code for 7 days.",
                    )

                    # Import Settings section - separate accordion
                    with gr.Accordion("Import Settings from JSON", open=False):
                        gr.Markdown(
                            "Paste a settings JSON string (copied from a previous generation) to load all parameters at once."
                        )
                        import_json_input_artistic = gr.Textbox(
                            label="Paste Settings JSON",
                            placeholder='{"pipeline": "artistic", "prompt": "...", "seed": 718313, ...}',
                            lines=3,
                        )
                        import_status_artistic = gr.Textbox(
                            label="Import Status",
                            interactive=False,
                            visible=False,
                            lines=2,
                        )
                        with gr.Row():
                            load_settings_btn_artistic = gr.Button(
                                "Load Settings", variant="primary"
                            )
                            clear_json_btn_artistic = gr.Button(
                                "Clear", variant="secondary"
                            )

                    # Change Settings Manually - separate accordion
                    with gr.Accordion("Change Settings Manually", open=False):
                        gr.Markdown(
                            "**Advanced controls including:** Animation toggle, Color Quantization, FreeU/SAG parameters, ControlNet strength, QR settings, and more."
                        )
                        # Negative Prompt
                        negative_prompt_artistic = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Describe what you don't want in the image",
                            value="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
                            lines=2,
                            info="Keywords to avoid in the generated image",
                        )

                        # Add image size slider for artistic QR
                        artistic_image_size = gr.Slider(
                            minimum=512,
                            maximum=1024,
                            step=64,
                            value=704,
                            label="Image Size",
                            info="Base size of the generated image. Final output will be 2x this size (e.g., 640 → 1280) due to the two-step enhancement process. Higher values use more VRAM and take longer to process.",
                        )

                        # Add border size slider for artistic QR
                        artistic_border_size = gr.Slider(
                            minimum=0,
                            maximum=8,
                            step=1,
                            value=6,
                            label="QR Code Border Size",
                            info="Number of modules (squares) to use as border around the QR code. Higher values add more whitespace.",
                        )

                        # Add error correction dropdown for artistic QR
                        artistic_error_correction = gr.Dropdown(
                            choices=[
                                "Low (7%)",
                                "Medium (15%)",
                                "Quartile (25%)",
                                "High (30%)",
                            ],
                            value="Medium (15%)",
                            label="Error Correction Level",
                            info="Higher error correction makes the QR code more scannable when damaged or obscured, but increases its size and complexity. High (30%) is recommended for artistic QR codes.",
                        )

                        # Add module size slider for artistic QR
                        artistic_module_size = gr.Slider(
                            minimum=4,
                            maximum=16,
                            step=1,
                            value=14,
                            label="QR Module Size",
                            info="Pixel width of the smallest QR code unit. Larger values improve readability but require a larger image size. 14 is a good starting point.",
                        )

                        # Add module drawer dropdown with style examples for artistic QR
                        artistic_module_drawer = gr.Dropdown(
                            choices=[
                                "Square",
                                "Gapped square",
                                "Circle",
                                "Rounded",
                                "Vertical bars",
                                "Horizontal bars",
                            ],
                            value="Square",
                            label="QR Code Style",
                            info="Select the style of the QR code modules (squares). See examples below. Different styles can give your QR code a unique look while maintaining scannability.",
                        )

                        # Add style examples with labels
                        gr.Markdown("### Style Examples:")

                        # First row of examples
                        with gr.Row():
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("**Square**", show_label=False)
                                gr.Image(
                                    "custom_nodes/ComfyQR/img/square.png",
                                    width=100,
                                    show_label=False,
                                    show_download_button=False,
                                )
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("**Gapped Square**", show_label=False)
                                gr.Image(
                                    "custom_nodes/ComfyQR/img/gapped_square.png",
                                    width=100,
                                    show_label=False,
                                    show_download_button=False,
                                )
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("**Circle**", show_label=False)
                                gr.Image(
                                    "custom_nodes/ComfyQR/img/circle.png",
                                    width=100,
                                    show_label=False,
                                    show_download_button=False,
                                )

                        # Second row of examples
                        with gr.Row():
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("**Rounded**", show_label=False)
                                gr.Image(
                                    "custom_nodes/ComfyQR/img/rounded.png",
                                    width=100,
                                    show_label=False,
                                    show_download_button=False,
                                )
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("**Vertical Bars**", show_label=False)
                                gr.Image(
                                    "custom_nodes/ComfyQR/img/vertical-bars.png",
                                    width=100,
                                    show_label=False,
                                    show_download_button=False,
                                )
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("**Horizontal Bars**", show_label=False)
                                gr.Image(
                                    "custom_nodes/ComfyQR/img/horizontal-bars.png",
                                    width=100,
                                    show_label=False,
                                    show_download_button=False,
                                )

                        # Add upscale checkbox
                        artistic_enable_upscale = gr.Checkbox(
                            label="Enable Upscaling",
                            value=False,
                            info="Enable upscaling with RealESRGAN for higher quality output (disabled by default to reduce GPU time)",
                        )

                        # Animation toggle
                        artistic_enable_animation = gr.Checkbox(
                            label="Enable Animation (Show KSampler Progress)",
                            value=True,
                            info="Shows intermediate images every 5 steps during generation. Disable for faster generation.",
                        )

                        # Experimental Settings Section
                        with gr.Accordion(
                            "⚙️ Experimental Settings (Advanced Users Only)", open=False
                        ):
                            gr.Markdown("""
                            ⚠️ **Warning:** These features are experimental and may affect generation quality, 
                            scannability, or processing time. Only enable if you understand their effects.
                            
                            These settings allow fine-tuning of the artistic pipeline for advanced users who want 
                            more control over detail preservation, QR preprocessing, and enhancement strategies.
                            
                            **Recommendation:** Start with defaults, then enable one feature at a time to understand 
                            its impact on your specific use case.
                            """)

                            # Section 1: Stable Cascade QR Filter
                            gr.Markdown("### 🔹 Stable Cascade QR Filter")
                            gr.Markdown(
                                "Advanced preprocessing filter using HSV color space, Gaussian blur, and "
                                "adaptive thresholding. Based on Stable Cascade implementation."
                            )

                            enable_cascade_filter_artistic = gr.Checkbox(
                                label="Enable Stable Cascade QR Filter",
                                value=False,
                                info="Apply HSV-based brightness filter to QR code before ControlNet",
                            )

                            # Advanced Cascade parameters (nested accordion, visible only when filter enabled)
                            cascade_advanced_accordion = gr.Accordion(
                                "Advanced Cascade Parameters", open=False, visible=False
                            )

                            with cascade_advanced_accordion:
                                gr.Markdown(
                                    "Fine-tune filter behavior. Higher blur = smoother, higher threshold = sharper cutoff."
                                )

                                cascade_blur_kernel_artistic = gr.Slider(
                                    minimum=5,
                                    maximum=35,
                                    step=2,
                                    value=15,
                                    label="Blur Kernel Size",
                                    info="Gaussian blur kernel size (must be odd). Default: 15",
                                )

                                cascade_threshold_ratio_artistic = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.5,
                                    step=0.05,
                                    value=0.33,
                                    label="Threshold Ratio",
                                    info="Adaptive threshold ratio. Default: 0.33",
                                )

                            # Show/hide advanced parameters when filter is toggled
                            enable_cascade_filter_artistic.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[enable_cascade_filter_artistic],
                                outputs=[cascade_advanced_accordion],
                            )

                            # Section 2: Detail Sharpening
                            gr.Markdown("### 🔹 Detail Sharpening")
                            gr.Markdown(
                                "Apply unsharp mask between first and second pass to preserve QR code details. "
                                "Allows lower first-pass ControlNet strength (e.g., 0.35-0.40) while maintaining scannability."
                            )

                            enable_detail_sharpening_artistic = gr.Checkbox(
                                label="Enable Detail Sharpening",
                                value=False,
                                info="Sharpen first-pass output before second pass to preserve QR edges",
                            )

                            # Sharpening parameters (nested accordion, visible only when enabled)
                            sharpening_params_accordion = gr.Accordion(
                                "Sharpening Parameters", open=False, visible=False
                            )

                            with sharpening_params_accordion:
                                gr.Markdown(
                                    "Adjust sharpening behavior. Start with defaults, increase amount for stronger effect."
                                )

                                sharpening_radius_artistic = gr.Slider(
                                    minimum=1.0,
                                    maximum=5.0,
                                    step=0.5,
                                    value=2.0,
                                    label="Sharpening Radius",
                                    info="Sharpening width in pixels. Higher = wider effect. Default: 2.0",
                                )

                                sharpening_amount_artistic = gr.Slider(
                                    minimum=0.5,
                                    maximum=3.0,
                                    step=0.1,
                                    value=1.5,
                                    label="Sharpening Amount",
                                    info="Sharpening strength. Higher = stronger. Default: 1.5",
                                )

                                sharpening_threshold_artistic = gr.Slider(
                                    minimum=0,
                                    maximum=10,
                                    step=1,
                                    value=0,
                                    label="Sharpening Threshold",
                                    info="Minimum brightness change. 0 = all pixels, higher = edges only. Default: 0",
                                )

                            # Show/hide parameters when sharpening is toggled
                            enable_detail_sharpening_artistic.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[enable_detail_sharpening_artistic],
                                outputs=[sharpening_params_accordion],
                            )

                            # Section 3: Tile Preprocessor Configuration
                            gr.Markdown("### 🔹 Tile Preprocessor Configuration")
                            gr.Markdown(
                                "Adjust tile preprocessor detail level. Lower values preserve more details but may "
                                "reduce composition coherence. Higher values create smoother, more coherent results but lose fine details."
                            )

                            customize_tile_preprocessing_artistic = gr.Checkbox(
                                label="Customize Tile Preprocessing",
                                value=False,
                                info="Override default pyrUp iterations (default: 3)",
                            )

                            # Tile parameters (nested accordion, visible only when enabled)
                            tile_params_accordion = gr.Accordion(
                                "Tile Preprocessing Parameters",
                                open=False,
                                visible=False,
                            )

                            with tile_params_accordion:
                                gr.Markdown(
                                    "pyrUp iterations control detail vs smoothness trade-off. "
                                    "Default (3) is balanced. Lower = sharper/more details, Higher = smoother/less details."
                                )

                                tile_pyrup_iters_artistic = gr.Slider(
                                    minimum=1,
                                    maximum=4,
                                    step=1,
                                    value=3,
                                    label="Tile Detail Level (pyrUp iterations)",
                                    info="1=sharpest, 2=sharp, 3=balanced (default), 4=smoothest",
                                )

                            # Show/hide parameters when customization is toggled
                            customize_tile_preprocessing_artistic.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[customize_tile_preprocessing_artistic],
                                outputs=[tile_params_accordion],
                            )

                        # Color Quantization Section
                        gr.Markdown("### Color Quantization (Optional)")
                        gr.Markdown(
                            "Use this option to specify a custom color scheme for your QR code. Perfect for matching brand colors or creating themed designs."
                        )
                        artistic_enable_color_quantization = gr.Checkbox(
                            label="Enable Color Quantization",
                            value=False,
                            info="Apply a custom color palette to the generated image",
                        )

                        artistic_num_colors = gr.Slider(
                            minimum=2,
                            maximum=4,
                            step=1,
                            value=4,
                            label="Number of Colors",
                            info="How many colors to use from the palette (2-4)",
                            visible=False,
                        )

                        # Colors 1 & 2 (QR code colors - hidden when gradient enabled)
                        with gr.Row(visible=False) as artistic_color_pickers_row_1_2:
                            artistic_color_1 = gr.ColorPicker(
                                label="Color 1 (QR Dark)",
                                value="#000000",
                                info="Preserved when using gradients",
                            )
                            artistic_color_2 = gr.ColorPicker(
                                label="Color 2 (QR Light)",
                                value="#FFFFFF",
                                info="Preserved when using gradients",
                            )

                        # Colors 3 & 4 (Background colors - always editable)
                        with gr.Row(visible=False) as artistic_color_pickers_row_3_4:
                            artistic_color_3 = gr.ColorPicker(
                                label="Color 3 (Background)", value="#FF0000"
                            )
                            artistic_color_4 = gr.ColorPicker(
                                label="Color 4 (Background)", value="#00FF00"
                            )

                        # Gradient Filter Section (nested under color quantization)
                        artistic_apply_gradient_filter = gr.Checkbox(
                            label="Apply Gradient Filter",
                            value=False,
                            visible=False,
                            elem_id="artistic_gradient_checkbox",
                            info="Create gradient variations around colors 3-4 while preserving colors 1-2 for QR scannability",
                        )

                        artistic_gradient_strength = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            step=0.1,
                            value=0.3,
                            label="Gradient Strength",
                            info="Brightness variation (0.3 = ±30%)",
                            visible=False,
                        )

                        artistic_variation_steps = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=5,
                            label="Variation Steps",
                            info="Number of gradient steps (higher = smoother)",
                            visible=False,
                        )

                        # Visibility toggle for gradient filter
                        artistic_apply_gradient_filter.change(
                            fn=lambda gradient_enabled: (
                                gr.update(visible=gradient_enabled),
                                gr.update(visible=gradient_enabled),
                                gr.update(
                                    visible=not gradient_enabled
                                ),  # Hide colors 1&2 when gradient ON
                            ),
                            inputs=[artistic_apply_gradient_filter],
                            outputs=[
                                artistic_gradient_strength,
                                artistic_variation_steps,
                                artistic_color_pickers_row_1_2,
                            ],
                        )

                        # Visibility toggle for color quantization
                        artistic_enable_color_quantization.change(
                            fn=lambda enabled: (
                                gr.update(visible=enabled),
                                gr.update(visible=enabled),
                                gr.update(visible=enabled),
                                gr.update(visible=enabled),
                            ),
                            inputs=[artistic_enable_color_quantization],
                            outputs=[
                                artistic_num_colors,
                                artistic_color_pickers_row_1_2,
                                artistic_color_pickers_row_3_4,
                                artistic_apply_gradient_filter,
                            ],
                        )

                        # Add seed controls for artistic QR
                        artistic_use_custom_seed = gr.Checkbox(
                            label="Use Custom Seed",
                            value=True,
                            info="Enable to use a specific seed for reproducible results",
                        )
                        artistic_seed = gr.Slider(
                            minimum=0,
                            maximum=2**32 - 1,
                            step=1,
                            value=718313,
                            label="Seed",
                            visible=True,  # Initially visible since artistic_use_custom_seed=True
                            info="Seed value for reproducibility. Same seed with same settings will produce the same result.",
                        )

                        # FreeU Parameters
                        gr.Markdown("### FreeU Quality Enhancement")
                        enable_freeu_artistic = gr.Checkbox(
                            label="Enable FreeU",
                            value=True,
                            info="Enable FreeU quality enhancement (enabled by default for artistic pipeline)",
                        )
                        freeu_b1 = gr.Slider(
                            minimum=1.0,
                            maximum=1.6,
                            step=0.01,
                            value=1.4,
                            label="FreeU B1 (Backbone 1)",
                            info="Backbone feature enhancement for first layer. Higher values improve detail but may reduce blending. Range: 1.0-1.6, Default: 1.4",
                        )
                        freeu_b2 = gr.Slider(
                            minimum=1.0,
                            maximum=1.6,
                            step=0.01,
                            value=1.3,
                            label="FreeU B2 (Backbone 2)",
                            info="Backbone feature enhancement for second layer. Higher values improve texture. Range: 1.0-1.6, Default: 1.3",
                        )
                        freeu_s1 = gr.Slider(
                            minimum=0.0,
                            maximum=1.5,
                            step=0.01,
                            value=0.0,
                            label="FreeU S1 (Skip 1)",
                            info="Skip connection dampening for first layer. Lower values hide QR structure more. Range: 0.0-1.5, Default: 0.0",
                        )
                        freeu_s2 = gr.Slider(
                            minimum=0.0,
                            maximum=1.5,
                            step=0.01,
                            value=1.3,
                            label="FreeU S2 (Skip 2)",
                            info="Skip connection dampening for second layer. Balances scannability. Range: 0.0-1.5, Default: 1.3",
                        )

                        # SAG (Self-Attention Guidance) Parameters
                        gr.Markdown("### SAG (Self-Attention Guidance)")
                        enable_sag = gr.Checkbox(
                            label="Enable SAG",
                            value=True,
                            info="Enable Self-Attention Guidance for improved structural coherence and artistic blending",
                        )
                        sag_scale = gr.Slider(
                            minimum=0.0,
                            maximum=3.0,
                            step=0.1,
                            value=0.5,
                            label="SAG Scale",
                            info="Guidance strength. Higher values provide more structural coherence. Range: 0.0-3.0, Default: 0.5",
                        )
                        sag_blur_sigma = gr.Slider(
                            minimum=0.0,
                            maximum=5.0,
                            step=0.1,
                            value=0.5,
                            label="SAG Blur Sigma",
                            info="Blur amount for artistic blending. Higher values create softer, more artistic effects. Range: 0.0-5.0, Default: 0.5",
                        )

                        # ControlNet Strength Parameters
                        gr.Markdown("### ControlNet Strength (QR Code Preservation)")
                        gr.Markdown(
                            "**IMPORTANT:** Lower values preserve QR structure better (more scannable). Higher values create more artistic effects but may reduce scannability."
                        )
                        controlnet_strength_first = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.45,
                            label="First Pass Strength",
                            info="Controls how much the AI modifies the QR in the first pass. LOWER = more scannable, HIGHER = more artistic. Try 0.30-0.40 for better scannability. Default: 0.45",
                        )
                        controlnet_strength_final = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.7,
                            label="Final Pass Strength",
                            info="Controls how much the AI modifies the QR in the refinement pass. LOWER = preserves QR structure, HIGHER = more creative. Try 0.55-0.65 for balance. Default: 0.70",
                        )

                    artistic_validation_message = gr.Markdown(visible=False)
                    artistic_autofix_btn = gr.Button(
                        value="Increase image size automatically to fit QR code",
                        variant="secondary",
                        visible=False,
                    )
                    artistic_recommended_image_size = gr.State(value=None)

                    # The generate button for artistic QR
                    artistic_generate_btn = gr.Button(
                        "Generate Artistic QR", variant="primary"
                    )

                with gr.Column():
                    # Examples Gallery (initially visible)
                    gr.Markdown("### Featured Examples")

                    example_gallery = gr.Gallery(
                        value=[(ex["image"], ex["label"]) for ex in ARTISTIC_EXAMPLES],
                        label="Example Gallery",
                        columns=3,
                        rows=3,
                        height="auto",
                        object_fit="cover",
                        allow_preview=True,
                        show_download_button=False,
                    )

                    # State to track currently selected example index
                    current_example_index = gr.State(value=None)

                    # The output image for artistic QR (initially hidden)
                    artistic_output_image = gr.Image(
                        label="Generated Artistic QR Code",
                        visible=False,
                    )
                    artistic_error_message = gr.Textbox(
                        label="Status / Errors",
                        interactive=False,
                        lines=3,
                        visible=True,  # Keep visible to show status messages
                    )
                    # Wrap settings output in accordion (initially hidden)
                    with gr.Accordion(
                        "Shareable Settings (JSON)", open=True, visible=False
                    ) as settings_accordion_artistic:
                        settings_output_artistic = gr.Textbox(
                            label="Copy this JSON to share your exact settings",
                            interactive=True,
                            lines=5,
                            show_copy_button=True,
                        )

                    # Export buttons — hidden until generation succeeds
                    with gr.Row(visible=False) as export_row_artistic:
                        png_download_artistic = gr.DownloadButton(
                            "⬇ PNG",
                            variant="primary",
                            size="sm",
                            value=_download_png,
                            inputs=[
                                artistic_output_image,
                                artistic_text_input,
                                artistic_seed,
                                analytics_opt_in_global,
                            ],
                        )
                        svg_download_artistic = gr.DownloadButton(
                            "⬇ SVG",
                            variant="secondary",
                            size="sm",
                            value=_download_svg,
                            inputs=[
                                artistic_output_image,
                                artistic_text_input,
                                artistic_seed,
                                analytics_opt_in_global,
                            ],
                        )

                    # Button to show examples again (initially hidden)
                    show_examples_btn = gr.Button(
                        "🎨 Try Another Example",
                        variant="secondary",
                        visible=False,
                    )

            # When clicking the button, it will trigger the artistic function
            artistic_generate_btn.click(
                fn=generate_artistic_qr,
                inputs=[
                    artistic_prompt_input,
                    negative_prompt_artistic,
                    artistic_text_input,
                    artistic_input_type,
                    artistic_use_temporary_short_link,
                    artistic_image_size,
                    artistic_border_size,
                    artistic_error_correction,
                    artistic_module_size,
                    artistic_module_drawer,
                    artistic_use_custom_seed,
                    artistic_seed,
                    artistic_enable_upscale,
                    artistic_enable_animation,
                    enable_cascade_filter_artistic,
                    cascade_blur_kernel_artistic,
                    cascade_threshold_ratio_artistic,
                    enable_detail_sharpening_artistic,
                    sharpening_radius_artistic,
                    sharpening_amount_artistic,
                    sharpening_threshold_artistic,
                    customize_tile_preprocessing_artistic,
                    tile_pyrup_iters_artistic,
                    enable_freeu_artistic,
                    freeu_b1,
                    freeu_b2,
                    freeu_s1,
                    freeu_s2,
                    enable_sag,
                    sag_scale,
                    sag_blur_sigma,
                    controlnet_strength_first,
                    controlnet_strength_final,
                    artistic_enable_color_quantization,
                    artistic_num_colors,
                    artistic_color_1,
                    artistic_color_2,
                    artistic_color_3,
                    artistic_color_4,
                    artistic_apply_gradient_filter,
                    artistic_gradient_strength,
                    artistic_variation_steps,
                    analytics_opt_in_global,
                ],
                outputs=[
                    artistic_output_image,
                    artistic_error_message,
                    settings_output_artistic,
                    settings_accordion_artistic,
                    example_gallery,  # Control gallery visibility
                    show_examples_btn,  # Control button visibility
                    export_row_artistic,  # Control export buttons visibility
                ],
            )

            # Load Settings button event handler
            load_settings_btn_artistic.click(
                fn=load_settings_from_json_artistic,
                inputs=[import_json_input_artistic],
                outputs=[
                    artistic_prompt_input,
                    negative_prompt_artistic,
                    artistic_text_input,
                    artistic_input_type,
                    artistic_use_temporary_short_link,
                    artistic_image_size,
                    artistic_border_size,
                    artistic_error_correction,
                    artistic_module_size,
                    artistic_module_drawer,
                    artistic_use_custom_seed,
                    artistic_seed,
                    artistic_enable_upscale,
                    artistic_enable_animation,
                    enable_freeu_artistic,
                    freeu_b1,
                    freeu_b2,
                    freeu_s1,
                    freeu_s2,
                    enable_sag,
                    sag_scale,
                    sag_blur_sigma,
                    controlnet_strength_first,
                    controlnet_strength_final,
                    artistic_enable_color_quantization,
                    artistic_num_colors,
                    artistic_color_1,
                    artistic_color_2,
                    artistic_color_3,
                    artistic_color_4,
                    artistic_apply_gradient_filter,
                    artistic_gradient_strength,
                    artistic_variation_steps,
                    import_status_artistic,
                ],
            )

            # Clear button event handler for artistic tab
            clear_json_btn_artistic.click(
                fn=lambda: ("", gr.update(visible=False)),
                inputs=[],
                outputs=[import_json_input_artistic, import_status_artistic],
            )

            load_settings_btn_artistic.click(
                fn=_get_artistic_validation_state,
                inputs=[
                    artistic_text_input,
                    artistic_input_type,
                    artistic_image_size,
                    artistic_border_size,
                    artistic_error_correction,
                    artistic_module_size,
                ],
                outputs=[
                    artistic_validation_message,
                    artistic_generate_btn,
                    artistic_autofix_btn,
                    artistic_recommended_image_size,
                ],
            )

            # Seed slider visibility toggle for artistic tab
            artistic_use_custom_seed.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[artistic_use_custom_seed],
                outputs=[artistic_seed],
            )

            for component in [
                artistic_text_input,
                artistic_input_type,
                artistic_image_size,
                artistic_border_size,
                artistic_error_correction,
                artistic_module_size,
            ]:
                component.change(
                    fn=_get_artistic_validation_state,
                    inputs=[
                        artistic_text_input,
                        artistic_input_type,
                        artistic_image_size,
                        artistic_border_size,
                        artistic_error_correction,
                        artistic_module_size,
                    ],
                    outputs=[
                        artistic_validation_message,
                        artistic_generate_btn,
                        artistic_autofix_btn,
                        artistic_recommended_image_size,
                    ],
                )

            # Event handler for "Try Another Example" button
            def show_examples_again(current_idx):
                """Show the gallery with a random different example and load its settings"""
                # Pick a random index different from current
                available = [
                    i for i in range(len(ARTISTIC_EXAMPLES)) if i != current_idx
                ]
                new_idx = random.choice(available) if available else 0
                example = ARTISTIC_EXAMPLES[new_idx]
                validation_message, button_state, autofix_button, recommended_size = (
                    _get_artistic_validation_state(
                        example["text_input"],
                        example["input_type"],
                        example["image_size"],
                        example["border_size"],
                        example["error_correction"],
                        example["module_size"],
                    )
                )
                return (
                    gr.update(visible=False),  # Hide output image
                    "Settings loaded! Click 'Generate Artistic QR' to create your QR code",  # Status message
                    gr.update(visible=False),  # Hide settings accordion
                    gr.update(
                        visible=True, selected_index=new_idx
                    ),  # Show gallery with random selection
                    gr.update(visible=False),  # Hide this button
                    gr.update(visible=False),  # Hide export buttons
                    # Load the example settings
                    example["prompt"],
                    example["text_input"],
                    example["input_type"],
                    example["image_size"],
                    example["border_size"],
                    example["error_correction"],
                    example["module_size"],
                    example["module_drawer"],
                    example["use_custom_seed"],
                    example["seed"],
                    example["sag_blur_sigma"],
                    new_idx,  # Update current example index
                    validation_message,
                    button_state,
                    autofix_button,
                    recommended_size,
                )

            show_examples_btn.click(
                fn=show_examples_again,
                inputs=[current_example_index],
                outputs=[
                    artistic_output_image,
                    artistic_error_message,
                    settings_accordion_artistic,
                    example_gallery,
                    show_examples_btn,
                    export_row_artistic,  # Hide export buttons
                    # Settings outputs
                    artistic_prompt_input,
                    artistic_text_input,
                    artistic_input_type,
                    artistic_image_size,
                    artistic_border_size,
                    artistic_error_correction,
                    artistic_module_size,
                    artistic_module_drawer,
                    artistic_use_custom_seed,
                    artistic_seed,
                    sag_blur_sigma,
                    current_example_index,
                    artistic_validation_message,
                    artistic_generate_btn,
                    artistic_autofix_btn,
                    artistic_recommended_image_size,
                ],
            )

            # Event handler to load settings when user clicks an example
            def load_example_settings(evt: gr.SelectData):
                """Load settings when user clicks an example image"""
                example = ARTISTIC_EXAMPLES[evt.index]
                validation_message, button_state, autofix_button, recommended_size = (
                    _get_artistic_validation_state(
                        example["text_input"],
                        example["input_type"],
                        example["image_size"],
                        example["border_size"],
                        example["error_correction"],
                        example["module_size"],
                    )
                )
                return (
                    example["prompt"],
                    example["text_input"],
                    example["input_type"],
                    example["image_size"],
                    example["border_size"],
                    example["error_correction"],
                    example["module_size"],
                    example["module_drawer"],
                    example["use_custom_seed"],
                    example["seed"],
                    example["sag_blur_sigma"],
                    gr.update(visible=False),  # Hide output image
                    "Settings loaded! Click 'Generate Artistic QR' to create your QR code",  # Show in Status/Errors
                    gr.update(visible=False),  # Hide settings accordion
                    evt.index,  # Store the selected example index
                    gr.update(visible=False),  # Hide export buttons
                    validation_message,
                    button_state,
                    autofix_button,
                    recommended_size,
                )

            # Attach the event handler
            example_gallery.select(
                fn=load_example_settings,
                inputs=None,
                outputs=[
                    artistic_prompt_input,
                    artistic_text_input,
                    artistic_input_type,
                    artistic_image_size,
                    artistic_border_size,
                    artistic_error_correction,
                    artistic_module_size,
                    artistic_module_drawer,
                    artistic_use_custom_seed,
                    artistic_seed,
                    sag_blur_sigma,
                    artistic_output_image,  # Reset visibility
                    artistic_error_message,  # Show status message
                    settings_accordion_artistic,  # Reset visibility
                    current_example_index,  # Store the selected example index
                    export_row_artistic,  # Hide export buttons
                    artistic_validation_message,
                    artistic_generate_btn,
                    artistic_autofix_btn,
                    artistic_recommended_image_size,
                ],
            )

            demo.load(
                fn=_get_artistic_validation_state,
                inputs=[
                    artistic_text_input,
                    artistic_input_type,
                    artistic_image_size,
                    artistic_border_size,
                    artistic_error_correction,
                    artistic_module_size,
                ],
                outputs=[
                    artistic_validation_message,
                    artistic_generate_btn,
                    artistic_autofix_btn,
                    artistic_recommended_image_size,
                ],
            )

            artistic_autofix_btn.click(
                fn=_apply_recommended_image_size,
                inputs=[artistic_recommended_image_size],
                outputs=[artistic_image_size],
            )

        # STANDARD QR TAB
        with gr.TabItem("Standard QR"):
            # Short description
            gr.Markdown("""
                ⚡ **2x faster than Artistic pipeline** - perfect for quota management

                🎯 **More stable and scannable results** with proven reliability

                🛠️ **Advanced QR customization** with module styles and border controls
                """)

            # Full documentation in collapsed accordion
            with gr.Accordion("📖 Full Documentation & Tips", open=False):
                gr.Markdown("""
                    ### About Standard QR Pipeline
                    The Standard pipeline uses Stable Diffusion 1.5 with ControlNet for fast, reliable QR code generation. This pipeline offers:
                    - ~2x faster generation than Artistic (saves GPU quota)
                    - More scannable, stable output
                    - Customizable module styles (Square, Circle, Rounded, Bars, etc.)
                    - Border controls and error correction levels
                    - Optional upscaling (disabled by default to save quota)

                    ### Tips for Best Results:
                    - **Speed**: Use default 512px without upscaling (~10 seconds per generation)
                    - **Scannability**: Lower ControlNet strength = more scannable (try 0.35-0.50)
                    - **Module Styles**: Experiment with different QR patterns (see style examples below in settings)
                    - **Animation**: Disable to save ~20% GPU time
                    - **Border Size**: Higher values (6-8) add more whitespace around QR

                    ### GPU Usage:
                    - Default settings (512px): ~10 seconds per generation
                    - With upscaling: ~20-30 seconds
                    - Best for quota management compared to Artistic pipeline

                    ### Comparison with Artistic:
                    - **Standard**: Faster, more scannable, less creative
                    - **Artistic**: Slower, more creative, photorealistic

                    Choose Standard when you need speed and guaranteed scannability!
                    """)

            with gr.Row():
                with gr.Column():
                    # Add input type selector
                    input_type = gr.Radio(
                        choices=["URL", "Plain Text"],
                        value="URL",
                        label="Input Type",
                        info="URL: For web links (auto-removes https://). Plain Text: For VCARD, WiFi, calendar, location, etc. (no manipulation)",
                    )

                    # Add inputs
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate (check examples below for inspiration)",
                        value="some clothes spread on ropes, realistic, great details, out in the open air sunny day realistic, great details,absence of people, Detailed and Intricate, CGI, Photoshoot,rim light, 8k, 16k, ultra detail",
                        lines=3,
                    )
                    text_input = gr.Textbox(
                        label="QR Code Content",
                        placeholder="Enter URL or plain text",
                        value="https://www.google.com",
                        lines=3,
                        info="URL mode automatically removes common tracking params like utm_*, fbclid, and gclid before QR generation.",
                    )
                    use_temporary_short_link = gr.Checkbox(
                        label="Use URL shortener",
                        value=False,
                        info="URL mode only. Best option for longer links when you want cleaner QR codes without raising image size too much. The short link expires if nobody opens the QR code for 7 days.",
                    )

                    # Import Settings section - separate accordion
                    with gr.Accordion("Import Settings from JSON", open=False):
                        gr.Markdown(
                            "Paste a settings JSON string (copied from a previous generation) to load all parameters at once."
                        )
                        import_json_input_standard = gr.Textbox(
                            label="Paste Settings JSON",
                            placeholder='{"pipeline": "standard", "prompt": "...", "seed": 718313, ...}',
                            lines=3,
                        )
                        import_status_standard = gr.Textbox(
                            label="Import Status",
                            interactive=False,
                            visible=False,
                            lines=2,
                        )
                        with gr.Row():
                            load_settings_btn_standard = gr.Button(
                                "Load Settings", variant="primary"
                            )
                            clear_json_btn_standard = gr.Button(
                                "Clear", variant="secondary"
                            )

                    # Change Settings Manually - separate accordion
                    with gr.Accordion("Change Settings Manually", open=False):
                        gr.Markdown(
                            "**Advanced controls including:** Animation toggle, Color Quantization, ControlNet strength, QR settings, and more."
                        )
                        # Negative Prompt
                        negative_prompt_standard = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Describe what you don't want in the image",
                            value="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
                            lines=2,
                            info="Keywords to avoid in the generated image",
                        )

                        # Add image size slider
                        image_size = gr.Slider(
                            minimum=512,
                            maximum=1024,
                            step=64,
                            value=512,
                            label="Image Size",
                            info="Base size of the generated image. Final output will be 2x this size (e.g., 512 → 1024) due to the two-step enhancement process. Higher values use more VRAM and take longer to process.",
                        )

                        # Add border size slider
                        border_size = gr.Slider(
                            minimum=0,
                            maximum=8,
                            step=1,
                            value=4,
                            label="QR Code Border Size",
                            info="Number of modules (squares) to use as border around the QR code. Higher values add more whitespace.",
                        )

                        # Add error correction dropdown
                        error_correction = gr.Dropdown(
                            choices=[
                                "Low (7%)",
                                "Medium (15%)",
                                "Quartile (25%)",
                                "High (30%)",
                            ],
                            value="Medium (15%)",
                            label="Error Correction Level",
                            info="Higher error correction makes the QR code more scannable when damaged or obscured, but increases its size and complexity. Medium (15%) is a good starting point for most uses.",
                        )

                        # Add module size slider
                        module_size = gr.Slider(
                            minimum=4,
                            maximum=16,
                            step=1,
                            value=12,
                            label="QR Module Size",
                            info="Pixel width of the smallest QR code unit. Larger values improve readability but require a larger image size. 12 is a good starting point.",
                        )

                        # Add module drawer dropdown with style examples
                        module_drawer = gr.Dropdown(
                            choices=[
                                "Square",
                                "Gapped square",
                                "Circle",
                                "Rounded",
                                "Vertical bars",
                                "Horizontal bars",
                            ],
                            value="Square",
                            label="QR Code Style",
                            info="Select the style of the QR code modules (squares). See examples below. Different styles can give your QR code a unique look while maintaining scannability.",
                        )

                        # Add style examples with labels
                        gr.Markdown("### Style Examples:")

                        # First row of examples
                        with gr.Row():
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("**Square**", show_label=False)
                                gr.Image(
                                    "custom_nodes/ComfyQR/img/square.png",
                                    width=100,
                                    show_label=False,
                                    show_download_button=False,
                                )
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("**Gapped Square**", show_label=False)
                                gr.Image(
                                    "custom_nodes/ComfyQR/img/gapped_square.png",
                                    width=100,
                                    show_label=False,
                                    show_download_button=False,
                                )
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("**Circle**", show_label=False)
                                gr.Image(
                                    "custom_nodes/ComfyQR/img/circle.png",
                                    width=100,
                                    show_label=False,
                                    show_download_button=False,
                                )

                        # Second row of examples
                        with gr.Row():
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("**Rounded**", show_label=False)
                                gr.Image(
                                    "custom_nodes/ComfyQR/img/rounded.png",
                                    width=100,
                                    show_label=False,
                                    show_download_button=False,
                                )
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("**Vertical Bars**", show_label=False)
                                gr.Image(
                                    "custom_nodes/ComfyQR/img/vertical-bars.png",
                                    width=100,
                                    show_label=False,
                                    show_download_button=False,
                                )
                            with gr.Column(scale=1, min_width=0):
                                gr.Markdown("**Horizontal Bars**", show_label=False)
                                gr.Image(
                                    "custom_nodes/ComfyQR/img/horizontal-bars.png",
                                    width=100,
                                    show_label=False,
                                    show_download_button=False,
                                )

                        # Add upscale checkbox
                        enable_upscale = gr.Checkbox(
                            label="Enable Upscaling",
                            value=False,
                            info="Enable upscaling with RealESRGAN for higher quality output (disabled by default for standard pipeline)",
                        )

                        # Animation toggle
                        enable_animation = gr.Checkbox(
                            label="Enable Animation (Show KSampler Progress)",
                            value=True,
                            info="Shows intermediate images every 5 steps during generation. Disable for faster generation.",
                        )

                        # Color Quantization Section
                        gr.Markdown("### Color Quantization (Optional)")
                        gr.Markdown(
                            "Use this option to specify a custom color scheme for your QR code. Perfect for matching brand colors or creating themed designs."
                        )
                        enable_color_quantization = gr.Checkbox(
                            label="Enable Color Quantization",
                            value=False,
                            info="Apply a custom color palette to the generated image",
                        )

                        num_colors = gr.Slider(
                            minimum=2,
                            maximum=4,
                            step=1,
                            value=4,
                            label="Number of Colors",
                            info="How many colors to use from the palette (2-4)",
                            visible=False,
                        )

                        # Colors 1 & 2 (QR code colors - hidden when gradient enabled)
                        with gr.Row(visible=False) as color_pickers_row_1_2:
                            color_1 = gr.ColorPicker(
                                label="Color 1 (QR Dark)",
                                value="#000000",
                                info="Preserved when using gradients",
                            )
                            color_2 = gr.ColorPicker(
                                label="Color 2 (QR Light)",
                                value="#FFFFFF",
                                info="Preserved when using gradients",
                            )

                        # Colors 3 & 4 (Background colors - always editable)
                        with gr.Row(visible=False) as color_pickers_row_3_4:
                            color_3 = gr.ColorPicker(
                                label="Color 3 (Background)", value="#FF0000"
                            )
                            color_4 = gr.ColorPicker(
                                label="Color 4 (Background)", value="#00FF00"
                            )

                        # Gradient Filter Section (nested under color quantization)
                        apply_gradient_filter = gr.Checkbox(
                            label="Apply Gradient Filter",
                            value=False,
                            visible=False,
                            elem_id="gradient_checkbox",
                            info="Create gradient variations around colors 3-4 while preserving colors 1-2 for QR scannability",
                        )

                        gradient_strength = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            step=0.1,
                            value=0.3,
                            label="Gradient Strength",
                            info="Brightness variation (0.3 = ±30%)",
                            visible=False,
                        )

                        variation_steps = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=5,
                            label="Variation Steps",
                            info="Number of gradient steps (higher = smoother)",
                            visible=False,
                        )

                        # Visibility toggle for gradient filter
                        apply_gradient_filter.change(
                            fn=lambda gradient_enabled: (
                                gr.update(visible=gradient_enabled),
                                gr.update(visible=gradient_enabled),
                                gr.update(
                                    visible=not gradient_enabled
                                ),  # Hide colors 1&2 when gradient ON
                            ),
                            inputs=[apply_gradient_filter],
                            outputs=[
                                gradient_strength,
                                variation_steps,
                                color_pickers_row_1_2,
                            ],
                        )

                        # Visibility toggle for color quantization
                        enable_color_quantization.change(
                            fn=lambda enabled: (
                                gr.update(visible=enabled),
                                gr.update(visible=enabled),
                                gr.update(visible=enabled),
                                gr.update(visible=enabled),
                            ),
                            inputs=[enable_color_quantization],
                            outputs=[
                                num_colors,
                                color_pickers_row_1_2,
                                color_pickers_row_3_4,
                                apply_gradient_filter,
                            ],
                        )

                        # Add seed controls
                        use_custom_seed = gr.Checkbox(
                            label="Use Custom Seed",
                            value=True,
                            info="Enable to use a specific seed for reproducible results",
                        )
                        seed = gr.Slider(
                            minimum=0,
                            maximum=2**32 - 1,
                            step=1,
                            value=718313,
                            label="Seed",
                            visible=True,  # Initially visible since use_custom_seed=True
                            info="Seed value for reproducibility. Same seed with same settings will produce the same result.",
                        )

                        # ControlNet Strength Parameters
                        gr.Markdown("### ControlNet Strength (QR Code Preservation)")
                        gr.Markdown(
                            "**IMPORTANT:** Lower values preserve QR structure better (more scannable). Higher values create more artistic effects but may reduce scannability."
                        )
                        controlnet_strength_standard_first = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.45,
                            label="First Pass Strength (Brightness + Tile)",
                            info="Controls how much the AI modifies the QR in both ControlNet passes. LOWER = more scannable, HIGHER = more artistic. Try 0.35-0.50 for good balance. Default: 0.45",
                        )
                        controlnet_strength_standard_final = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=1.0,
                            label="Final Pass Strength (Tile Refinement)",
                            info="Controls the final tile ControlNet pass strength. Usually kept at 1.0 for clarity. Default: 1.0",
                        )

                    standard_validation_message = gr.Markdown(visible=False)
                    standard_autofix_btn = gr.Button(
                        value="Increase image size automatically to fit QR code",
                        variant="secondary",
                        visible=False,
                    )
                    standard_recommended_image_size = gr.State(value=None)

                    # The generate button
                    generate_btn = gr.Button("Generate Standard QR", variant="primary")

                with gr.Column():
                    # The output image
                    output_image = gr.Image(label="Generated Standard QR Code")
                    error_message = gr.Textbox(
                        label="Status / Errors",
                        interactive=False,
                        lines=3,
                    )
                    # Wrap settings output in accordion (initially hidden)
                    with gr.Accordion(
                        "Shareable Settings (JSON)", open=True, visible=False
                    ) as settings_accordion_standard:
                        settings_output_standard = gr.Textbox(
                            label="Copy this JSON to share your exact settings",
                            interactive=True,
                            lines=5,
                            show_copy_button=True,
                        )

                    # Export buttons — hidden until generation succeeds
                    with gr.Row(visible=False) as export_row_standard:
                        png_download_standard = gr.DownloadButton(
                            "⬇ PNG",
                            variant="primary",
                            size="sm",
                            value=_download_png,
                            inputs=[
                                output_image,
                                text_input,
                                seed,
                                analytics_opt_in_global,
                            ],
                        )
                        svg_download_standard = gr.DownloadButton(
                            "⬇ SVG",
                            variant="secondary",
                            size="sm",
                            value=_download_svg,
                            inputs=[
                                output_image,
                                text_input,
                                seed,
                                analytics_opt_in_global,
                            ],
                        )

            # When clicking the button, it will trigger the main function
            generate_btn.click(
                fn=generate_standard_qr,
                inputs=[
                    prompt_input,
                    negative_prompt_standard,
                    text_input,
                    input_type,
                    use_temporary_short_link,
                    image_size,
                    border_size,
                    error_correction,
                    module_size,
                    module_drawer,
                    use_custom_seed,
                    seed,
                    enable_upscale,
                    enable_animation,
                    controlnet_strength_standard_first,
                    controlnet_strength_standard_final,
                    enable_color_quantization,
                    num_colors,
                    color_1,
                    color_2,
                    color_3,
                    color_4,
                    apply_gradient_filter,
                    gradient_strength,
                    variation_steps,
                    analytics_opt_in_global,
                ],
                outputs=[
                    output_image,
                    error_message,
                    settings_output_standard,
                    settings_accordion_standard,
                    export_row_standard,  # Control export buttons visibility
                ],
                show_progress="full",
            )

            # Load Settings button event handler
            load_settings_btn_standard.click(
                fn=load_settings_from_json_standard,
                inputs=[import_json_input_standard],
                outputs=[
                    prompt_input,
                    negative_prompt_standard,
                    text_input,
                    input_type,
                    use_temporary_short_link,
                    image_size,
                    border_size,
                    error_correction,
                    module_size,
                    module_drawer,
                    use_custom_seed,
                    seed,
                    enable_upscale,
                    enable_animation,
                    controlnet_strength_standard_first,
                    controlnet_strength_standard_final,
                    enable_color_quantization,
                    num_colors,
                    color_1,
                    color_2,
                    color_3,
                    color_4,
                    apply_gradient_filter,
                    gradient_strength,
                    variation_steps,
                    import_status_standard,
                ],
            )

            load_settings_btn_standard.click(
                fn=_get_standard_validation_state,
                inputs=[
                    text_input,
                    input_type,
                    image_size,
                    border_size,
                    error_correction,
                    module_size,
                ],
                outputs=[
                    standard_validation_message,
                    generate_btn,
                    standard_autofix_btn,
                    standard_recommended_image_size,
                ],
            )

            # Clear button event handler
            clear_json_btn_standard.click(
                fn=lambda: ("", gr.update(visible=False)),
                inputs=[],
                outputs=[import_json_input_standard, import_status_standard],
            )

            # Seed slider visibility toggle
            use_custom_seed.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[use_custom_seed],
                outputs=[seed],
            )

            for component in [
                text_input,
                input_type,
                image_size,
                border_size,
                error_correction,
                module_size,
            ]:
                component.change(
                    fn=_get_standard_validation_state,
                    inputs=[
                        text_input,
                        input_type,
                        image_size,
                        border_size,
                        error_correction,
                        module_size,
                    ],
                    outputs=[
                        standard_validation_message,
                        generate_btn,
                        standard_autofix_btn,
                        standard_recommended_image_size,
                    ],
                )

            standard_autofix_btn.click(
                fn=_apply_recommended_image_size,
                inputs=[standard_recommended_image_size],
                outputs=[image_size],
            )

            # Add examples
            examples = STANDARD_EXAMPLES

            demo.load(
                fn=_get_standard_validation_state,
                inputs=[
                    text_input,
                    input_type,
                    image_size,
                    border_size,
                    error_correction,
                    module_size,
                ],
                outputs=[
                    standard_validation_message,
                    generate_btn,
                    standard_autofix_btn,
                    standard_recommended_image_size,
                ],
            )

            gr.Examples(
                examples=examples,
                inputs=[
                    prompt_input,
                    text_input,
                    input_type,
                    image_size,
                    border_size,
                    error_correction,
                    module_size,
                    module_drawer,
                ],
                cache_examples=False,  # Caching would require all 24 function parameters in examples
                examples_per_page=10,
                label="Example Presets (Click to Load)",
            )

        # ARTISTIC QR TAB

# Queue is required for gr.Progress() to work!
demo.queue()

# Launch the app when run directly (not during hot reload)
if __name__ == "__main__":
    # Call AOT compilation during startup (only on CUDA, not MPS)
    if not torch.backends.mps.is_available() and not os.environ.get("QR_TESTING_MODE"):
        compile_models_with_aoti()
    else:
        print("ℹ️  AOT compilation skipped (MPS or testing mode)\n")

    demo.launch(share=False, mcp_server=True)
    # Note: Automatic file cleanup via delete_cache not available in Gradio 5.49.1
    # Files will be cleaned up when the server is restarted
