from __future__ import annotations

import base64
import io
import json
import random
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    mode: str = Field(default="standard", pattern="^(standard|artistic)$")
    prompt: str = Field(..., min_length=1, max_length=1000)
    qr_text: str = Field(..., min_length=1, max_length=4000)
    input_type: str = Field(default="URL")
    image_size: int = Field(default=512, ge=256, le=1024)
    border_size: int = Field(default=4, ge=0, le=20)
    error_correction: str = Field(default="Medium (15%)")
    module_size: int = Field(default=12, ge=4, le=32)
    module_drawer: str = Field(default="Square")
    use_custom_seed: bool = Field(default=False)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    enable_upscale: bool = Field(default=False)
    enable_animation: bool = Field(default=False)
    include_svg: bool = Field(default=False)
    include_image_base64: bool = Field(default=True)
    use_temporary_short_link: bool = Field(default=False)
    analytics_opt_in: bool = Field(default=False)
    negative_prompt: str = Field(
        default=(
            "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, "
            "out of frame, extra limbs, body out of frame, blurry, bad anatomy, "
            "blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo"
        )
    )
    freeu_b1: float = Field(default=1.4)
    freeu_b2: float = Field(default=1.3)
    freeu_s1: float = Field(default=0.0)
    freeu_s2: float = Field(default=1.3)
    enable_sag: bool = Field(default=True)
    sag_scale: float = Field(default=0.5)
    sag_blur_sigma: float = Field(default=0.5)
    controlnet_strength_first: float = Field(default=0.45)
    controlnet_strength_final: float = Field(default=0.7)
    controlnet_strength_standard_first: float = Field(default=0.45)
    controlnet_strength_standard_final: float = Field(default=1.0)
    enable_color_quantization: bool = Field(default=False)
    num_colors: int = Field(default=4, ge=2, le=4)
    color_1: str = Field(default="#000000")
    color_2: str = Field(default="#FFFFFF")
    color_3: str = Field(default="#FF0000")
    color_4: str = Field(default="#00FF00")
    apply_gradient_filter: bool = Field(default=False)
    gradient_strength: float = Field(default=0.3)
    variation_steps: int = Field(default=5, ge=1, le=10)
    enable_cascade_filter: bool = Field(default=False)
    cascade_blur_kernel: int = Field(default=15)
    cascade_threshold_ratio: float = Field(default=0.33)
    enable_detail_sharpening: bool = Field(default=False)
    sharpening_radius: float = Field(default=2.0)
    sharpening_amount: float = Field(default=1.5)
    sharpening_threshold: int = Field(default=0)
    customize_tile_preprocessing: bool = Field(default=False)
    tile_pyrup_iters: int = Field(default=3, ge=1, le=4)


def build_generation_kwargs(request: GenerateRequest) -> dict[str, Any]:
    common = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "text_input": request.qr_text,
        "input_type": request.input_type,
        "use_temporary_short_link": request.use_temporary_short_link,
        "image_size": request.image_size,
        "border_size": request.border_size,
        "error_correction": request.error_correction,
        "module_size": request.module_size,
        "module_drawer": request.module_drawer,
        "use_custom_seed": request.use_custom_seed,
        "seed": request.seed,
        "enable_upscale": request.enable_upscale,
        "enable_animation": request.enable_animation,
        "analytics_opt_in": request.analytics_opt_in,
        "controlnet_strength_standard_first": request.controlnet_strength_standard_first,
        "controlnet_strength_standard_final": request.controlnet_strength_standard_final,
        "enable_color_quantization": request.enable_color_quantization,
        "num_colors": request.num_colors,
        "color_1": request.color_1,
        "color_2": request.color_2,
        "color_3": request.color_3,
        "color_4": request.color_4,
        "apply_gradient_filter": request.apply_gradient_filter,
        "gradient_strength": request.gradient_strength,
        "variation_steps": request.variation_steps,
        "progress": None,
        "request": None,
    }

    if request.mode == "artistic":
        return {
            **common,
            "freeu_b1": request.freeu_b1,
            "freeu_b2": request.freeu_b2,
            "freeu_s1": request.freeu_s1,
            "freeu_s2": request.freeu_s2,
            "enable_sag": request.enable_sag,
            "sag_scale": request.sag_scale,
            "sag_blur_sigma": request.sag_blur_sigma,
            "controlnet_strength_first": request.controlnet_strength_first,
            "controlnet_strength_final": request.controlnet_strength_final,
            "enable_cascade_filter": request.enable_cascade_filter,
            "cascade_blur_kernel": request.cascade_blur_kernel,
            "cascade_threshold_ratio": request.cascade_threshold_ratio,
            "enable_detail_sharpening": request.enable_detail_sharpening,
            "sharpening_radius": request.sharpening_radius,
            "sharpening_amount": request.sharpening_amount,
            "sharpening_threshold": request.sharpening_threshold,
            "customize_tile_preprocessing": request.customize_tile_preprocessing,
            "tile_pyrup_iters": request.tile_pyrup_iters,
        }

    return common


def consume_final_result(
    results: Iterable[Any],
) -> tuple[Any, str, dict[str, Any] | None]:
    final_image = None
    final_status = ""
    final_settings = None
    for result in results:
        if not isinstance(result, tuple) or len(result) < 2:
            continue
        image = result[0]
        status = result[1]
        if image is not None:
            final_image = image
        final_status = status
        if len(result) >= 3:
            maybe_settings = _extract_settings_dict(result[2])
            if maybe_settings is not None:
                final_settings = maybe_settings
    if final_image is None:
        raise ValueError("Generation finished without producing an image")
    return final_image, final_status, final_settings


def _extract_settings_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(value, Mapping):
        nested = value.get("value")
        if isinstance(nested, str):
            try:
                parsed = json.loads(nested)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def encode_image_to_base64_png(image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_image_to_embedded_svg(image) -> str:
    png_base64 = encode_image_to_base64_png(image)
    width, height = image.size
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f"<title>AI QR Code</title>"
        f'<image href="data:image/png;base64,{png_base64}" '
        f'x="0" y="0" width="{width}" height="{height}" '
        f'preserveAspectRatio="xMidYMid meet"/></svg>'
    )


def resolve_request_seed(request: GenerateRequest, randrange=random.randint) -> int:
    if request.use_custom_seed:
        return request.seed
    return randrange(1, 2**32 - 1)


def build_response_payload(
    image_obj,
    final_status: str,
    request: GenerateRequest,
    actual_seed: int,
    elapsed: float,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "status": "completed",
        "mode": request.mode,
        "final_status": final_status,
        "seed": actual_seed,
        "image_base64": encode_image_to_base64_png(image_obj)
        if request.include_image_base64
        else None,
        "image_format": "png",
        "svg": encode_image_to_embedded_svg(image_obj) if request.include_svg else None,
        "svg_format": "embedded-png-svg" if request.include_svg else None,
        "width": image_obj.width,
        "height": image_obj.height,
        "duration_seconds": elapsed,
    }
    if settings:
        for key in (
            "use_temporary_short_link",
            "shortener_applied",
            "short_url",
            "shortener_expires_at",
            "shortener_error",
            "effective_qr_text",
            "url_normalization_applied",
            "url_tracking_params_removed",
            "url_chars_saved",
        ):
            if key in settings:
                payload[key] = settings[key]
    return payload
