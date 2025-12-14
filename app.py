import json
import os
import random
import sys
import warnings
from typing import Any, Mapping, Sequence, Union

import gradio as gr
import numpy as np
import spaces
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

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
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


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


# Device-specific optimizations
if torch.cuda.is_available() and not torch.backends.mps.is_available():
    # CUDA device - check bfloat16 support
    print(f"CUDA device detected (PyTorch {torch.__version__})")

    # Check if bfloat16 is supported (requires compute capability >= 8.0, e.g., A100, H100)
    if torch.cuda.is_bf16_supported():
        print("  ✓ Using bfloat16 precision for optimal performance")
        print("  ✓ Memory optimizations enabled")
        # Note: bfloat16 is handled automatically by model_management on CUDA
        # No dtype forcing needed - ComfyUI uses optimal dtypes by default
    else:
        print("  ⚠️  bfloat16 not supported on this GPU, using default precision")
        print("  ℹ️  For best performance, use GPU with compute capability >= 8.0")

elif torch.backends.mps.is_available():
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

# Add all the models that load a safetensors file
model_loaders = [checkpointloadersimple_4, checkpointloadersimple_artistic]

# Check which models are valid and how to best load them
valid_models = [
    getattr(loader[0], "patcher", loader[0])
    for loader in model_loaders
    if not isinstance(loader[0], dict)
    and not isinstance(getattr(loader[0], "patcher", None), dict)
]

model_management.load_models_gpu(valid_models)


# Apply torch.compile to diffusion models for 1.5-1.7× speedup
# Compilation happens once at startup (30-60s), then cached for fast inference
def _apply_torch_compile_optimizations():
    """Apply torch.compile to both pipeline models using ComfyUI's infrastructure"""
    try:
        from comfy_api.torch_helpers.torch_compile import set_torch_compile_wrapper

        print("\n🔧 Applying torch.compile optimizations...")

        # Compile standard pipeline model (DreamShaper 3.32)
        standard_model = get_value_at_index(checkpointloadersimple_4, 0)
        set_torch_compile_wrapper(
            model=standard_model,
            backend="inductor",
            mode="reduce-overhead",  # Best for iterative sampling
            fullgraph=False,  # ControlNet prevents full graph
            dynamic=False,  # Fixed image sizes per pipeline
            keys=["diffusion_model"],  # Compile UNet only
        )
        print("  ✓ Compiled standard pipeline diffusion model")

        # Compile artistic pipeline model (DreamShaper 6.31)
        artistic_model = get_value_at_index(checkpointloadersimple_artistic, 0)
        set_torch_compile_wrapper(
            model=artistic_model,
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=False,
            keys=["diffusion_model"],
        )
        print("  ✓ Compiled artistic pipeline diffusion model")
        print("✅ torch.compile optimizations applied successfully!\n")

    except Exception as e:
        print(f"⚠️  torch.compile optimization failed: {e}")
        print("   Continuing without compilation (slower but functional)\n")


# DISABLED: torch.compile causes graph breaks in ComfyUI timestep_embedding
# Error: "Cannot construct `ConstantVariable` for value of type <class 'torch.device'>"
# This is a known PyTorch limitation - torch.compile can't handle torch.device in graph
# Uncomment when PyTorch/ComfyUI fixes ConstantVariable handling for torch.device
#
# if torch.cuda.is_available():
#     _apply_torch_compile_optimizations()
# else:
#     print("ℹ️  Skipping torch.compile (not on CUDA)")

if torch.cuda.is_available():
    print("ℹ️  torch.compile disabled (compatibility issues with ComfyUI)")
    print("   App uses bfloat16 + VAE tiling + cache clearing for optimization")


@spaces.GPU(duration=30)
def generate_qr_code_unified(
    prompt: str,
    text_input: str,
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
    progress=gr.Progress(),
):
    # Only manipulate the text if it's a URL input type
    qr_text = text_input
    if input_type == "URL":
        if "https://" in qr_text:
            qr_text = qr_text.replace("https://", "")
        if "http://" in qr_text:
            qr_text = qr_text.replace("http://", "")

    # Use custom seed or random
    actual_seed = seed if use_custom_seed else random.randint(1, 2**64)

    with torch.inference_mode():
        if pipeline == "standard":
            yield from _pipeline_standard(
                prompt,
                qr_text,
                input_type,
                image_size,
                border_size,
                error_correction,
                module_size,
                module_drawer,
                actual_seed,
                enable_upscale,
                controlnet_strength_standard_first,
                controlnet_strength_standard_final,
                progress,
            )
        else:  # artistic
            yield from _pipeline_artistic(
                prompt,
                qr_text,
                input_type,
                image_size,
                border_size,
                error_correction,
                module_size,
                module_drawer,
                actual_seed,
                enable_upscale,
                freeu_b1,
                freeu_b2,
                freeu_s1,
                freeu_s2,
                enable_sag,
                sag_scale,
                sag_blur_sigma,
                controlnet_strength_first,
                controlnet_strength_final,
                progress,
            )


def generate_standard_qr(
    prompt: str,
    text_input: str,
    input_type: str = "URL",
    image_size: int = 512,
    border_size: int = 4,
    error_correction: str = "Medium (15%)",
    module_size: int = 12,
    module_drawer: str = "Square",
    use_custom_seed: bool = False,
    seed: int = 0,
    enable_upscale: bool = False,
    enable_freeu: bool = False,
    controlnet_strength_standard_first: float = 0.45,
    controlnet_strength_standard_final: float = 1.0,
    progress=gr.Progress(),
):
    """Wrapper function for standard QR generation"""
    # Get actual seed used (custom or random)
    actual_seed = seed if use_custom_seed else random.randint(1, 2**64)

    # Create settings JSON once
    settings_dict = {
        "pipeline": "standard",
        "prompt": prompt,
        "text_input": text_input,
        "input_type": input_type,
        "image_size": image_size,
        "border_size": border_size,
        "error_correction": error_correction,
        "module_size": module_size,
        "module_drawer": module_drawer,
        "seed": actual_seed,
        "use_custom_seed": True,
        "enable_upscale": enable_upscale,
        "enable_freeu": enable_freeu,
        "controlnet_strength_standard_first": controlnet_strength_standard_first,
        "controlnet_strength_standard_final": controlnet_strength_standard_final,
    }
    settings_json = generate_settings_json(settings_dict)

    # Generate QR and yield progressive results
    generator = generate_qr_code_unified(
        prompt,
        text_input,
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
        controlnet_strength_standard_first=controlnet_strength_standard_first,
        controlnet_strength_standard_final=controlnet_strength_standard_final,
        progress=progress,
    )

    final_image = None
    final_status = None

    for image, status in generator:
        final_image = image
        final_status = status
        # Show progressive updates but don't show accordion yet
        yield (image, status, gr.update(), gr.update())

    # After all steps complete, show the accordion with JSON
    if final_image is not None:
        yield (
            final_image,
            final_status,
            gr.update(value=settings_json),  # Update textbox content
            gr.update(visible=True),  # Make accordion visible only at the end
        )


def generate_artistic_qr(
    prompt: str,
    text_input: str,
    input_type: str = "URL",
    image_size: int = 512,
    border_size: int = 4,
    error_correction: str = "Medium (15%)",
    module_size: int = 12,
    module_drawer: str = "Square",
    use_custom_seed: bool = False,
    seed: int = 0,
    enable_upscale: bool = True,
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
    progress=gr.Progress(),
):
    """Wrapper function for artistic QR generation with FreeU and SAG parameters"""
    # Get actual seed used (custom or random)
    actual_seed = seed if use_custom_seed else random.randint(1, 2**64)

    # Create settings JSON once
    settings_dict = {
        "pipeline": "artistic",
        "prompt": prompt,
        "text_input": text_input,
        "input_type": input_type,
        "image_size": image_size,
        "border_size": border_size,
        "error_correction": error_correction,
        "module_size": module_size,
        "module_drawer": module_drawer,
        "seed": actual_seed,
        "use_custom_seed": True,
        "enable_upscale": enable_upscale,
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
    }
    settings_json = generate_settings_json(settings_dict)

    # Generate QR and yield progressive results
    generator = generate_qr_code_unified(
        prompt,
        text_input,
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
        progress=progress,
    )

    final_image = None
    final_status = None

    for image, status in generator:
        final_image = image
        final_status = status
        # Show progressive updates but don't show accordion yet
        yield (image, status, gr.update(), gr.update())

    # After all steps complete, show the accordion with JSON
    if final_image is not None:
        yield (
            final_image,
            final_status,
            gr.update(value=settings_json),  # Update textbox content
            gr.update(visible=True),  # Make accordion visible only at the end
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
                gr.update(value=error_msg, visible=True),
            )

        # Extract parameters with defaults
        prompt = params.get("prompt", "")
        text_input = params.get("text_input", "")
        input_type = params.get("input_type", "URL")
        image_size = params.get("image_size", 512)
        border_size = params.get("border_size", 4)
        error_correction = params.get("error_correction", "Medium (15%)")
        module_size = params.get("module_size", 12)
        module_drawer = params.get("module_drawer", "Square")
        use_custom_seed = params.get("use_custom_seed", True)
        seed = params.get("seed", 718313)
        enable_upscale = params.get("enable_upscale", False)
        enable_freeu = params.get("enable_freeu", False)
        controlnet_strength_standard_first = params.get(
            "controlnet_strength_standard_first", 0.45
        )
        controlnet_strength_standard_final = params.get(
            "controlnet_strength_standard_final", 1.0
        )

        success_msg = "✅ Settings loaded successfully!"
        return (
            prompt,
            text_input,
            input_type,
            image_size,
            border_size,
            error_correction,
            module_size,
            module_drawer,
            use_custom_seed,
            seed,
            enable_upscale,
            enable_freeu,
            controlnet_strength_standard_first,
            controlnet_strength_standard_final,
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
                gr.update(value=error_msg, visible=True),
            )

        # Extract parameters with defaults
        prompt = params.get("prompt", "")
        text_input = params.get("text_input", "")
        input_type = params.get("input_type", "URL")
        image_size = params.get("image_size", 704)
        border_size = params.get("border_size", 6)
        error_correction = params.get("error_correction", "High (30%)")
        module_size = params.get("module_size", 16)
        module_drawer = params.get("module_drawer", "Square")
        use_custom_seed = params.get("use_custom_seed", True)
        seed = params.get("seed", 718313)
        enable_upscale = params.get("enable_upscale", True)
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

        success_msg = "✅ Settings loaded successfully!"
        return (
            prompt,
            text_input,
            input_type,
            image_size,
            border_size,
            error_correction,
            module_size,
            module_drawer,
            use_custom_seed,
            seed,
            enable_upscale,
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
    img_np = image_tensor.cpu().numpy()

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
    qr_text: str,
    input_type: str,
    image_size: int,
    border_size: int,
    error_correction: str,
    module_size: int,
    module_drawer: str,
    seed: int,
    enable_upscale: bool = False,
    controlnet_strength_first: float = 0.45,
    controlnet_strength_final: float = 1.0,
    gr_progress=None,
):
    emptylatentimage_5 = emptylatentimage.generate(
        width=image_size, height=image_size, batch_size=1
    )

    cliptextencode_6 = cliptextencode.encode(
        text=prompt,
        clip=get_value_at_index(checkpointloadersimple_4, 1),
    )

    cliptextencode_7 = cliptextencode.encode(
        text="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
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

    # 1) Yield the base QR image as the first intermediate result
    base_qr_tensor = get_value_at_index(comfy_qr_by_module_size_15, 0)
    base_qr_np = (base_qr_tensor.cpu().numpy() * 255).astype(np.uint8)
    base_qr_np = base_qr_np[0]
    base_qr_pil = Image.fromarray(base_qr_np)
    msg = "Generated base QR pattern… enhancing with AI (step 1/3)"
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

        # Yield progress update after first sampling completes
        msg = "First pass sampling complete... decoding image"
        log_progress(msg, gr_progress, 0.4)
        yield base_qr_pil, msg  # Yield with same image as before

        # Calculate optimal tile size for this image
        tile_size, overlap = calculate_vae_tile_size(image_size)

        if tile_size is not None:
            # Use tiled decode for larger images
            vaedecode_8 = vaedecodetiled.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
                tile_size=tile_size,
                overlap=overlap,
            )
        else:
            # Small image, use standard decode (faster)
            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

        # 2) Yield the first decoded image as a second intermediate result
        mid_tensor = get_value_at_index(vaedecode_8, 0)
        mid_np = (mid_tensor.cpu().numpy() * 255).astype(np.uint8)
        mid_np = mid_np[0]
        mid_pil = Image.fromarray(mid_np)
        msg = "First enhancement pass complete (step 2/3)… refining details"
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

        # Yield progress update after second sampling completes
        msg = "Second pass sampling complete... decoding final image"
        log_progress(msg, gr_progress, 0.8)
        yield mid_pil, msg  # Yield with previous image

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
            pre_upscale_np = (pre_upscale_tensor.cpu().numpy() * 255).astype(np.uint8)
            pre_upscale_np = pre_upscale_np[0]
            pre_upscale_pil = Image.fromarray(pre_upscale_np)
            msg = "Enhancement complete (step 3/4)... upscaling image"
            log_progress(msg, gr_progress, 0.9)
            yield pre_upscale_pil, msg

            # Upscale the final image (load model on-demand)
            upscale_model = get_upscale_model()
            upscaled = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscale_model, 0),
                image=get_value_at_index(vaedecode_21, 0),
            )

            image_tensor = get_value_at_index(upscaled, 0)
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            image_np = image_np[0]
            pil_image = Image.fromarray(image_np)
            msg = "No errors, all good! Final QR art generated and upscaled. (step 4/4)"
            log_progress(msg, gr_progress, 1.0)
            yield (pil_image, msg)
        else:
            # No upscaling
            image_tensor = get_value_at_index(vaedecode_21, 0)
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            image_np = image_np[0]
            pil_image = Image.fromarray(image_np)
            msg = "No errors, all good! Final QR art generated."
            log_progress(msg, gr_progress, 1.0)
            yield pil_image, msg


def _pipeline_artistic(
    prompt: str,
    qr_text: str,
    input_type: str,
    image_size: int,
    border_size: int,
    error_correction: str,
    module_size: int,
    module_drawer: str,
    seed: int,
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
    gr_progress=None,
):
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
    base_qr_np = (base_qr_tensor.cpu().numpy() * 255).astype(np.uint8)
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
        noisy_qr_np = (qr_with_border_noise.cpu().numpy() * 255).astype(np.uint8)
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

    negative_prompt = cliptextencode.encode(
        text="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, closed eyes, text, logo",
        clip=get_value_at_index(checkpointloadersimple_artistic, 1),
    )

    # Load controlnets
    brightness_controlnet = controlnetloader.load_controlnet(
        control_net_name="models/control_v1p_sd15_brightness.safetensors"
    )

    tile_controlnet = controlnetloader.load_controlnet(
        control_net_name="control_v11f1e_sd15_tile_fp16.safetensors"
    )

    # First ControlNet pass (using QR with border cubics)
    controlnet_apply = controlnetapplyadvanced.apply_controlnet(
        strength=controlnet_strength_first,
        start_percent=0,
        end_percent=1,
        positive=get_value_at_index(positive_prompt, 0),
        negative=get_value_at_index(negative_prompt, 0),
        control_net=get_value_at_index(brightness_controlnet, 0),
        image=qr_with_border_noise,
        vae=get_value_at_index(checkpointloadersimple_artistic, 2),
    )

    # Tile preprocessor (using QR with border cubics)
    tile_processed = tilepreprocessor.execute(
        pyrUp_iters=3,
        resolution=image_size,
        image=qr_with_border_noise,
    )

    # Second ControlNet pass (using tile processed from QR with border cubics)
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

    # Yield progress update after first sampling completes
    msg = f"First pass sampling complete... decoding image (step {current_step}/{total_steps})"
    log_progress(msg, gr_progress, 0.4)
    yield (noisy_qr_pil if border_size > 0 else base_qr_pil, msg)

    # First decode with dynamic tiling
    tile_size, overlap = calculate_vae_tile_size(image_size)

    if tile_size is not None:
        decoded = vaedecodetiled.decode(
            samples=get_value_at_index(samples, 0),
            vae=get_value_at_index(checkpointloadersimple_artistic, 2),
            tile_size=tile_size,
            overlap=overlap,
        )
    else:
        decoded = vaedecode.decode(
            samples=get_value_at_index(samples, 0),
            vae=get_value_at_index(checkpointloadersimple_artistic, 2),
        )

    # Show first pass result
    first_pass_tensor = get_value_at_index(decoded, 0)
    first_pass_np = (first_pass_tensor.cpu().numpy() * 255).astype(np.uint8)
    first_pass_np = first_pass_np[0]
    first_pass_pil = Image.fromarray(first_pass_np)
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
        negative=get_value_at_index(negative_prompt, 0),
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

    final_samples = ksampler.sample(
        seed=seed + 1,
        steps=30,
        cfg=7,
        sampler_name="dpmpp_3m_sde",
        scheduler="karras",
        denoise=0.8,
        model=enhanced_model,  # Using FreeU + SAG enhanced model
        positive=get_value_at_index(controlnet_apply_final, 0),
        negative=get_value_at_index(controlnet_apply_final, 1),
        latent_image=get_value_at_index(upscaled_latent, 0),
    )

    # Yield progress update after second sampling completes
    msg = f"Second pass sampling complete... decoding final image (step {current_step}/{total_steps})"
    log_progress(msg, gr_progress, 0.8)
    yield (first_pass_pil, msg)

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
        pre_upscale_np = (pre_upscale_tensor.cpu().numpy() * 255).astype(np.uint8)
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
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        image_np = image_np[0]
        final_image = Image.fromarray(image_np)
        msg = f"No errors, all good! Final artistic QR code generated and upscaled. (step {current_step}/{total_steps})"
        log_progress(msg, gr_progress, 1.0)
        yield (final_image, msg)
    else:
        # No upscaling
        image_tensor = get_value_at_index(final_decoded, 0)
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        image_np = image_np[0]
        final_image = Image.fromarray(image_np)
        msg = f"No errors, all good! Final artistic QR code generated. (step {current_step}/{total_steps})"
        log_progress(msg, gr_progress, 1.0)
        yield (final_image, msg)


if __name__ == "__main__" and not os.environ.get("QR_TESTING_MODE"):
    # Start your Gradio app with automatic cache cleanup
    # delete_cache=(3600, 3600) means: check every hour and delete files older than 1 hour
    with gr.Blocks(delete_cache=(3600, 3600)) as app:
        # Add a title and description
        gr.Markdown("# QR Code Art Generator")
        gr.Markdown("""
        This is an AI-powered QR code generator that creates artistic QR codes using Stable Diffusion 1.5 and ControlNet models.
        The application uses a custom ComfyUI workflow to generate QR codes.

        **Privacy Notice:** Generated images are automatically deleted after 1 hour.
        Temporary files are checked and cleaned every hour. Download your QR codes promptly after generation.

        ### Tips:
        - Use detailed prompts for better results
        - Include style keywords like 'photorealistic', 'detailed', '8k'
        - Choose **URL** mode for web links or **Plain Text** mode for VCARD, WiFi credentials, calendar events, etc.
        - Try the examples below for inspiration
        - **Copy/paste settings**: After generation, copy the JSON settings string that appears below the image and paste it into "Import Settings from JSON" to reproduce exact results or share with others

        ### Two Modes:
        - **Artistic QR** (New pipeline, default): More artistic and creative results with upscaling (slower, more creative, less scannable)
        - **Standard QR** (Old pipeline, more stable): Stable, accurate QR code generation (faster, more scannable, less creative)

        ### Note:
        Selecting image_size more then 704 might fail to generate image when other users are trying app at the same time.

        Feel free to share your suggestions or feedback on how to improve the app! Thanks!
        """)

        # Add tabs for different generation methods
        with gr.Tabs():
            # ARTISTIC QR TAB
            with gr.TabItem("Artistic QR"):
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
                            value="Enter your prompt here... For example: 'a beautiful sunset over mountains, photorealistic, detailed landscape'",
                            lines=3,
                        )
                        artistic_text_input = gr.Textbox(
                            label="QR Code Content",
                            placeholder="Enter URL or plain text",
                            value="Enter your URL or text here... For example: https://github.com",
                            lines=3,
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
                            # Add image size slider for artistic QR
                            artistic_image_size = gr.Slider(
                                minimum=512,
                                maximum=1024,
                                step=64,
                                value=704,
                                label="Image Size",
                                info="Base size of the generated image. Final output will be 2x this size (e.g., 704 → 1408) due to the two-step enhancement process. Higher values use more VRAM and take longer to process.",
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
                                value="High (30%)",
                                label="Error Correction Level",
                                info="Higher error correction makes the QR code more scannable when damaged or obscured, but increases its size and complexity. High (30%) is recommended for artistic QR codes.",
                            )

                            # Add module size slider for artistic QR
                            artistic_module_size = gr.Slider(
                                minimum=4,
                                maximum=16,
                                step=1,
                                value=16,
                                label="QR Module Size",
                                info="Pixel width of the smallest QR code unit. Larger values improve readability but require a larger image size. 16 is a good starting point.",
                            )

                            # Add module drawer dropdown with style examples for artistic QR
                            artistic_module_drawer = gr.Dropdown(
                                choices=[
                                    "Square",
                                    "Gapped Square",
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
                                value=True,
                                info="Enable upscaling with RealESRGAN for higher quality output (enabled by default for artistic pipeline)",
                            )

                            # Add seed controls for artistic QR
                            artistic_use_custom_seed = gr.Checkbox(
                                label="Use Custom Seed",
                                value=True,
                                info="Enable to use a specific seed for reproducible results",
                            )
                            artistic_seed = gr.Slider(
                                minimum=0,
                                maximum=2000000,
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
                            gr.Markdown(
                                "### ControlNet Strength (QR Code Preservation)"
                            )
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

                        # The generate button for artistic QR
                        artistic_generate_btn = gr.Button(
                            "Generate Artistic QR", variant="primary"
                        )

                    with gr.Column():
                        # The output image for artistic QR
                        artistic_output_image = gr.Image(
                            label="Generated Artistic QR Code"
                        )
                        artistic_error_message = gr.Textbox(
                            label="Status / Errors",
                            interactive=False,
                            lines=3,
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

                # When clicking the button, it will trigger the artistic function
                artistic_generate_btn.click(
                    fn=generate_artistic_qr,
                    inputs=[
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
                        artistic_enable_upscale,
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
                    ],
                    outputs=[
                        artistic_output_image,
                        artistic_error_message,
                        settings_output_artistic,
                        settings_accordion_artistic,
                    ],
                )

                # Load Settings button event handler
                load_settings_btn_artistic.click(
                    fn=load_settings_from_json_artistic,
                    inputs=[import_json_input_artistic],
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
                        artistic_enable_upscale,
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
                        import_status_artistic,
                    ],
                )

                # Clear button event handler for artistic tab
                clear_json_btn_artistic.click(
                    fn=lambda: ("", gr.update(visible=False)),
                    inputs=[],
                    outputs=[import_json_input_artistic, import_status_artistic],
                )

                # Seed slider visibility toggle for artistic tab
                artistic_use_custom_seed.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[artistic_use_custom_seed],
                    outputs=[artistic_seed],
                )

                # Custom Examples Gallery with Images
                gr.Markdown("### Featured Examples")
                gr.Markdown(
                    "Click 'Load Settings' under any example to populate the form with those exact settings"
                )

                # First row (3 images)
                with gr.Row():
                    # Example 1: Japanese Temple
                    with gr.Column(scale=1):
                        ex1_img = gr.Image(
                            "examples/artistic/japanese_temple.jpg",
                            label="Japanese Temple",
                            show_label=True,
                            interactive=False,
                            show_download_button=False,
                            height=280,
                        )
                        ex1_btn = gr.Button(
                            "Load Settings", size="sm", variant="secondary"
                        )

                    # Example 2: Sunset Mountains
                    with gr.Column(scale=1):
                        ex2_img = gr.Image(
                            "examples/artistic/sunset_mountains.jpg",
                            label="Sunset Mountains",
                            show_label=True,
                            interactive=False,
                            show_download_button=False,
                            height=280,
                        )
                        ex2_btn = gr.Button(
                            "Load Settings", size="sm", variant="secondary"
                        )

                    # Example 3: Roman City
                    with gr.Column(scale=1):
                        ex3_img = gr.Image(
                            "examples/artistic/roman_city.jpg",
                            label="Roman City",
                            show_label=True,
                            interactive=False,
                            show_download_button=False,
                            height=280,
                        )
                        ex3_btn = gr.Button(
                            "Load Settings", size="sm", variant="secondary"
                        )

                # Second row (3 images)
                with gr.Row():
                    # Example 4: Neapolitan Pizza
                    with gr.Column(scale=1):
                        ex4_img = gr.Image(
                            "examples/artistic/neapolitan_pizza.webp",
                            label="Neapolitan Pizza",
                            show_label=True,
                            interactive=False,
                            show_download_button=False,
                            height=280,
                        )
                        ex4_btn = gr.Button(
                            "Load Settings", size="sm", variant="secondary"
                        )

                    # Example 5: Poker Chips
                    with gr.Column(scale=1):
                        ex5_img = gr.Image(
                            "examples/artistic/poker_chips.webp",
                            label="Poker Chips",
                            show_label=True,
                            interactive=False,
                            show_download_button=False,
                            height=280,
                        )
                        ex5_btn = gr.Button(
                            "Load Settings", size="sm", variant="secondary"
                        )

                    # Example 6: Underwater Fish
                    with gr.Column(scale=1):
                        ex6_img = gr.Image(
                            "examples/artistic/underwater_fish.webp",
                            label="Underwater Fish",
                            show_label=True,
                            interactive=False,
                            show_download_button=False,
                            height=280,
                        )
                        ex6_btn = gr.Button(
                            "Load Settings", size="sm", variant="secondary"
                        )

                # Third row (3 images)
                with gr.Row():
                    # Example 7: Mediterranean Garden
                    with gr.Column(scale=1):
                        ex7_img = gr.Image(
                            "examples/artistic/mediterranean_garden.jpg",
                            label="Mediterranean Garden",
                            show_label=True,
                            interactive=False,
                            show_download_button=False,
                            height=280,
                        )
                        ex7_btn = gr.Button(
                            "Load Settings", size="sm", variant="secondary"
                        )

                    # Example 8: Rice Fields
                    with gr.Column(scale=1):
                        ex8_img = gr.Image(
                            "examples/artistic/rice_fields.jpg",
                            label="Rice Fields",
                            show_label=True,
                            interactive=False,
                            show_download_button=False,
                            height=280,
                        )
                        ex8_btn = gr.Button(
                            "Load Settings", size="sm", variant="secondary"
                        )

                    # Example 9: Cyberpunk City
                    with gr.Column(scale=1):
                        ex9_img = gr.Image(
                            "examples/artistic/cyberpunk_city.webp",
                            label="Cyberpunk City",
                            show_label=True,
                            interactive=False,
                            show_download_button=False,
                            height=280,
                        )
                        ex9_btn = gr.Button(
                            "Load Settings", size="sm", variant="secondary"
                        )

                # Load settings button handlers
                # Ex1: Japanese Temple
                ex1_btn.click(
                    fn=lambda: (
                        "some clothes spread on ropes, Japanese girl sits inside in the middle of the image, few sakura flowers, realistic, great details, out in the open air sunny day realistic, great details, absence of people, Detailed and Intricate, CGI, Photoshoot, rim light, 8k, 16k, ultra detail",
                        "https://www.google.com",
                        "URL",
                        640,
                        6,
                        "Medium (15%)",
                        14,
                        "Square",
                        True,
                        718313,
                        0.5,
                    ),
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
                    ],
                )
                # Ex2: Sunset Mountains
                ex2_btn.click(
                    fn=lambda: (
                        "a beautiful sunset over mountains, photorealistic, detailed landscape, golden hour, dramatic lighting, 8k, ultra detailed",
                        "https://github.com",
                        "URL",
                        704,
                        6,
                        "High (30%)",
                        16,
                        "Square",
                        True,
                        718313,
                        0.5,
                    ),
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
                    ],
                )
                # Ex3: Roman City
                ex3_btn.click(
                    fn=lambda: (
                        "aerial bird view of ancient Roman city, cobblestone streets and pathways forming intricate patterns, vintage illustration style, sepia tones, aged parchment look, detailed architecture, 8k, ultra detailed",
                        "WIFI:T:WPA;S:MyNetwork;P:MyPassword123;;",
                        "Plain Text",
                        832,
                        6,
                        "High (30%)",
                        16,
                        "Square",
                        True,
                        718313,
                        0.5,
                    ),
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
                    ],
                )
                # Ex4: Neapolitan Pizza
                ex4_btn.click(
                    fn=lambda: (
                        "artisan Neapolitan pizza on rustic wooden board, fresh basil leaves scattered on top and around, oregano sprinkled, flour dust particles floating in air, melted mozzarella with char marks, traditional Italian pizzeria ambiance, warm brick oven glow in background, detailed food photography, photorealistic, 8k, ultra detailed",
                        "https://www.pizzamaking.com",
                        "URL",
                        704,
                        6,
                        "High (30%)",
                        16,
                        "Square",
                        True,
                        856749,
                        2.0,
                    ),
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
                    ],
                )
                # Ex5: Poker Chips
                ex5_btn.click(
                    fn=lambda: (
                        "some cards on poker tale, realistic, great details, realistic, great details,absence of people, Detailed and Intricate, CGI, Photoshoot,rim light, 8k, 16k, ultra detail",
                        "https://store.steampowered.com",
                        "URL",
                        768,
                        6,
                        "High (30%)",
                        16,
                        "Square",
                        True,
                        718313,
                        2.5,
                    ),
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
                    ],
                )
                # Ex6: Underwater Fish
                ex6_btn.click(
                    fn=lambda: (
                        "underwater scene with tropical fish, coral reef, rays of sunlight penetrating water, vibrant colors, detailed marine life, photorealistic, 8k, ultra detailed",
                        "https://www.reddit.com",
                        "URL",
                        704,
                        6,
                        "High (30%)",
                        16,
                        "Square",
                        True,
                        718313,
                        0.5,
                    ),
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
                    ],
                )
                # Ex7: Mediterranean Garden
                ex7_btn.click(
                    fn=lambda: (
                        "ancient stone sundial in Mediterranean garden, olive trees, dappled sunlight through leaves, weathered stone texture, peaceful afternoon scene, photorealistic, detailed, 8k, ultra detailed",
                        "BEGIN:VEVENT\\nSUMMARY:Team Meeting\\nDTSTART:20251115T140000Z\\nDTEND:20251115T150000Z\\nLOCATION:Conference Room A\\nEND:VEVENT",
                        "Plain Text",
                        1024,
                        6,
                        "High (30%)",
                        14,
                        "Square",
                        True,
                        413468,
                        0.5,
                    ),
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
                    ],
                )
                # Ex8: Rice Fields
                ex8_btn.click(
                    fn=lambda: (
                        "aerial view of terraced rice fields on mountainside, winding pathways between green paddies, Asian countryside, bird's eye perspective, detailed landscape, golden hour lighting, photorealistic, 8k, ultra detailed",
                        "geo:37.7749,-122.4194",
                        "Plain Text",
                        704,
                        6,
                        "High (30%)",
                        16,
                        "Square",
                        True,
                        962359,
                        0.5,
                    ),
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
                    ],
                )
                # Ex9: Cyberpunk City
                ex9_btn.click(
                    fn=lambda: (
                        "futuristic cityscape with flying cars and neon lights, cyberpunk style, detailed architecture, night scene, 8k, ultra detailed",
                        "https://linkedin.com",
                        "URL",
                        704,
                        6,
                        "High (30%)",
                        16,
                        "Square",
                        True,
                        718313,
                        1.5,
                    ),
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
                    ],
                )

            # STANDARD QR TAB
            with gr.TabItem("Standard QR"):
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
                            value="Enter your prompt here... For example: 'a beautiful sunset over mountains, photorealistic, detailed landscape'",
                            lines=3,
                        )
                        text_input = gr.Textbox(
                            label="QR Code Content",
                            placeholder="Enter URL or plain text",
                            value="Enter your URL or text here... For example: https://github.com",
                            lines=3,
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
                                    "Gapped Square",
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

                            # Add FreeU checkbox
                            enable_freeu_standard = gr.Checkbox(
                                label="Enable FreeU",
                                value=False,
                                info="Enable FreeU quality enhancement (disabled by default for standard pipeline)",
                            )

                            # Add seed controls
                            use_custom_seed = gr.Checkbox(
                                label="Use Custom Seed",
                                value=True,
                                info="Enable to use a specific seed for reproducible results",
                            )
                            seed = gr.Slider(
                                minimum=0,
                                maximum=2000000,
                                step=1,
                                value=718313,
                                label="Seed",
                                visible=True,  # Initially visible since use_custom_seed=True
                                info="Seed value for reproducibility. Same seed with same settings will produce the same result.",
                            )

                            # ControlNet Strength Parameters
                            gr.Markdown(
                                "### ControlNet Strength (QR Code Preservation)"
                            )
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

                        # The generate button
                        generate_btn = gr.Button(
                            "Generate Standard QR", variant="primary"
                        )

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

                # When clicking the button, it will trigger the main function
                generate_btn.click(
                    fn=generate_standard_qr,
                    inputs=[
                        prompt_input,
                        text_input,
                        input_type,
                        image_size,
                        border_size,
                        error_correction,
                        module_size,
                        module_drawer,
                        use_custom_seed,
                        seed,
                        enable_upscale,
                        enable_freeu_standard,
                        controlnet_strength_standard_first,
                        controlnet_strength_standard_final,
                    ],
                    outputs=[
                        output_image,
                        error_message,
                        settings_output_standard,
                        settings_accordion_standard,
                    ],
                )

                # Load Settings button event handler
                load_settings_btn_standard.click(
                    fn=load_settings_from_json_standard,
                    inputs=[import_json_input_standard],
                    outputs=[
                        prompt_input,
                        text_input,
                        input_type,
                        image_size,
                        border_size,
                        error_correction,
                        module_size,
                        module_drawer,
                        use_custom_seed,
                        seed,
                        enable_upscale,
                        enable_freeu_standard,
                        controlnet_strength_standard_first,
                        controlnet_strength_standard_final,
                        import_status_standard,
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

                # Add examples
                examples = [
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
                    outputs=[output_image, error_message],
                    fn=generate_standard_qr,
                    cache_examples=False,
                )

            # ARTISTIC QR TAB
    app.queue()  # Required for gr.Progress() to work!
    app.launch(share=False, mcp_server=True)
    # Note: Automatic file cleanup via delete_cache not available in Gradio 5.49.1
    # Files will be cleaned up when the server is restarted
