import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import gradio as gr
from PIL import Image
import numpy as np

import spaces

# Import FreeU for quality improvements
from comfy_extras.nodes_freelunch import FreeU_V2

from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5", filename="v1-5-pruned-emaonly.ckpt", local_dir="models/checkpoints")
hf_hub_download(repo_id="Lykon/DreamShaper", filename="DreamShaper_3.32_baked_vae_clip_fix_half.safetensors", local_dir="models/checkpoints")
hf_hub_download(repo_id="Lykon/DreamShaper", filename="DreamShaper_6.31_BakedVae_pruned.safetensors", local_dir="models/checkpoints")
hf_hub_download(repo_id="latentcat/latentcat-controlnet", filename="models/control_v1p_sd15_brightness.safetensors", local_dir="models/controlnet")
hf_hub_download(repo_id="comfyanonymous/ControlNet-v1-1_fp16_safetensors", filename="control_v11f1e_sd15_tile_fp16.safetensors", local_dir="models/controlnet")
hf_hub_download(repo_id="Lykon/dreamshaper-7", filename="vae/diffusion_pytorch_model.fp16.safetensors", local_dir="models")
hf_hub_download(repo_id="stabilityai/sd-vae-ft-mse-original", filename="vae-ft-mse-840000-ema-pruned.safetensors", local_dir="models/vae")
hf_hub_download(repo_id="lllyasviel/Annotators", filename="RealESRGAN_x4plus.pth", local_dir="models/upscale_models")
hf_hub_download(repo_id="lykon/RealESRGAN_x4plus", filename="RealESRGAN_x4plus.pth", local_dir="models/upscale_models")

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
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS

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

import_custom_nodes()
comfy_qr_by_module_size = NODE_CLASS_MAPPINGS["comfy-qr-by-module-size"]()
tilepreprocessor = NODE_CLASS_MAPPINGS["TilePreprocessor"]()

# Load upscale model and additional nodes for artistic pipeline
upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
upscalemodelloader_30 = upscalemodelloader.load_model(
    model_name="RealESRGAN_x4plus.pth"
)
imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()
latentupscaleby = NODE_CLASS_MAPPINGS["LatentUpscaleBy"]()

from comfy import model_management

# Add all the models that load a safetensors file
model_loaders = [checkpointloadersimple_4, checkpointloadersimple_artistic]

# Check which models are valid and how to best load them
valid_models = [
    getattr(loader[0], 'patcher', loader[0])
    for loader in model_loaders
    if not isinstance(loader[0], dict) and not isinstance(getattr(loader[0], 'patcher', None), dict)
]

model_management.load_models_gpu(valid_models)

@spaces.GPU(duration=30)
def generate_qr_code_unified(prompt: str, text_input: str, input_type: str = "URL", image_size: int = 512, border_size: int = 4, error_correction: str = "Medium (15%)", module_size: int = 12, module_drawer: str = "Square", use_custom_seed: bool = False, seed: int = 0, pipeline: str = "standard", enable_upscale: bool = False, freeu_b1: float = 1.4, freeu_b2: float = 1.3, freeu_s1: float = 0.0, freeu_s2: float = 1.3, enable_sag: bool = True, sag_scale: float = 0.5, sag_blur_sigma: float = 1.5, controlnet_strength_first: float = 0.45, controlnet_strength_final: float = 0.7):
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
            yield from _pipeline_standard(prompt, qr_text, input_type, image_size, border_size, error_correction, module_size, module_drawer, actual_seed, enable_upscale)
        else:  # artistic
            yield from _pipeline_artistic(prompt, qr_text, input_type, image_size, border_size, error_correction, module_size, module_drawer, actual_seed, enable_upscale, freeu_b1, freeu_b2, freeu_s1, freeu_s2, enable_sag, sag_scale, sag_blur_sigma, controlnet_strength_first, controlnet_strength_final)

def generate_standard_qr(prompt: str, text_input: str, input_type: str = "URL", image_size: int = 512, border_size: int = 4, error_correction: str = "Medium (15%)", module_size: int = 12, module_drawer: str = "Square", use_custom_seed: bool = False, seed: int = 0, enable_upscale: bool = False):
    """Wrapper function for standard QR generation"""
    yield from generate_qr_code_unified(prompt, text_input, input_type, image_size, border_size, error_correction, module_size, module_drawer, use_custom_seed, seed, pipeline="standard", enable_upscale=enable_upscale)

def generate_artistic_qr(prompt: str, text_input: str, input_type: str = "URL", image_size: int = 512, border_size: int = 4, error_correction: str = "Medium (15%)", module_size: int = 12, module_drawer: str = "Square", use_custom_seed: bool = False, seed: int = 0, enable_upscale: bool = True, freeu_b1: float = 1.4, freeu_b2: float = 1.3, freeu_s1: float = 0.0, freeu_s2: float = 1.3, enable_sag: bool = True, sag_scale: float = 0.5, sag_blur_sigma: float = 1.5, controlnet_strength_first: float = 0.45, controlnet_strength_final: float = 0.7):
    """Wrapper function for artistic QR generation with FreeU and SAG parameters"""
    yield from generate_qr_code_unified(prompt, text_input, input_type, image_size, border_size, error_correction, module_size, module_drawer, use_custom_seed, seed, pipeline="artistic", enable_upscale=enable_upscale, freeu_b1=freeu_b1, freeu_b2=freeu_b2, freeu_s1=freeu_s1, freeu_s2=freeu_s2, enable_sag=enable_sag, sag_scale=sag_scale, sag_blur_sigma=sag_blur_sigma, controlnet_strength_first=controlnet_strength_first, controlnet_strength_final=controlnet_strength_final)

def add_noise_to_border_only(image_tensor, seed: int, border_size: int, image_size: int, module_size: int = 12):
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
    border_mask[height-border_thickness:height, :] = True
    # Left border
    border_mask[:, 0:border_thickness] = True
    # Right border
    border_mask[:, width-border_thickness:width] = True

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

def _pipeline_standard(prompt: str, qr_text: str, input_type: str, image_size: int, border_size: int, error_correction: str, module_size: int, module_drawer: str, seed: int, enable_upscale: bool = False):
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
            "Try with a shorter text, increase the image size, or decrease the border size, module size, and error correction level under Advanced Settings."
        )
        yield None, error_msg
        return

    # 1) Yield the base QR image as the first intermediate result
    base_qr_tensor = get_value_at_index(comfy_qr_by_module_size_15, 0)
    base_qr_np = (base_qr_tensor.cpu().numpy() * 255).astype(np.uint8)
    base_qr_np = base_qr_np[0]
    base_qr_pil = Image.fromarray(base_qr_np)
    yield base_qr_pil, "Generated base QR pattern… enhancing with AI (step 1/3)"

    emptylatentimage_17 = emptylatentimage.generate(
        width=image_size*2, height=image_size*2, batch_size=1
    )

    controlnetloader_19 = controlnetloader.load_controlnet(
        control_net_name="control_v11f1e_sd15_tile_fp16.safetensors"
    )

    for q in range(1):
        controlnetapplyadvanced_11 = controlnetapplyadvanced.apply_controlnet(
            strength=0.45,
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
            strength=0.45,
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

        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(ksampler_3, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        # 2) Yield the first decoded image as a second intermediate result
        mid_tensor = get_value_at_index(vaedecode_8, 0)
        mid_np = (mid_tensor.cpu().numpy() * 255).astype(np.uint8)
        mid_np = mid_np[0]
        mid_pil = Image.fromarray(mid_np)
        yield mid_pil, "First enhancement pass complete (step 2/3)… refining details"

        controlnetapplyadvanced_20 = controlnetapplyadvanced.apply_controlnet(
            strength=1,
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
            yield pre_upscale_pil, "Enhancement complete (step 3/4)... upscaling image"

            # Upscale the final image
            upscaled = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscalemodelloader_30, 0),
                image=get_value_at_index(vaedecode_21, 0),
            )

            image_tensor = get_value_at_index(upscaled, 0)
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            image_np = image_np[0]
            pil_image = Image.fromarray(image_np)
            yield pil_image, f"No errors, all good! Final QR art generated and upscaled. (step 4/4) | Seed: {seed}"
        else:
            # No upscaling
            image_tensor = get_value_at_index(vaedecode_21, 0)
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            image_np = image_np[0]
            pil_image = Image.fromarray(image_np)
            yield pil_image, f"No errors, all good! Final QR art generated. | Seed: {seed}"

def _pipeline_artistic(prompt: str, qr_text: str, input_type: str, image_size: int, border_size: int, error_correction: str, module_size: int, module_drawer: str, seed: int, enable_upscale: bool = True, freeu_b1: float = 1.4, freeu_b2: float = 1.3, freeu_s1: float = 0.0, freeu_s2: float = 1.3, enable_sag: bool = True, sag_scale: float = 0.5, sag_blur_sigma: float = 1.5, controlnet_strength_first: float = 0.45, controlnet_strength_final: float = 0.7):
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
            "Try with a shorter text, increase the image size, or decrease the border size, module size, and error correction level under Advanced Settings."
        )
        yield None, error_msg
        return

    # Show the base QR code
    base_qr_tensor = get_value_at_index(comfy_qr, 0)
    base_qr_np = (base_qr_tensor.cpu().numpy() * 255).astype(np.uint8)
    base_qr_np = base_qr_np[0]
    base_qr_pil = Image.fromarray(base_qr_np)

    # Only add noise if there's a border (border_size > 0)
    if border_size > 0:
        yield base_qr_pil, "Generated base QR pattern... adding QR-like cubics to border (step 1/5)"

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
        yield noisy_qr_pil, "Added QR-like cubics to border... enhancing with AI (step 2/5)"
    else:
        # No border, skip noise
        qr_with_border_noise = get_value_at_index(comfy_qr, 0)
        yield base_qr_pil, "Generated base QR pattern (no border)... enhancing with AI (step 1/4)"

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
        s2=freeu_s2   # Skip connection dampening (layer 2) - customizable scannability balance
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

    # First decode
    decoded = vaedecode.decode(
        samples=get_value_at_index(samples, 0),
        vae=get_value_at_index(checkpointloadersimple_artistic, 2),
    )

    # Show first pass result
    first_pass_tensor = get_value_at_index(decoded, 0)
    first_pass_np = (first_pass_tensor.cpu().numpy() * 255).astype(np.uint8)
    first_pass_np = first_pass_np[0]
    first_pass_pil = Image.fromarray(first_pass_np)
    step_msg = "First enhancement pass complete (step 3/5)... final refinement pass" if border_size > 0 else "First enhancement pass complete (step 2/4)... final refinement pass"
    yield first_pass_pil, step_msg

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

    # Final decode
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
        step_msg = "Final refinement complete (step 4/5)... upscaling image" if border_size > 0 else "Final refinement complete (step 3/4)... upscaling image"
        yield pre_upscale_pil, step_msg

        # Upscale image with model (after final samples, before returning)
        upscaled = imageupscalewithmodel.upscale(
            upscale_model=get_value_at_index(upscalemodelloader_30, 0),
            image=get_value_at_index(final_decoded, 0),
        )

        # Convert upscaled image to PIL Image and return
        image_tensor = get_value_at_index(upscaled, 0)
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        image_np = image_np[0]
        final_image = Image.fromarray(image_np)
        step_msg = f"No errors, all good! Final artistic QR code generated and upscaled. (step 5/5) | Seed: {seed}" if border_size > 0 else f"No errors, all good! Final artistic QR code generated and upscaled. (step 4/4) | Seed: {seed}"
        yield final_image, step_msg
    else:
        # No upscaling
        image_tensor = get_value_at_index(final_decoded, 0)
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        image_np = image_np[0]
        final_image = Image.fromarray(image_np)
        step_msg = f"No errors, all good! Final artistic QR code generated. (step 4/4) | Seed: {seed}" if border_size > 0 else f"No errors, all good! Final artistic QR code generated. (step 3/3) | Seed: {seed}"
        yield final_image, step_msg


if __name__ == "__main__" and not os.environ.get('QR_TESTING_MODE'):

    # Start your Gradio app
    with gr.Blocks() as app:
        # Add a title and description
        gr.Markdown("# QR Code Art Generator")
        gr.Markdown("""
        This is an AI-powered QR code generator that creates artistic QR codes using Stable Diffusion 1.5 and ControlNet models.
        The application uses a custom ComfyUI workflow to generate QR codes.

        ### Tips:
        - Use detailed prompts for better results
        - Include style keywords like 'photorealistic', 'detailed', '8k'
        - Choose **URL** mode for web links or **Plain Text** mode for VCARD, WiFi credentials, calendar events, etc.
        - Try the examples below for inspiration

        ### Two Modes:
        - **Standard QR**: Stable, accurate QR code generation (faster, more scannable)
        - **Artistic QR**: More artistic and creative results with upscaling (slower, more creative)

        ### Note:
        Feel free to share your suggestions or feedback on how to improve the app! Thanks!
        """)

        # Add tabs for different generation methods
        with gr.Tabs():
            # STANDARD QR TAB
            with gr.TabItem("Standard QR"):
                with gr.Row():
                    with gr.Column():
                        # Add input type selector
                        input_type = gr.Radio(
                            choices=["URL", "Plain Text"],
                            value="URL",
                            label="Input Type",
                            info="URL: For web links (auto-removes https://). Plain Text: For VCARD, WiFi, calendar, location, etc. (no manipulation)"
                        )

                        # Add inputs
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the image you want to generate (check examples below for inspiration)",
                            value="Enter your prompt here... For example: 'a beautiful sunset over mountains, photorealistic, detailed landscape'",
                            lines=3
                        )
                        text_input = gr.Textbox(
                            label="QR Code Content",
                            placeholder="Enter URL or plain text",
                            value="Enter your URL or text here... For example: https://github.com",
                            lines=3
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            # Add image size slider
                            image_size = gr.Slider(
                                minimum=512,
                                maximum=1024,
                                step=64,
                                value=512,
                                label="Image Size",
                                info="Base size of the generated image. Final output will be 2x this size (e.g., 512 → 1024) due to the two-step enhancement process. Higher values use more VRAM and take longer to process."
                            )

                            # Add border size slider
                            border_size = gr.Slider(
                                minimum=0,
                                maximum=8,
                                step=1,
                                value=4,
                                label="QR Code Border Size",
                                info="Number of modules (squares) to use as border around the QR code. Higher values add more whitespace."
                            )

                            # Add error correction dropdown
                            error_correction = gr.Dropdown(
                                choices=["Low (7%)", "Medium (15%)", "Quartile (25%)", "High (30%)"],
                                value="Medium (15%)",
                                label="Error Correction Level",
                                info="Higher error correction makes the QR code more scannable when damaged or obscured, but increases its size and complexity. Medium (15%) is a good starting point for most uses."
                            )

                            # Add module size slider
                            module_size = gr.Slider(
                                minimum=4,
                                maximum=16,
                                step=1,
                                value=12,
                                label="QR Module Size",
                                info="Pixel width of the smallest QR code unit. Larger values improve readability but require a larger image size. 12 is a good starting point."
                            )

                            # Add module drawer dropdown with style examples
                            module_drawer = gr.Dropdown(
                                choices=["Square", "Gapped Square", "Circle", "Rounded", "Vertical bars", "Horizontal bars"],
                                value="Square",
                                label="QR Code Style",
                                info="Select the style of the QR code modules (squares). See examples below. Different styles can give your QR code a unique look while maintaining scannability."
                            )

                            # Add style examples with labels
                            gr.Markdown("### Style Examples:")

                            # First row of examples
                            with gr.Row():
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("**Square**", show_label=False)
                                    gr.Image("custom_nodes/ComfyQR/img/square.png", width=100, show_label=False, show_download_button=False)
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("**Gapped Square**", show_label=False)
                                    gr.Image("custom_nodes/ComfyQR/img/gapped_square.png", width=100, show_label=False, show_download_button=False)
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("**Circle**", show_label=False)
                                    gr.Image("custom_nodes/ComfyQR/img/circle.png", width=100, show_label=False, show_download_button=False)

                            # Second row of examples
                            with gr.Row():
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("**Rounded**", show_label=False)
                                    gr.Image("custom_nodes/ComfyQR/img/rounded.png", width=100, show_label=False, show_download_button=False)
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("**Vertical Bars**", show_label=False)
                                    gr.Image("custom_nodes/ComfyQR/img/vertical-bars.png", width=100, show_label=False, show_download_button=False)
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("**Horizontal Bars**", show_label=False)
                                    gr.Image("custom_nodes/ComfyQR/img/horizontal-bars.png", width=100, show_label=False, show_download_button=False)

                            # Add upscale checkbox
                            enable_upscale = gr.Checkbox(
                                label="Enable Upscaling",
                                value=False,
                                info="Enable upscaling with RealESRGAN for higher quality output (disabled by default for standard pipeline)"
                            )

                            # Add seed controls
                            use_custom_seed = gr.Checkbox(
                                label="Use Custom Seed",
                                value=True,
                                info="Enable to use a specific seed for reproducible results"
                            )
                            seed = gr.Slider(
                                minimum=0,
                                maximum=2000000,
                                step=1,
                                value=718313,
                                label="Seed",
                                info="Seed value for reproducibility. Same seed with same settings will produce the same result."
                            )

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

                # When clicking the button, it will trigger the main function
                generate_btn.click(
                    fn=generate_standard_qr,
                    inputs=[prompt_input, text_input, input_type, image_size, border_size, error_correction, module_size, module_drawer, use_custom_seed, seed, enable_upscale],
                    outputs=[output_image, error_message]
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
                        "Square"
                    ],
                    [
                        "some cards on poker tale, realistic, great details, realistic, great details,absence of people, Detailed and Intricate, CGI, Photoshoot,rim light, 8k, 16k, ultra detail",
                        "https://store.steampowered.com",
                        "URL",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "a beautiful sunset over mountains, photorealistic, detailed landscape, golden hour, dramatic lighting, 8k, ultra detailed",
                        "https://github.com",
                        "URL",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "underwater scene with coral reef and tropical fish, photorealistic, detailed, crystal clear water, sunlight rays, 8k, ultra detailed",
                        "https://twitter.com",
                        "URL",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "futuristic cityscape with flying cars and neon lights, cyberpunk style, detailed architecture, night scene, 8k, ultra detailed",
                        "https://linkedin.com",
                        "URL",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "vintage camera on wooden table, photorealistic, detailed textures, soft lighting, bokeh background, 8k, ultra detailed",
                        "https://instagram.com",
                        "URL",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "business card design, professional, modern, clean layout, corporate style, detailed, 8k, ultra detailed",
                        "BEGIN:VCARD\nVERSION:3.0\nFN:John Doe\nORG:Acme Corporation\nTITLE:Software Engineer\nTEL:+1-555-123-4567\nEMAIL:john.doe@example.com\nEND:VCARD",
                        "Plain Text",
                        832,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "wifi network symbol, modern tech, digital art, glowing blue, detailed, 8k, ultra detailed",
                        "WIFI:T:WPA;S:MyNetwork;P:MyPassword123;;",
                        "Plain Text",
                        576,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "calendar appointment reminder, organized planner, professional office, detailed, 8k, ultra detailed",
                        "BEGIN:VEVENT\nSUMMARY:Team Meeting\nDTSTART:20251115T140000Z\nDTEND:20251115T150000Z\nLOCATION:Conference Room A\nEND:VEVENT",
                        "Plain Text",
                        832,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "location pin on map, travel destination, scenic view, detailed cartography, 8k, ultra detailed",
                        "geo:37.7749,-122.4194",
                        "Plain Text",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ]
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
                        module_drawer
                    ],
                    outputs=[output_image, error_message],
                    fn=generate_standard_qr,
                    cache_examples=False
                )

            # ARTISTIC QR TAB
            with gr.TabItem("Artistic QR"):
                with gr.Row():
                    with gr.Column():
                        # Add input type selector for artistic QR
                        artistic_input_type = gr.Radio(
                            choices=["URL", "Plain Text"],
                            value="URL",
                            label="Input Type",
                            info="URL: For web links (auto-removes https://). Plain Text: For VCARD, WiFi, calendar, location, etc. (no manipulation)"
                        )

                        # Add inputs for artistic QR
                        artistic_prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the image you want to generate (check examples below for inspiration)",
                            value="Enter your prompt here... For example: 'a beautiful sunset over mountains, photorealistic, detailed landscape'",
                            lines=3
                        )
                        artistic_text_input = gr.Textbox(
                            label="QR Code Content",
                            placeholder="Enter URL or plain text",
                            value="Enter your URL or text here... For example: https://github.com",
                            lines=3
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            # Add image size slider for artistic QR
                            artistic_image_size = gr.Slider(
                                minimum=512,
                                maximum=1024,
                                step=64,
                                value=704,
                                label="Image Size",
                                info="Base size of the generated image. Final output will be 2x this size (e.g., 704 → 1408) due to the two-step enhancement process. Higher values use more VRAM and take longer to process."
                            )

                            # Add border size slider for artistic QR
                            artistic_border_size = gr.Slider(
                                minimum=0,
                                maximum=8,
                                step=1,
                                value=6,
                                label="QR Code Border Size",
                                info="Number of modules (squares) to use as border around the QR code. Higher values add more whitespace."
                            )

                            # Add error correction dropdown for artistic QR
                            artistic_error_correction = gr.Dropdown(
                                choices=["Low (7%)", "Medium (15%)", "Quartile (25%)", "High (30%)"],
                                value="High (30%)",
                                label="Error Correction Level",
                                info="Higher error correction makes the QR code more scannable when damaged or obscured, but increases its size and complexity. High (30%) is recommended for artistic QR codes."
                            )

                            # Add module size slider for artistic QR
                            artistic_module_size = gr.Slider(
                                minimum=4,
                                maximum=16,
                                step=1,
                                value=16,
                                label="QR Module Size",
                                info="Pixel width of the smallest QR code unit. Larger values improve readability but require a larger image size. 16 is a good starting point."
                            )

                            # Add module drawer dropdown with style examples for artistic QR
                            artistic_module_drawer = gr.Dropdown(
                                choices=["Square", "Gapped Square", "Circle", "Rounded", "Vertical bars", "Horizontal bars"],
                                value="Square",
                                label="QR Code Style",
                                info="Select the style of the QR code modules (squares). See examples below. Different styles can give your QR code a unique look while maintaining scannability."
                            )

                            # Add style examples with labels
                            gr.Markdown("### Style Examples:")

                            # First row of examples
                            with gr.Row():
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("**Square**", show_label=False)
                                    gr.Image("custom_nodes/ComfyQR/img/square.png", width=100, show_label=False, show_download_button=False)
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("**Gapped Square**", show_label=False)
                                    gr.Image("custom_nodes/ComfyQR/img/gapped_square.png", width=100, show_label=False, show_download_button=False)
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("**Circle**", show_label=False)
                                    gr.Image("custom_nodes/ComfyQR/img/circle.png", width=100, show_label=False, show_download_button=False)

                            # Second row of examples
                            with gr.Row():
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("**Rounded**", show_label=False)
                                    gr.Image("custom_nodes/ComfyQR/img/rounded.png", width=100, show_label=False, show_download_button=False)
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("**Vertical Bars**", show_label=False)
                                    gr.Image("custom_nodes/ComfyQR/img/vertical-bars.png", width=100, show_label=False, show_download_button=False)
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("**Horizontal Bars**", show_label=False)
                                    gr.Image("custom_nodes/ComfyQR/img/horizontal-bars.png", width=100, show_label=False, show_download_button=False)

                            # Add upscale checkbox
                            artistic_enable_upscale = gr.Checkbox(
                                label="Enable Upscaling",
                                value=True,
                                info="Enable upscaling with RealESRGAN for higher quality output (enabled by default for artistic pipeline)"
                            )

                            # Add seed controls for artistic QR
                            artistic_use_custom_seed = gr.Checkbox(
                                label="Use Custom Seed",
                                value=True,
                                info="Enable to use a specific seed for reproducible results"
                            )
                            artistic_seed = gr.Slider(
                                minimum=0,
                                maximum=2000000,
                                step=1,
                                value=718313,
                                label="Seed",
                                info="Seed value for reproducibility. Same seed with same settings will produce the same result."
                            )

                            # FreeU Parameters
                            gr.Markdown("### FreeU Quality Enhancement")
                            freeu_b1 = gr.Slider(
                                minimum=1.0,
                                maximum=1.6,
                                step=0.01,
                                value=1.4,
                                label="FreeU B1 (Backbone 1)",
                                info="Backbone feature enhancement for first layer. Higher values improve detail but may reduce blending. Range: 1.0-1.6, Default: 1.4"
                            )
                            freeu_b2 = gr.Slider(
                                minimum=1.0,
                                maximum=1.6,
                                step=0.01,
                                value=1.3,
                                label="FreeU B2 (Backbone 2)",
                                info="Backbone feature enhancement for second layer. Higher values improve texture. Range: 1.0-1.6, Default: 1.3"
                            )
                            freeu_s1 = gr.Slider(
                                minimum=0.0,
                                maximum=1.5,
                                step=0.01,
                                value=0.0,
                                label="FreeU S1 (Skip 1)",
                                info="Skip connection dampening for first layer. Lower values hide QR structure more. Range: 0.0-1.5, Default: 0.0"
                            )
                            freeu_s2 = gr.Slider(
                                minimum=0.0,
                                maximum=1.5,
                                step=0.01,
                                value=1.3,
                                label="FreeU S2 (Skip 2)",
                                info="Skip connection dampening for second layer. Balances scannability. Range: 0.0-1.5, Default: 1.3"
                            )

                            # SAG (Self-Attention Guidance) Parameters
                            gr.Markdown("### SAG (Self-Attention Guidance)")
                            enable_sag = gr.Checkbox(
                                label="Enable SAG",
                                value=True,
                                info="Enable Self-Attention Guidance for improved structural coherence and artistic blending"
                            )
                            sag_scale = gr.Slider(
                                minimum=0.0,
                                maximum=3.0,
                                step=0.1,
                                value=0.5,
                                label="SAG Scale",
                                info="Guidance strength. Higher values provide more structural coherence. Range: 0.0-3.0, Default: 0.5"
                            )
                            sag_blur_sigma = gr.Slider(
                                minimum=0.0,
                                maximum=5.0,
                                step=0.1,
                                value=1.5,
                                label="SAG Blur Sigma",
                                info="Blur amount for artistic blending. Higher values create softer, more artistic effects. Range: 0.0-5.0, Default: 1.5"
                            )

                        # The generate button for artistic QR
                        artistic_generate_btn = gr.Button("Generate Artistic QR", variant="primary")

                    with gr.Column():
                        # The output image for artistic QR
                        artistic_output_image = gr.Image(label="Generated Artistic QR Code")
                        artistic_error_message = gr.Textbox(
                            label="Status / Errors",
                            interactive=False,
                            lines=3,
                        )

                # When clicking the button, it will trigger the artistic function
                artistic_generate_btn.click(
                    fn=generate_artistic_qr,
                    inputs=[artistic_prompt_input, artistic_text_input, artistic_input_type, artistic_image_size, artistic_border_size, artistic_error_correction, artistic_module_size, artistic_module_drawer, artistic_use_custom_seed, artistic_seed, artistic_enable_upscale, freeu_b1, freeu_b2, freeu_s1, freeu_s2, enable_sag, sag_scale, sag_blur_sigma],
                    outputs=[artistic_output_image, artistic_error_message]
                )

                # Add examples for artistic QR
                artistic_examples = [
                    [
                        "some clothes spread on ropes, realistic, great details, out in the open air sunny day realistic, great details, absence of people, Detailed and Intricate, CGI, Photoshoot, rim light, 8k, 16k, ultra detail",
                        "https://www.google.com",
                        "URL",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "some cards on poker tale, realistic, great details, realistic, great details,absence of people, Detailed and Intricate, CGI, Photoshoot,rim light, 8k, 16k, ultra detail",
                        "https://store.steampowered.com",
                        "URL",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "a beautiful sunset over mountains, photorealistic, detailed landscape, golden hour, dramatic lighting, 8k, ultra detailed",
                        "https://github.com",
                        "URL",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "underwater scene with coral reef and tropical fish, photorealistic, detailed, crystal clear water, sunlight rays, 8k, ultra detailed",
                        "https://twitter.com",
                        "URL",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "futuristic cityscape with flying cars and neon lights, cyberpunk style, detailed architecture, night scene, 8k, ultra detailed",
                        "https://linkedin.com",
                        "URL",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "vintage camera on wooden table, photorealistic, detailed textures, soft lighting, bokeh background, 8k, ultra detailed",
                        "https://instagram.com",
                        "URL",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "business card design, professional, modern, clean layout, corporate style, detailed, 8k, ultra detailed",
                        "BEGIN:VCARD\nVERSION:3.0\nFN:John Doe\nORG:Acme Corporation\nTITLE:Software Engineer\nTEL:+1-555-123-4567\nEMAIL:john.doe@example.com\nEND:VCARD",
                        "Plain Text",
                        832,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "wifi network symbol, modern tech, digital art, glowing blue, detailed, 8k, ultra detailed",
                        "WIFI:T:WPA;S:MyNetwork;P:MyPassword123;;",
                        "Plain Text",
                        576,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "calendar appointment reminder, organized planner, professional office, detailed, 8k, ultra detailed",
                        "BEGIN:VEVENT\nSUMMARY:Team Meeting\nDTSTART:20251115T140000Z\nDTEND:20251115T150000Z\nLOCATION:Conference Room A\nEND:VEVENT",
                        "Plain Text",
                        832,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ],
                    [
                        "location pin on map, travel destination, scenic view, detailed cartography, 8k, ultra detailed",
                        "geo:37.7749,-122.4194",
                        "Plain Text",
                        512,
                        4,
                        "Medium (15%)",
                        12,
                        "Square"
                    ]
                ]

                gr.Examples(
                    examples=artistic_examples,
                    inputs=[
                        artistic_prompt_input,
                        artistic_text_input,
                        artistic_input_type,
                        artistic_image_size,
                        artistic_border_size,
                        artistic_error_correction,
                        artistic_module_size,
                        artistic_module_drawer
                    ],
                    outputs=[artistic_output_image, artistic_error_message],
                    fn=generate_artistic_qr,
                    cache_examples=False
                )

    app.launch(share=False, mcp_server=True)
