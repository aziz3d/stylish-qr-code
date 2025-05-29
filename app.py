import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import gradio as gr
from PIL import Image
import numpy as np

# import spaces


from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5", filename="v1-5-pruned-emaonly.ckpt", local_dir="models/checkpoints")
hf_hub_download(repo_id="Lykon/DreamShaper", filename="DreamShaper_3.32_baked_vae_clip_fix_half.safetensors", local_dir="models/checkpoints")
hf_hub_download(repo_id="latentcat/latentcat-controlnet", filename="models/control_v1p_sd15_brightness.safetensors", local_dir="models/controlnet")
hf_hub_download(repo_id="comfyanonymous/ControlNet-v1-1_fp16_safetensors", filename="control_v11f1e_sd15_tile_fp16.safetensors", local_dir="models/controlnet")
hf_hub_download(repo_id="Lykon/dreamshaper-7", filename="vae/diffusion_pytorch_model.fp16.safetensors", local_dir="models")
hf_hub_download(repo_id="stabilityai/sd-vae-ft-mse-original", filename="vae-ft-mse-840000-ema-pruned.safetensors", local_dir="models/vae")

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

checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
    ckpt_name="DreamShaper_3.32_baked_vae_clip_fix_half.safetensors"
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

from comfy import model_management

#Add all the models that load a safetensors file
model_loaders = [checkpointloadersimple_4]

# Check which models are valid and how to best load them
valid_models = [
    getattr(loader[0], 'patcher', loader[0]) 
    for loader in model_loaders
    if not isinstance(loader[0], dict) and not isinstance(getattr(loader[0], 'patcher', None), dict)
]

model_management.load_models_gpu(valid_models)

# @spaces.GPU(duration=60)
def generate_qr_code(prompt: str, url: str):
    if "https://" in url:
        url = url.replace("https://", "")
    if "http://" in url:
        url = url.replace("http://", "")

    with torch.inference_mode():

        emptylatentimage_5 = emptylatentimage.generate(
            width=512, height=512, batch_size=1
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

        comfy_qr_by_module_size_15 = comfy_qr_by_module_size.generate_qr(
            protocol="Https",
            text=url,
            module_size=12,
            max_image_size=512,
            fill_hexcolor="#000000",
            back_hexcolor="#FFFFFF",
            error_correction="Medium",
            border=4,
            module_drawer="Square",
        )

        emptylatentimage_17 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )

        controlnetloader_19 = controlnetloader.load_controlnet(
            control_net_name="control_v11f1e_sd15_tile_fp16.safetensors"
        )

        # saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

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
                resolution=512,
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
                seed=random.randint(1, 2**64),
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

            # saveimage_9 = saveimage.save_images(
            #     filename_prefix="qr-new", images=get_value_at_index(vaedecode_8, 0)
            # )

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
                seed=random.randint(1, 2**64),
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

            # saveimage_22 = saveimage.save_images(
            #     filename_prefix="qr-new-improved",
            #     images=get_value_at_index(vaedecode_21, 0),
            # )

            # Convert torch tensor to PIL Image
            image_tensor = get_value_at_index(vaedecode_21, 0)
            # Convert from [0,1] to [0,255] range and to uint8
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            # Remove batch dimension and convert to PIL Image
            image_np = image_np[0]  # Shape will be (1024, 1024, 3)
            pil_image = Image.fromarray(image_np)
            return pil_image


if __name__ == "__main__":

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
        - Try the examples below for inspiration
        """)

        with gr.Row():
            with gr.Column():
                # Add inputs
                prompt_input = gr.Textbox(
                    label="Prompt", 
                    placeholder="Describe the image you want to generate (check examples below for inspiration)",
                    value="Enter your prompt here... For example: 'a beautiful sunset over mountains, photorealistic, detailed landscape'"
                )
                url_input = gr.Textbox(
                    label="URL for QR Code",
                    placeholder="Enter the URL you want to convert into a QR code (e.g., https://example.com)",
                    value="Enter your URL here... For example: https://github.com"
                )
                # The generate button
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                # The output image
                output_image = gr.Image(label="Generated QR Code Art")

            # When clicking the button, it will trigger the main function
            generate_btn.click(
                fn=generate_qr_code,
                inputs=[prompt_input, url_input],
                outputs=[output_image]
            )

        # Add examples
        examples = [
            [
                "some clothes spread on ropes, realistic, great details, out in the open air sunny day realistic, great details,absence of people, Detailed and Intricate, CGI, Photoshoot,rim light, 8k, 16k, ultra detail",
                "https://www.google.com"
            ],
            [
                "a beautiful sunset over mountains, photorealistic, detailed landscape, golden hour, dramatic lighting, 8k, ultra detailed",
                "https://github.com"
            ],
            [
                "underwater scene with coral reef and tropical fish, photorealistic, detailed, crystal clear water, sunlight rays, 8k, ultra detailed",
                "https://twitter.com"
            ],
            [
                "futuristic cityscape with flying cars and neon lights, cyberpunk style, detailed architecture, night scene, 8k, ultra detailed",
                "https://linkedin.com"
            ],
            [
                "vintage camera on wooden table, photorealistic, detailed textures, soft lighting, bokeh background, 8k, ultra detailed",
                "https://instagram.com"
            ]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[prompt_input, url_input],
            outputs=[output_image],
            fn=generate_qr_code,
            cache_examples=True
        )

        app.launch(share=False, mcp_server=True)

