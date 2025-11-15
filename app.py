import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import gradio as gr
from PIL import Image
import numpy as np

import spaces


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

@spaces.GPU(duration=20)
def generate_qr_code(prompt: str, text_input: str, input_type: str = "URL", image_size: int = 512, border_size: int = 4, error_correction: str = "Medium (15%)", module_size: int = 12, module_drawer: str = "Square"):
    # Only manipulate the text if it's a URL input type
    qr_text = text_input
    if input_type == "URL":
        if "https://" in qr_text:
            qr_text = qr_text.replace("https://", "")
        if "http://" in qr_text:
            qr_text = qr_text.replace("http://", "")

    with torch.inference_mode():

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
            # Stream a single error message
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

            # 3) Yield the final enhanced image
            image_tensor = get_value_at_index(vaedecode_21, 0)
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            image_np = image_np[0]
            pil_image = Image.fromarray(image_np)
            yield pil_image, "No errors, all good! Final QR art generated."


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
        - Choose **URL** mode for web links or **Plain Text** mode for VCARD, WiFi credentials, calendar events, etc.
        - Try the examples below for inspiration

        ### Note:
        Feel free to share your suggestions or feedback on how to improve the app! Thanks!
       
         """)

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

                # The generate button
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                # The output image
                output_image = gr.Image(label="Generated QR Code Art")
                error_message = gr.Textbox(
                    label="Status / Errors",
                    interactive=False,
                    lines=3,
                )

            # When clicking the button, it will trigger the main function
            generate_btn.click(
                fn=generate_qr_code,
                inputs=[prompt_input, text_input, input_type, image_size, border_size, error_correction, module_size, module_drawer],
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
                512,
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
            fn=generate_qr_code,
            cache_examples=False
        )

        app.launch(share=False, mcp_server=True)

