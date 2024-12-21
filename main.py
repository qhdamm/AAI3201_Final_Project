import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection


def initialize_pipeline(base_model_path: str, lora_model_path: str, refiner_model_path: str, ip_adapter_path: str, device: str = "cuda"):
    """
    Initializes the pipeline, including IP-Adapter and LoRA weights.

    Parameters:
        base_model_path (str): Path to the base model.
        lora_model_path (str): Path to the LoRA-trained model.
        refiner_model_path (str): Path to the refiner model.
        ip_adapter_path (str): Path to the IP-Adapter weights.
        device (str): Device for model execution.

    Returns:
        tuple: pipeline and refiner objects.
    """
    # Load base model
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
    ).to(device)

    # Load LoRA
    pipeline.load_lora_weights(lora_model_path)

    # Load IP-Adapter
    pipeline.load_ip_adapter(
        ip_adapter_path,
        subfolder="sdxl_models",
        weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"],
    )
    pipeline.set_ip_adapter_scale([0.7, 0.3])
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # Load refiner model
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        refiner_model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    ).to(device)

    return pipeline, refiner


def generate_image(pipeline, refiner, prompt: str, negative_prompt: str, img_name: str, style_folder: str, device: str = "cuda"):
    """
    Generates an image using the initialized pipeline and refiner.

    Parameters:
        pipeline: Initialized Stable Diffusion XL pipeline.
        refiner: Initialized refiner pipeline.
        prompt (str): Text prompt for generation.
        negative_prompt (str): Negative text prompt.
        img_name (str): Name of the source image.
        style_folder (str): Path to style images folder.
        device (str): Device to execute on (default: "cuda").
    """
    # Load source and style images
    face_image = load_image(f"./source/{img_name}.jpg")
    style_images = [load_image(f"{style_folder}/tini{i}.png") for i in range(10)]

    generator = torch.Generator(device=device)

    # Generate initial image
    image = pipeline(
        prompt=prompt,
        ip_adapter_image=[style_images, face_image],
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]

    # Refine the image
    refined_image = refiner(
        prompt="character",
        image=image[None, :],
        generator=generator,
    ).images[0]

    # Save the image
    output_path = f"output/{img_name}_tini0703.png"
    refined_image.save(output_path)
    print(f"Image saved at {output_path}")


def main(img_name: str) -> None:
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_model_path = "bodam/lora-trained-xl"
    ip_adapter_path = "h94/IP-Adapter"
    refiner_model_path = "stabilityai/stable-diffusion-xl-refiner-1.0"

    prompt = "tiniping character"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    style_folder = "./tini"

    pipeline, refiner = initialize_pipeline(base_model_path, lora_model_path, refiner_model_path, ip_adapter_path)
    generate_image(pipeline, refiner, prompt, negative_prompt, img_name, style_folder)


if __name__ == "__main__":
    main("yoo")  # Pass source image name
