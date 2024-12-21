import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler, StableDiffusionXLImg2ImgPipeline
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image
from torchvision import transforms

def initialize_pipeline(ip_adapter_path: str, lora_model_path: str, refiner_model_path: str, device: str = "cuda"):
    """
    Initializes the pipeline, image encoder, and refiner model.

    Parameters:
        ip_adapter_path (str): Path to the IP-Adapter model.
        lora_model_path (str): Path to the LoRA-trained model. (Dreambooth fine-tuned model)
        refiner_model_path (str): Path to the refiner model.
        device (str): Device to load the models (default: "cuda").

    Returns:
        tuple: pipeline and refiner objects.
    """
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        ip_adapter_path,
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    ).to(device)

    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        image_encoder=image_encoder,
    ).to(device)
    pipeline.load_lora_weights(lora_model_path)

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    pipeline.load_ip_adapter(
        ip_adapter_path,
        subfolder="sdxl_models",
        weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"],
    )
    pipeline.set_ip_adapter_scale([0.7, 0.4])

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        refiner_model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    ).to(device)

    return pipeline, refiner


def generate_image(pipeline, refiner, prompt: str, negative_prompt: str, img_name: str, style_folder: str, device: str = "cuda"):
    """
    Generates an image using the initialized pipeline and refiner.
    -> Save the generated image.

    Parameters:
        pipeline: The initialized text-to-image pipeline.
        refiner: The initialized refiner pipeline.
        prompt (str): Text prompt for image generation.
        negative_prompt (str): Negative text prompt to filter out unwanted features.
        img_name (str): Name of the input image (without extension).
        style_folder (str): Path to the folder containing style images.
        device (str): Device for generating images (default: "cuda").
    """
    face_image = load_image(f"./source/{img_name}.jpg")
    style_images = [load_image(f"./{style_folder}/tini{i}.png") for i in range(9)]

    generator = torch.Generator(device=device)

    image = pipeline(
        prompt=prompt,
        ip_adapter_image=[style_images, face_image],
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Refine the generated image for better quality
    image = refiner(prompt=prompt, image=image_tensor, generator=generator).images[0]

    # Save the generated image
    output_path = f"output/{img_name}.png"
    image.save(output_path)
    print(f"Image saved at {output_path}")

def main(img_name: str):
    ip_adapter_path = "h94/IP-Adapter"
    lora_model_path = "bodam/lora-trained-xl"
    refiner_model_path = "stabilityai/stable-diffusion-xl-refiner-1.0"

    # prompt definition
    prompt = "a tiniping dog, no background, no glow, high quality, detailed, colorful"
    negative_prompt = "worst quality, low quality"
    style_folder = "tini"

    # Initialize pipeline and refiner
    pipeline, refiner = initialize_pipeline(ip_adapter_path, lora_model_path, refiner_model_path)

    # Generate and refine image
    generate_image(pipeline, refiner, prompt, negative_prompt, img_name, style_folder)

if __name__ == "__main__":
    main("dog")  # parameter: source image 
