# Pingify

## Overview

**Pingify** is a project aimed at generating **Tiniping-style images** from human or animal photos. This tool provides a creative and entertaining way to interact with the Tiniping animation series by transforming personal images into Tiniping characters. It can also support character designers in their creative process.

## Features

- **DreamBooth Fine-Tuning**: Personalizes image generation using small datasets and text prompts.
- **IP-Adapter Integration**: Enables effective multimodal generation with lightweight modules.
- **Multi IP-Adapter Support**: Combines multiple adapters to achieve diverse styles and consistent character designs.
- **Image-to-Image Generation**: Transforms source images into Tiniping-style images with enhanced quality using SDXL Refiners.

## Workflow

1. **Fine-Tune the SDXL Model with DreamBooth**:
   - Reference images: Tiniping characters.
   - Text prompt: `tiniping character`.
2. **Generate Tiniping-Style Images**:
   - Input: Source image + style image + positive/negative prompts.
   - Output: Source image transformed into Tiniping style.
3. **Enhance Image Quality**:
   - Apply SDXL Refiner for better results.

## Results

- Successfully transformed both human and animal photos into Tiniping-style images.
- Demonstrated the adaptability and creative potential of the approach.

## Technical Details

### Models & Techniques
- **Base Model**: [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **Refiner**: [SDXL Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
- **Adapters**:
  - **IP-Adapter-Face**: Ensures consistent face generation.
  - **IP-Adapter-Plus**: Generates images in specific styles.

### Codebase
- Fine-tuned the model using DreamBooth.
- Integrated LoRA weights into the IP-Adapter pipeline.
- Modified prompts and generation parameters for optimal results.

### References
- [DreamBooth Documentation](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)
- [IP-Adapter Documentation](https://huggingface.co/docs/diffusers/main/using-diffusers/ip_adapter)
