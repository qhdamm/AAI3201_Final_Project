export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="tini"
export OUTPUT_DIR="lora-trained-xl"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch --multi_gpu --num_processes 2  ./diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="tiniping character" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
   --report_to="wandb" \
  --max_train_steps=500 \
  --validation_prompt="tiniping character in the snow" \
  --validation_epochs=100 \
  --seed="0"  \
   --push_to_hub