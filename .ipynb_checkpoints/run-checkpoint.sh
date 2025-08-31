#!/bin/bash
# export HF_HOME=$(pwd)/cache
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch train.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev"  \
  --output_dir="kontext-lora" \
  --save_lora_dir "lora_out" \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --guidance_scale=1.0 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --optimizer="adamw" \
  --report_to="wandb" \
  --data_dir="/workspace/train" \
  --rank=64 \
  --lora_alpha=64 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --num_train_epochs=10000 \
  --max_train_steps=2000 \
  --checkpointing_steps=20000 \
  --validation_epochs=50000 \
  --validation_prompt="Turn this image into the paper cutting style" \
  --num_validation_images=2 \
  --dataloader_num_workers=4 \
  --max_sequence_length=512 \
  --weighting_scheme="none" \
  --seed=42 