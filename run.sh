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
  --data_dir="/workspace/data/train" \
  --rank=64 \
  --lora_alpha=64 \
  --learning_rate=5e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --num_train_epochs=100000 \
  --max_train_steps=5000 \
  --checkpointing_steps=500 \
  --dataloader_num_workers=4 \
  --max_sequence_length=512 \
  --weighting_scheme="none" \
  --seed=42 \
  --validation_steps 100 \
  --num_validation_samples 4 \
  --validation_inference_steps 30