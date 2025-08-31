#!/usr/bin/env python
# coding=utf-8
# Modified from original diffusers dreambooth training script for multi-context support
# UPDATED: Uses PEFT for LoRA + simple save flags for adapters and merged single-file model.

import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, get_peft_model  # NEW: PEFT
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

check_min_version("0.34.0.dev0")
logger = get_logger(__name__)


class MultiContextDataset(Dataset):
    def __init__(self, root_dir, max_context_images=6, enable_assets=False):
        self.root_dir = root_dir
        self.max_context_images = max_context_images
        self.enable_assets = enable_assets
        
        self.samples = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('sample_')
        ])
        
        self.PREFERRED_RESOLUTIONS = [
            (672, 1568), (688, 1504), (720, 1456), (752, 1392),
            (800, 1328), (832, 1248), (880, 1184), (944, 1104),
            (1024, 1024), (1104, 944), (1184, 880), (1248, 832),
            (1328, 800), (1392, 752), (1456, 720), (1504, 688), (1568, 672)
        ]
        
    def find_best_resolution(self, img):
        w, h = img.size
        aspect_ratio = w / h
        
        best_match = None
        min_diff = float('inf')
        
        for res_w, res_h in self.PREFERRED_RESOLUTIONS:
            res_ar = res_w / res_h
            diff = abs(aspect_ratio - res_ar)
            if diff < min_diff:
                min_diff = diff
                best_match = (res_w, res_h)
        
        return best_match
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_dir = os.path.join(self.root_dir, self.samples[idx])
        
        # Load prompt
        with open(os.path.join(sample_dir, 'prompt.txt'), 'r') as f:
            prompt = f.read().strip()
        
        # Load and process target image
        target_path = os.path.join(sample_dir, 'out.jpg')
        if not os.path.exists(target_path):
            for ext in ['.png', '.jpeg']:
                alt_path = os.path.join(sample_dir, f'out{ext}')
                if os.path.exists(alt_path):
                    target_path = alt_path
                    break
        
        target_img = Image.open(target_path).convert('RGB')
        target_w, target_h = self.find_best_resolution(target_img)
        target_img = target_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        target_tensor = transform(target_img)
        
        # Load context images
        context_tensors = []
        context_dir = os.path.join(sample_dir, 'in')
        if os.path.exists(context_dir):
            context_files = sorted([
                f for f in os.listdir(context_dir) 
                if f.endswith(('.jpg', '.jpeg', '.png'))
            ])
            for img_file in context_files[:self.max_context_images]:
                img = Image.open(os.path.join(context_dir, img_file)).convert('RGB')
                ctx_w, ctx_h = self.find_best_resolution(img)
                img = img.resize((ctx_w, ctx_h), Image.Resampling.LANCZOS)
                context_tensors.append(transform(img))
        
        # Load asset images
        asset_tensors = []
        if self.enable_assets:
            assets_dir = os.path.join(sample_dir, 'assets')
            if os.path.exists(assets_dir):
                asset_files = sorted([
                    f for f in os.listdir(assets_dir)
                    if f.endswith(('.jpg', '.jpeg', '.png'))
                ])
                for img_file in asset_files:
                    img = Image.open(os.path.join(assets_dir, img_file)).convert('RGB')
                    asset_w, asset_h = self.find_best_resolution(img)
                    img = img.resize((asset_w, asset_h), Image.Resampling.LANCZOS)
                    asset_tensors.append(transform(img))
        
        return {
            "txt": prompt,
            "img": target_tensor,
            "context_images": context_tensors,
            "asset_images": asset_tensors
        }


def save_model_card(repo_id, images=None, base_model=None, train_text_encoder=False,
                   instance_prompt=None, validation_prompt=None, repo_folder=None):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append({
                "text": validation_prompt if validation_prompt else " ",
                "output": {"url": f"image_{i}.png"}
            })

    model_description = f"""
# Flux Kontext Multi-Context LoRA - {repo_id}

## Model description
Multi-context LoRA weights for {base_model} trained with multiple reference images.

Was LoRA for the text encoder enabled? {train_text_encoder}.

## Trigger words
You should use `{instance_prompt}` to trigger the image generation.

## Download model
[Download]({repo_id}/tree/main) the *.safetensors LoRA in the Files & versions tab.
"""
    
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    
    tags = ["text-to-image", "diffusers-training", "diffusers", "lora", "flux", "flux-kontext", "template:sd-lora"]
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def load_text_encoders(class_one, class_two, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision, subfolder="text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Multi-context Flux Kontext LoRA training script.")
    
    # Model arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True,
                       help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False,
                       help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None,
                       help="Variant of the model files of the pretrained model identifier")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, default="data/train",
                       help="Directory containing training data")
    parser.add_argument("--max_context_images", type=int, default=6,
                       help="Maximum number of context images to use")
    parser.add_argument("--time_spacing", type=float, default=1.0,
                       help="Spacing between tau values for context images")
    parser.add_argument("--enable_assets", action="store_true",
                       help="Enable loading asset images")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="flux-kontext-lora-multi",
                       help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=1024,
                       help="The resolution for input images")
    parser.add_argument("--train_batch_size", type=int, default=1,
                       help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None,
                       help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                       help="Save a checkpoint of the training state every X updates")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None,
                       help="Max number of checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Whether training should be resumed from a previous checkpoint.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Whether or not to use gradient checkpointing to save memory")
    
    # LoRA arguments
    parser.add_argument("--rank", type=int, default=64,
                       help="The dimension of the LoRA update matrices.")
    parser.add_argument("--lora_alpha", type=int, default=64,
                       help="LoRA alpha to be used for additional scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="Dropout probability for LoRA layers")
    parser.add_argument("--train_text_encoder", action="store_true",
                       help="Whether to train the text encoder")
    
    # Optimizer arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False,
                       help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                       help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                       help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                       help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0,
                       help="Power factor of the polynomial scheduler.")
    
    parser.add_argument("--optimizer", type=str, default="AdamW",
                       help='The optimizer type to use. Choose between ["AdamW", "prodigy"]')
    parser.add_argument("--use_8bit_adam", action="store_true",
                       help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="The beta1 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="The beta2 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04,
                       help="Weight decay to use for unet params")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                       help="Epsilon value for the Adam optimizer and Prodigy optimizers.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                       help="Max gradient norm.")
    
    # Other arguments
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                       help="Guidance scale for generation")
    parser.add_argument("--vae_encode_mode", type=str, default="mode", choices=["sample", "mode"],
                       help="VAE encoding mode.")
    parser.add_argument("--weighting_scheme", type=str, default="none",
                       choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
                       help='Weighting scheme for loss')
    parser.add_argument("--logit_mean", type=float, default=0.0,
                       help="mean to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--logit_std", type=float, default=1.0,
                       help="std to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--mode_scale", type=float, default=1.29,
                       help="Scale of mode weighting scheme.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                       help="Number of subprocesses to use for data loading.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                       help="Whether to use mixed precision.")
    parser.add_argument("--allow_tf32", action="store_true",
                       help="Whether or not to allow TF32 on Ampere GPUs.")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                       help='The integration to report the results and logs to.')
    parser.add_argument("--validation_prompt", type=str, default=None,
                       help="A prompt that is used during validation to verify that the model is learning.")
    parser.add_argument("--num_validation_images", type=int, default=4,
                       help="Number of images that should be generated during validation")
    parser.add_argument("--validation_epochs", type=int, default=50,
                       help="Run validation every X epochs.")
    parser.add_argument("--logging_dir", type=str, default="logs",
                       help="TensorBoard log directory.")
    parser.add_argument("--hub_token", type=str, default=None,
                       help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None,
                       help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                       help="Maximum sequence length to use with the T5 text encoder")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="For distributed training: local_rank")

    # NEW: simple saving flags
    parser.add_argument("--save_lora_dir", type=str, default=None,
                        help="If set, save PEFT adapters (adapter_model.safetensors + adapter_config.json) here.")
    parser.add_argument("--merge_and_save_full", type=str, default=None,
                        help="If set, merge LoRA into base and save full unsharded model here.")
    parser.add_argument("--max_shard_size", type=str, default="100GB",
                        help="Shard threshold for safetensors; set large to force a single file.")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=max_sequence_length,
        truncation=True, return_length=False, return_overflowing_tokens=False,
        return_tensors="pt"
    )
    return text_inputs.input_ids


def _encode_prompt_with_t5(text_encoder, tokenizer, max_sequence_length=512,
                          prompt=None, num_images_per_prompt=1, device=None, text_input_ids=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_length=False, return_overflowing_tokens=False,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, device=None,
                            text_input_ids=None, num_images_per_prompt=1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True,
            return_overflowing_tokens=False, return_length=False, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
    return prompt_embeds


def encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length,
                 device=None, num_images_per_prompt=1, text_input_ids_list=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0], tokenizer=tokenizers[0],
        prompt=prompt, device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1], tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length, prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Create dataset
    train_dataset = MultiContextDataset(
        root_dir=args.data_dir,
        max_context_images=args.max_context_images,
        enable_assets=args.enable_assets
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )

    # Import text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two, args)
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant
    )
    
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant
    )

    # Freeze base model weights
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Set weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device and dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    # ---------------------------
    # NEW: Wrap with PEFT LoRA
    # ---------------------------
    target_modules = [
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
        "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2"
    ]
    lcfg_xf = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    transformer = get_peft_model(transformer, lcfg_xf)

    if args.train_text_encoder:
        lcfg_te = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one = get_peft_model(text_encoder_one, lcfg_te)

    # Set up optimizer (LoRA params only; PEFT marks them requires_grad=True)
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * 
            args.train_batch_size * accelerator.num_processes
        )
    params_to_optimize = [{"params": [p for p in transformer.parameters() if p.requires_grad],
                           "lr": args.learning_rate}]
    if args.train_text_encoder:
        params_to_optimize.append({
            "params": [p for p in text_encoder_one.parameters() if p.requires_grad],
            "weight_decay": args.adam_weight_decay,
            "lr": args.learning_rate
        })

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=False
    )

    # Compute text embeddings if not training text encoder
    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # Get VAE config values
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

    # Set up scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with accelerator
    if args.train_text_encoder:
        transformer, text_encoder_one, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, text_encoder_one, optimizer, train_dataloader, lr_scheduler
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # Set up trackers
    if accelerator.is_main_process:
        tracker_name = "flux-kontext-multi-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
    
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
                
            with accelerator.accumulate(models_to_accumulate):
                # Since batch_size=1, extract the single sample
                prompts = batch["txt"]
                target_images = batch["img"]
                context_images_list = batch["context_images"][0] if batch["context_images"] else []
                asset_images_list = batch["asset_images"][0] if batch["asset_images"] else []
    
                # Encode prompts
                if not args.train_text_encoder:
                    prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                        prompts, text_encoders, tokenizers
                    )
                else:
                    tokens_one = tokenize_prompt(tokenizer_one, prompts, max_sequence_length=77)
                    tokens_two = tokenize_prompt(tokenizer_two, prompts, max_sequence_length=args.max_sequence_length)
                    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=[None, None],
                        text_input_ids_list=[tokens_one, tokens_two],
                        max_sequence_length=args.max_sequence_length,
                        device=accelerator.device,
                        prompt=prompts,
                    )
    
                # Encode target images
                pixel_values = target_images.to(dtype=vae.dtype)
                if args.vae_encode_mode == "sample":
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                else:
                    model_input = vae.encode(pixel_values).latent_dist.mode()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)
    
                # Process context images
                context_latents_list = []
                context_ids_list = []

                if len(context_images_list) > 0:
                    for i, ctx_tensor in enumerate(context_images_list[:args.max_context_images]):
                        ctx_tensor = ctx_tensor.unsqueeze(0).to(dtype=vae.dtype, device=accelerator.device)
                        
                        if args.vae_encode_mode == "sample":
                            ctx_latent = vae.encode(ctx_tensor).latent_dist.sample()
                        else:
                            ctx_latent = vae.encode(ctx_tensor).latent_dist.mode()
                        
                        ctx_latent = (ctx_latent - vae_config_shift_factor) * vae_config_scaling_factor
                        ctx_latent = ctx_latent.to(dtype=weight_dtype)
                        
                        # Pack latents
                        ctx_h, ctx_w = ctx_latent.shape[2:]
                        ctx_packed = FluxKontextPipeline._pack_latents(
                            ctx_latent, 1, ctx_latent.shape[1], ctx_h, ctx_w
                        )
                        
                        # Prepare IDs with proper tau
                        ctx_ids = FluxKontextPipeline._prepare_latent_image_ids(
                            1, ctx_h // 2, ctx_w // 2, accelerator.device, weight_dtype
                        )
                        ctx_ids[..., 0] = 1.0 + i * args.time_spacing
                        
                        context_latents_list.append(ctx_packed)
                        context_ids_list.append(ctx_ids)
    
                # Process asset images if enabled
                asset_latents_list = []
                asset_ids_list = []
                
                if args.enable_assets and len(asset_images_list) > 0:
                    for j, asset_tensor in enumerate(asset_images_list):
                        asset_tensor = asset_tensor.unsqueeze(0).to(dtype=vae.dtype, device=accelerator.device)
                        
                        if args.vae_encode_mode == "sample":
                            asset_latent = vae.encode(asset_tensor).latent_dist.sample()
                        else:
                            asset_latent = vae.encode(asset_tensor).latent_dist.mode()
                        
                        asset_latent = (asset_latent - vae_config_shift_factor) * vae_config_scaling_factor
                        asset_latent = asset_latent.to(dtype=weight_dtype)
                        
                        # Pack latents
                        asset_h, asset_w = asset_latent.shape[2:]
                        asset_packed = FluxKontextPipeline._pack_latents(
                            asset_latent, 1, asset_latent.shape[1], asset_h, asset_w
                        )
                        
                        # Prepare IDs with tau = 101+
                        asset_ids = FluxKontextPipeline._prepare_latent_image_ids(
                            1, asset_h // 2, asset_w // 2, accelerator.device, weight_dtype
                        )
                        asset_ids[..., 0] = 101.0 + j
                        
                        asset_latents_list.append(asset_packed)
                        asset_ids_list.append(asset_ids)
    
                # Sample noise and timesteps
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
    
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
    
                # Prepare latent IDs for target
                latent_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                
                # Combine all IDs
                id_parts = [latent_ids]
                if context_ids_list:
                    id_parts.extend(context_ids_list)
                if asset_ids_list:
                    id_parts.extend(asset_ids_list)
                combined_latent_ids = torch.cat(id_parts, dim=0)
                
                # Add noise
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                
                # Pack noisy input
                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                
                # Prepare guidance
                # NOTE: transformer is a PEFT wrapper now; get base model's config
                unwrapped = accelerator.unwrap_model(transformer)
                base_xf = getattr(unwrapped, "get_base_model", None)
                if callable(base_xf):
                    base_xf = unwrapped.get_base_model()
                else:
                    base_xf = unwrapped  # if not PEFT (edge case)

                if getattr(base_xf.config, "guidance_embeds", False):
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None
                
                # Combine all latents
                latent_parts = [packed_noisy_model_input]
                if context_latents_list:
                    latent_parts.extend(context_latents_list)
                if asset_latents_list:
                    latent_parts.extend(asset_latents_list)
                latent_model_input = torch.cat(latent_parts, dim=1)
                
                # Forward pass
                model_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=combined_latent_ids,
                    return_dict=False,
                )[0]
                
                # Extract only target prediction
                model_pred = model_pred[:, :packed_noisy_model_input.size(1)]
                
                # Unpack
                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                
                # Compute loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - model_input
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(transformer.parameters(), text_encoder_one.parameters())
                        if args.train_text_encoder
                        else transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
    
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
    
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
    
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
    
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
    
            if global_step >= args.max_train_steps:
                break

    # ---------------------------
    # Saving (simple & robust)
    # ---------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Unwrap DDP/PEFT wrappers for saving as needed
        xf_peft = accelerator.unwrap_model(transformer)
        te_peft = accelerator.unwrap_model(text_encoder_one) if args.train_text_encoder else None

        # (A) Optional: save PEFT adapters with configs (plug-and-play)
        if args.save_lora_dir:
            os.makedirs(args.save_lora_dir, exist_ok=True)
            xf_adapter_dir = os.path.join(args.save_lora_dir, "transformer")
            xf_peft.save_pretrained(xf_adapter_dir, safe_serialization=True)
            if te_peft is not None:
                te_adapter_dir = os.path.join(args.save_lora_dir, "text_encoder")
                te_peft.save_pretrained(te_adapter_dir, safe_serialization=True)
            logger.info(f"Saved PEFT adapters to {args.save_lora_dir}")

        # (B) Optional: merge LoRA into base and save single-file weights
        if args.merge_and_save_full:
            os.makedirs(args.merge_and_save_full, exist_ok=True)
            # Merge transformer LoRA
            if hasattr(xf_peft, "merge_and_unload"):
                xf_merged = xf_peft.merge_and_unload()
            else:
                # If somehow not a PEFT model, just use as-is
                xf_merged = xf_peft
            xf_merged_dir = os.path.join(args.merge_and_save_full, "transformer")
            xf_merged.save_pretrained(
                xf_merged_dir,
                safe_serialization=True,
                max_shard_size=args.max_shard_size,  # e.g., "100GB" -> one file
            )
            logger.info(f"Merged + saved transformer to {xf_merged_dir}")

            # Merge text encoder LoRA (if trained)
            if te_peft is not None:
                if hasattr(te_peft, "merge_and_unload"):
                    te_merged = te_peft.merge_and_unload()
                else:
                    te_merged = te_peft
                te_merged_dir = os.path.join(args.merge_and_save_full, "text_encoder")
                te_merged.save_pretrained(
                    te_merged_dir,
                    safe_serialization=True,
                    max_shard_size=args.max_shard_size,
                )
                logger.info(f"Merged + saved text encoder to {te_merged_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
