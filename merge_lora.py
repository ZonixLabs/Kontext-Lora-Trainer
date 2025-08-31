# merge_peft_flux.py
import torch
from peft import PeftModel
from diffusers import FluxTransformer2DModel

BASE_ID = "black-forest-labs/FLUX.1-Kontext-dev"     # pulls config+weights
LORA    = "/workspace/Kontext-Lora-Trainer/lora_out/transformer"
OUT     = "/workspace/out/merged/transformer"

xf = FluxTransformer2DModel.from_pretrained(BASE_ID, subfolder="transformer", torch_dtype=torch.bfloat16)
xf = PeftModel.from_pretrained(xf, LORA)
xf = xf.merge_and_unload()  # bake LoRA into base
xf.save_pretrained(OUT, safe_serialization=True, max_shard_size="100GB")
print("Merged transformer saved to:", OUT)

