"""Merge a LoRA adapter into its base and save a standalone checkpoint (for vLLM scoring/serving).
  python -m jevlab.merge --base Qwen/Qwen3.5-4B --adapter RUN/final --out $BLINK_WORKDIR/ckpt/NAME
"""
import argparse
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.base)
m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
m.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
tok.save_pretrained(a.out)
print("saved", a.out)
