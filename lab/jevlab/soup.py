"""Uniform weight-space average of merged checkpoints (model soup).
  python -m jevlab.soup OUT CKPT_DIR1 CKPT_DIR2 [...]
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

out, dirs = sys.argv[1], sys.argv[2:]
base = AutoModelForCausalLM.from_pretrained(dirs[0], dtype=torch.float32, device_map="cpu")
sd = {k: v.clone() for k, v in base.state_dict().items()}
for d in dirs[1:]:
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32, device_map="cpu")
    for k, v in m.state_dict().items():
        sd[k] += v
    del m
for k in sd:
    sd[k] /= len(dirs)
base.load_state_dict(sd)
base.to(torch.bfloat16).save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(dirs[0]).save_pretrained(out)
print("soup saved", out, "from", len(dirs))
