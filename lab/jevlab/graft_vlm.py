"""Graft a merged text-only checkpoint back into its vision-language parent.

  python -m jevlab.graft_vlm --parent XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B --text $BLINK_WORKDIR/ckpt/mimo-9b-text \
      --out $BLINK_WORKDIR/ckpt/blink-mimo-9b [--check-vlm]

The parent's vision tower, config and processor files are kept byte-for-byte. LoRA-targeted language-model
tensors come from the merged checkpoint (text key "model.X" or "model.language_model.X" -> parent key
"model.language_model.X"; lm_head as is). Checks: identical key sets and shapes; every tensor LoRA never touched (embeddings, norms, convolutions,
decay parameters, lm_head) equals the parent's and is written from the parent's own bytes; every LoRA-targeted
tensor differs from the parent's.
--check-vlm reloads the result as a vision-language model and requires no missing or unexpected keys.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
           "out_proj", "gate_proj", "up_proj", "down_proj")
KEEP_FILES = ("config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
              "video_preprocessor_config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")
SHARD_BYTES = 5 * 1024**3


def weight_map(d: str) -> dict[str, str]:
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as fh:
            return json.load(fh)["weight_map"]
    with safe_open(os.path.join(d, "model.safetensors"), "pt") as fh:
        return {k: "model.safetensors" for k in fh.keys()}


def load_all(d: str, keys) -> dict[str, torch.Tensor]:
    wm = weight_map(d)
    by_file: dict[str, list[str]] = {}
    for k in keys:
        by_file.setdefault(wm[k], []).append(k)
    out = {}
    for f, ks in by_file.items():
        with safe_open(os.path.join(d, f), "pt") as fh:
            for k in ks:
                out[k] = fh.get_tensor(k)
    return out


def text_to_parent(k: str) -> str:
    # transformers 5 saves a text-only model in its source layout, so both spellings occur
    if k == "lm_head.weight" or k.startswith("model.language_model."):
        return k
    if k.startswith("model."):
        return "model.language_model." + k[len("model."):]
    raise SystemExit(f"unexpected key in the text checkpoint: {k}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-vlm", action="store_true")
    ap.add_argument("--drop-prefix", default="mtp.", help="parent tensors the text checkpoint cannot carry (multi-token "
                    "prediction heads, stale after fine-tuning); dropped and reported")
    a = ap.parse_args()

    parent = a.parent if os.path.isdir(a.parent) else snapshot_download(a.parent)
    pw, tw = weight_map(parent), weight_map(a.text)
    dropped = sorted(k for k in pw if a.drop_prefix and k.startswith(a.drop_prefix))
    pw = {k: v for k, v in pw.items() if k not in set(dropped)}
    mapping = {k: text_to_parent(k) for k in tw}
    lm_keys = {k for k in pw if not k.startswith("model.visual.")}
    missing, extra = lm_keys - set(mapping.values()), set(mapping.values()) - lm_keys
    assert not missing and not extra, f"key sets differ: missing {sorted(missing)[:5]} extra {sorted(extra)[:5]}"

    text = load_all(a.text, list(tw))
    orig = load_all(parent, list(pw))
    n_target = n_changed = n_frozen = 0
    merged = dict(orig)  # vision tower and every untouched tensor: the parent's own bytes
    for tk, pk in mapping.items():
        t, p = text[tk], orig[pk]
        assert t.shape == p.shape, (tk, t.shape, p.shape)
        if any(f".{name}." in tk for name in TARGETS):
            n_target += 1
            n_changed += int(not torch.equal(t, p.to(t.dtype)))
            merged[pk] = t.to(p.dtype)
        else:
            n_frozen += 1
            assert torch.equal(t, p.to(t.dtype)), f"{tk} differs from the parent, but LoRA never touched it"
    assert n_changed == n_target, f"only {n_changed} of {n_target} LoRA-targeted tensors changed"
    os.makedirs(a.out, exist_ok=True)
    shards, cur, size = [], {}, 0
    for k in pw:  # the parent's tensor order
        nb = merged[k].numel() * merged[k].element_size()
        if cur and size + nb > SHARD_BYTES:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = merged[k].contiguous()
        size += nb
    if cur:
        shards.append(cur)
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in merged.values())}, "weight_map": {}}
    for i, sh in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, os.path.join(a.out, name), metadata={"format": "pt"})
        index["weight_map"].update({k: name for k in sh})
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    kept = [f for f in KEEP_FILES if os.path.exists(os.path.join(parent, f))]
    for f in kept:
        shutil.copy(os.path.join(parent, f), os.path.join(a.out, f))
    report = {"parent": a.parent, "text": a.text, "tensors": len(pw), "language_model": len(lm_keys),
              "vision": len(pw) - len(lm_keys), "lora_targeted_changed": n_changed, "untouched_bitwise_equal": n_frozen,
              "dropped": len(dropped), "shards": len(shards), "files_kept": kept}

    if a.check_vlm:
        from transformers import AutoModelForImageTextToText

        _, info = AutoModelForImageTextToText.from_pretrained(a.out, dtype=torch.bfloat16, device_map="cpu",
                                                              output_loading_info=True)
        bad = {k: v for k, v in info.items() if v and k in ("missing_keys", "unexpected_keys", "mismatched_keys")}
        assert not bad, f"vision-language reload: {bad}"
        report["vlm_reload"] = "no missing, unexpected or mismatched keys"
    with open(os.path.join(a.out + ".graft.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
