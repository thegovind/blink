"""LoRA post-training of the next-token label readout (torchrun, one replica per process).

Rows (JSONL): {"id", "state", "question": typed q, "gold": key | None, "target": {key: p} | None,
               "weight": float (default 1), "src": str}
Loss per example: cross-entropy of the restricted label distribution against the target
(soft target when given, else one-hot gold). Option order and label assignment are re-drawn every
epoch (score levels keep their ordinal order; noul yes/no order is randomised).

  torchrun --nproc_per_node 8 -m jevlab.train --model Qwen/Qwen3.5-4B --data a.jsonl:1.0,b.jsonl:0.5 --out OUT
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .render import Renderer, question_options

TARGETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "delta": ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}


def load_rows(spec, rng):
    rows = []
    for part in spec.split(","):
        path, _, frac = part.partition(":")
        frac = float(frac or 1.0)
        rs = [json.loads(l) for l in open(path) if l.strip()]
        if frac < 1.0:
            rng.shuffle(rs)
            rs = rs[: int(len(rs) * frac)]
        elif frac > 1.0:
            rs = rs * int(frac) + rs[: int(len(rs) * (frac - int(frac)))]
        rows += rs
    return rows


def encode(rd, row, rng, max_len):
    q = row["question"]
    items = question_options(q)
    k = len(items)
    if q["type"] == "score":
        order = list(range(k))
    else:
        order = list(range(k))
        rng.shuffle(order)
    prompt, keys, cand = rd.render(row.get("state"), q, order=order)
    ids = rd.tok(prompt, add_special_tokens=False)["input_ids"]
    target = row.get("target")
    if target is not None:
        if not isinstance(target, dict) or set(target) - set(keys):
            raise ValueError(f"row {row.get('id', '?')}: target must use offered option keys")
        try:
            t = [float(target.get(key, 0.0)) for key in keys]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"row {row.get('id', '?')}: invalid target value") from exc
    else:
        gold = row.get("gold")
        if gold not in keys:
            raise ValueError(f"row {row.get('id', '?')}: gold must be an offered option key")
        t = [1.0 if key == gold else 0.0 for key in keys]
    s = sum(t)
    if any(not math.isfinite(x) or x < 0 for x in t) or not math.isfinite(s) or s <= 0:
        raise ValueError(f"row {row.get('id', '?')}: target must be finite, nonnegative and nonempty")
    try:
        weight = float(row.get("weight", 1.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"row {row.get('id', '?')}: invalid weight") from exc
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"row {row.get('id', '?')}: weight must be finite and nonnegative")
    if len(ids) > max_len:
        return None
    return {"ids": ids, "cand": cand, "t": [x / s for x in t], "w": weight, "src": row.get("src", "")}


def make_batches(exs, budget, rng):
    idx = list(range(len(exs)))
    rng.shuffle(idx)
    batches = []
    mega = 4096
    for m0 in range(0, len(idx), mega):
        chunk = sorted(idx[m0: m0 + mega], key=lambda i: len(exs[i]["ids"]))
        b, mx = [], 0
        for i in chunk:
            L = len(exs[i]["ids"])
            if b and max(mx, L) * (len(b) + 1) > budget:
                batches.append(b)
                b, mx = [], 0
            b.append(i)
            mx = max(mx, L)
        if b:
            batches.append(b)
    rng.shuffle(batches)
    return batches


def forward_batch(body, lm_head, exs, pad_id, device):
    L = max(len(e["ids"]) for e in exs)
    ids = torch.full((len(exs), L), pad_id, dtype=torch.long)
    for i, e in enumerate(exs):
        ids[i, : len(e["ids"])] = torch.tensor(e["ids"])
    ids = ids.to(device)
    h = body(input_ids=ids, use_cache=False).last_hidden_state
    last = torch.tensor([len(e["ids"]) - 1 for e in exs], device=device)
    losses, logps = [], []
    with torch.autocast(device.type, enabled=False):  # FP32 projection + normalisation, as at serving time
        h = h[torch.arange(len(exs), device=device), last].float()
        for i, e in enumerate(exs):
            w = lm_head[torch.tensor(e["cand"], device=device)].float()
            lp = torch.log_softmax(w @ h[i], dim=-1)
            t = torch.tensor(e["t"], device=device)
            losses.append(-(t * lp).sum())
            logps.append(lp.detach())
    return torch.stack(losses), logps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--template", default="semif")
    ap.add_argument("--targets", default="attn,delta,mlp")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--budget", type=int, default=16384, help="padded tokens per process per micro-batch")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--init-adapter", default=None)
    a = ap.parse_args(argv)

    dist.init_process_group("nccl" if a.device == "cuda" else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    if a.device == "cuda":
        torch.cuda.set_device(local)
    device = torch.device("cuda", local) if a.device == "cuda" else torch.device("cpu")
    torch.manual_seed(a.seed)
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if a.device == "cuda" else torch.float32,
                                                 device_map={"": device},
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    tmods = sum((TARGETS[t] for t in a.targets.split(",")), [])
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                                 target_modules=tmods, bias="none"))
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    base = model.get_base_model()
    body, lm_head = base.model, base.lm_head.weight
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rd = Renderer(tok, template=a.template)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    rng = random.Random(a.seed)
    rows = load_rows(a.data, rng)
    dev_rows = [json.loads(l) for l in open(a.dev) if l.strip()] if a.dev else []
    if a.dev and not dev_rows:
        raise ValueError("dev file has no rows")
    n_epochs = math.ceil(a.epochs)
    # pre-count steps from epoch-0 encoding
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=a.wd)
    step, t0 = 0, time.time()
    log = open(out / f"log.rank{rank}.jsonl", "a") if rank == 0 else None

    def evaluate(tag):
        if not dev_rows:
            return
        model.eval()
        erng = random.Random(99)
        exs = [e for e in (encode(rd, r, erng, a.max_len) for r in dev_rows[rank::world]) if e]
        tot, n, correct = 0.0, 0, 0
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=a.device == "cuda"):
            for b in make_batches(exs, a.budget, random.Random(0)):
                ls, lps = forward_batch(body, lm_head, [exs[i] for i in b], pad_id, device)
                tot += float(ls.sum()); n += len(b)
                for i, lp in zip(b, lps):
                    correct += int(int(lp.argmax()) == max(range(len(exs[i]["t"])), key=lambda j: exs[i]["t"][j]))
        v = torch.tensor([tot, n, correct], device=device, dtype=torch.float64)
        dist.all_reduce(v)
        if v[1] == 0:
            raise ValueError("no dev rows survived encoding; check --max-len")
        if rank == 0:
            rec = {"eval": tag, "step": step, "dev_nll": float(v[0] / v[1]), "dev_acc": float(v[2] / v[1]), "n": int(v[1])}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
        model.train()

    def save(tag):
        if rank == 0:
            model.save_pretrained(out / tag)
            print(f"saved {tag}", flush=True)
        dist.barrier()

    model.train()
    evaluate("init")
    total_steps = None
    for ep in range(n_epochs):
        erng = random.Random(a.seed * 1000 + ep)
        exs = [e for e in (encode(rd, r, erng, a.max_len) for r in rows) if e]
        if rank == 0:
            print(f"epoch {ep}: encoded {len(exs)}/{len(rows)} rows (dropped {len(rows) - len(exs)} over max_len {a.max_len})", flush=True)
        batches = make_batches(exs, a.budget, erng)
        if ep == n_epochs - 1 and a.epochs < n_epochs:
            batches = batches[: int(len(batches) * (a.epochs - ep))]
        nb = (len(batches) // (world * a.accum)) * world * a.accum
        if nb == 0:
            raise ValueError(f"epoch {ep}: no optimizer steps; check data, --max-len, --epochs, --accum and process count")
        mine = batches[:nb][rank::world]
        if total_steps is None:
            total_steps = int(n_epochs * len(mine) / a.accum)
            if rank == 0:
                print(f"examples {len(exs)} batches/rank {len(mine)} steps~{total_steps} trainable {ntrain / 1e6:.1f}M", flush=True)
        for bi in range(0, len(mine), a.accum):
            group = mine[bi: bi + a.accum]
            nex = torch.tensor([sum(len(b) for b in group)], device=device, dtype=torch.float32)
            dist.all_reduce(nex)
            lr = a.lr * min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total_steps))))
            for g in opt.param_groups:
                g["lr"] = lr
            tl, tw = 0.0, 0
            for j, b in enumerate(group):
                exb = [exs[i] for i in b]
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=a.device == "cuda"):
                    ls, _ = forward_batch(body, lm_head, exb, pad_id, device)
                w = torch.tensor([e["w"] for e in exb], device=device)
                loss = (ls * w).sum() / nex[0] * world
                loss.backward()
                tl += float(ls.detach().sum()); tw += len(b)
            # manual all-reduce of LoRA grads (model is not DDP-wrapped: simpler with peft + checkpointing)
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            flat = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flat)
            flat /= world
            for p, g in zip(params, torch._utils._unflatten_dense_tensors(flat, grads)):
                p.grad = g
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and (step % 10 == 0 or step == 1):
                rec = {"step": step, "ep": ep, "loss": tl / max(tw, 1), "lr": lr, "gnorm": float(gn),
                       "elapsed": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
            if a.eval_every and step % a.eval_every == 0:
                evaluate(f"step{step}")
            if a.save_every and step % a.save_every == 0:
                save(f"step{step}")
    evaluate("final")
    save("final")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
