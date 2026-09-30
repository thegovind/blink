#!/usr/bin/env python3
"""Fine-tune blink on a Mac with MLX: a LoRA adapter trained with the same objective as lab/jevlab/train.py.

  pip install "mlx-lm>=0.31.3"
  python examples/train/prepare.py my_rows.jsonl --out-dir data --dev-fraction 0.15
  python examples/mlx/train_mlx.py --model thegovind/blink-4b --data data/train.jsonl --dev data/dev.jsonl \\
      --out runs/mine
  python examples/mlx/blink_mlx.py --model thegovind/blink-4b --adapter runs/mine --request request.json

Rows are the trainer's JSONL (examples/train/README.md). Each question is rendered with blink's prompt; choice
and yes/no options get a new order and new letters every epoch (score levels keep their order). The loss is
cross-entropy over the offered letters only, against the row's target distribution (or its one-hot gold), times
the row's weight, averaged over the rows of a step. LoRA (rank 16, alpha 32) goes on the attention, Gated
DeltaNet and MLP projections; embeddings, norms and the letter rows stay frozen. Optimizer and schedule follow
jevlab/train.py: AdamW (betas 0.9/0.99, bias-corrected), linear warmup then cosine decay, gradients clipped to
norm 1. The output folder (adapters.safetensors, adapter_config.json) is an mlx-lm adapter that
blink_mlx.py --adapter loads. Run it from a clone of the repository: it renders with lab/jevlab/render.py.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "lab"))
from blink_mlx import MODELS, REVISION, adapter_base, load_text_model, model_path, readout_head  # noqa: E402
from jevlab.render import Renderer, question_options  # noqa: E402

TARGETS = {
    "attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    "delta": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
              "linear_attn.in_proj_b", "linear_attn.out_proj"],
    "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
}


def load_rows(spec: str, rng: random.Random) -> list[dict]:
    """`a.jsonl,b.jsonl:0.5` - each file, optionally subsampled (<1) or repeated (>1), as in jevlab/train.py."""
    rows = []
    for part in spec.split(","):
        path, _, frac = part.partition(":")
        frac = float(frac or 1.0)
        with open(path, encoding="utf-8") as fh:
            rs = [json.loads(line) for line in fh if line.strip()]
        if frac < 1.0:
            rng.shuffle(rs)
            rs = rs[: int(len(rs) * frac)]
        elif frac > 1.0:
            rs = rs * int(frac) + rs[: int(len(rs) * (frac - int(frac)))]
        rows += rs
    return rows


def encode(rd: Renderer, row: dict, rng: random.Random, max_len: int) -> dict | None:
    """One row: prompt ids with a fresh option order, the offered letters' token ids, and the target over them."""
    q = row["question"]
    order = list(range(len(question_options(q))))
    if q["type"] != "score":
        rng.shuffle(order)
    prompt, keys, cand = rd.render(row.get("state"), q, order=order)
    ids = rd.tok(prompt, add_special_tokens=False)["input_ids"]
    where = f"row {row.get('id', '?')}"
    if row.get("target") is not None:
        target = row["target"]
        if not isinstance(target, dict) or set(target) - set(keys):
            raise ValueError(f"{where}: target must use offered option keys")
        t = [float(target.get(k, 0.0)) for k in keys]
    else:
        if row.get("gold") not in keys:
            raise ValueError(f"{where}: gold must be an offered option key")
        t = [1.0 if k == row["gold"] else 0.0 for k in keys]
    total = sum(t)
    if any(not math.isfinite(x) or x < 0 for x in t) or not math.isfinite(total) or total <= 0:
        raise ValueError(f"{where}: target must be finite, nonnegative and nonempty")
    weight = float(row.get("weight", 1.0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"{where}: weight must be finite and nonnegative")
    if len(ids) > max_len:
        return None
    return {"ids": ids, "cand": cand, "t": [x / total for x in t], "w": weight}


def metrics(records: list[dict]) -> dict:
    """dev_run.py's summary: accuracy, log loss, Brier and top-label ECE (10 bins), overall and per source."""
    def ece(pairs):
        bins = [[0, 0.0, 0] for _ in range(10)]
        for c, ok in pairs:
            b = bins[min(int(c * 10), 9)]
            b[0] += 1
            b[1] += c
            b[2] += ok
        n = sum(b[0] for b in bins)
        return sum(b[0] / n * abs(b[2] / b[0] - b[1] / b[0]) for b in bins if b[0]) if n else None

    by_src = {}
    for src in sorted({r["src"] for r in records}):
        rs = [r for r in records if r["src"] == src]
        by_src[src] = {"n": len(rs), "acc": sum(r["pred"] == r["gold"] for r in rs) / len(rs),
                       "nll": sum(r["nll"] for r in rs) / len(rs), "brier": sum(r["brier"] for r in rs) / len(rs),
                       "ece": ece([(max(r["p"].values()), r["pred"] == r["gold"]) for r in rs])}
    return {"macro_acc": sum(v["acc"] for v in by_src.values()) / len(by_src),
            "overall_ece": ece([(max(r["p"].values()), r["pred"] == r["gold"]) for r in records]),
            "by_src": by_src}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="a blink model repo or a local model folder (default: thegovind/blink-4b, or "
                                    "the base model --init-adapter was trained on)")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--data", help="train JSONL; several files as a.jsonl,b.jsonl:0.5")
    ap.add_argument("--dev", help="development JSONL, read before and after training")
    ap.add_argument("--out", required=True, help="adapter folder to write (must not exist)")
    ap.add_argument("--targets", default="attn,delta,mlp")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8, help="rows per optimizer step")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=4096, help="longer prompts are skipped, never cut")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--init-adapter", help="continue from an adapter this script wrote for the same model")
    ap.add_argument("--no-grad-checkpoint", action="store_true", help="faster, but uses more memory")
    ap.add_argument("--eval-only", action="store_true",
                    help="read --dev with the model (and --init-adapter, if given), write eval-init.json and stop")
    a = ap.parse_args(argv)
    if a.eval_only and not a.dev:
        ap.error("--eval-only needs --dev")
    if not a.eval_only and not a.data:
        ap.error("--data is required unless --eval-only")

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from transformers import AutoTokenizer

    a.model = adapter_base(a.init_adapter, a.model) if a.init_adapter else (a.model or MODELS[0])
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new --out")
    mx.random.seed(a.seed)
    path = model_path(a.model, a.revision)
    tok = AutoTokenizer.from_pretrained(str(path))
    rd = Renderer(tok, template="semif")
    model = load_text_model(path)
    model.freeze()
    keys = sorted({k for t in a.targets.split(",") for k in TARGETS[t]})
    lora = {"rank": a.rank, "scale": a.alpha / a.rank, "dropout": 0.0, "keys": keys}
    linear_to_lora_layers(model, len(model.layers), lora)
    if a.init_adapter:
        model.load_weights(str(Path(a.init_adapter) / "adapters.safetensors"), strict=False)
    if not a.no_grad_checkpoint:
        grad_checkpoint(model.layers[0])
    head = readout_head(model)
    ntrain = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    def loss_fn(m, ids, cand, target, weight):
        h = m.language_model.model(ids[None])[0, -1].astype(mx.float32)
        logits = head.weight[cand].astype(mx.float32) @ h
        return -(target * (logits - mx.logsumexp(logits))).sum() * weight

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    rng = random.Random(a.seed)
    rows = load_rows(a.data, rng) if a.data else []
    dev_rows = []
    if a.dev:
        with open(a.dev, encoding="utf-8") as fh:
            dev_rows = [json.loads(line) for line in fh if line.strip()]
    out.mkdir(parents=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    log = open(out / "log.jsonl", "w", encoding="utf-8")  # noqa: SIM115

    def record(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    def evaluate(tag, step):
        """Every dev row in its given option order (as served); writes eval-<tag>.json in dev_run.py's format."""
        if not dev_rows:
            return
        model.eval()
        records = []
        for r in dev_rows:
            q = r["question"]
            prompt, keys, cand = rd.render(r.get("state"), q)
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > a.max_len:
                continue
            target = r.get("target") or {r.get("gold"): 1.0}
            t = [float(target.get(k, 0.0)) for k in keys]
            if not sum(t) > 0:
                raise SystemExit(f"dev row {r.get('id', '?')}: needs a gold or target over the offered options "
                                 "(examples/train/prepare.py writes them)")
            h = model.language_model.model(mx.array([ids]))[0, -1].astype(mx.float32)
            logits = (head.weight[mx.array(cand)].astype(mx.float32) @ h).tolist()
            top = max(logits)
            z = sum(math.exp(x - top) for x in logits)
            p = [math.exp(x - top) / z for x in logits]
            records.append({"id": r.get("id"), "src": r.get("src", "?"), "pred": keys[p.index(max(p))],
                            "gold": keys[t.index(max(t))], "p": dict(zip(keys, p)),
                            "nll": -sum(ti * math.log(max(pi, 1e-12)) for ti, pi in zip(t, p)) / sum(t),
                            "brier": sum((pi - ti / sum(t)) ** 2 for pi, ti in zip(p, t))})
        model.train()
        if not records:
            raise SystemExit("no dev rows fit --max-len")
        summary = metrics(records)
        (out / f"eval-{tag}.json").write_text(json.dumps(
            {"summary": {"model": a.model, "adapter": str(out) if tag != "init" else a.init_adapter,
                         "temperature": 1.0, **summary},
             "records": [{k: r[k] for k in ("id", "src", "pred", "gold", "p")} for r in records]}))
        record({"eval": tag, "step": step, "dev_nll": sum(r["nll"] for r in records) / len(records),
                "dev_acc": sum(r["pred"] == r["gold"] for r in records) / len(records),
                "macro_acc": summary["macro_acc"], "ece": summary["overall_ece"], "n": len(records)})

    opt = optim.AdamW(learning_rate=a.lr, betas=[0.9, 0.99], eps=1e-8, weight_decay=a.wd, bias_correction=True)
    model.train()
    evaluate("init", 0)
    if a.eval_only:
        log.close()
        print(f"wrote {out / 'eval-init.json'}", flush=True)
        return 0
    step, total_steps, t0 = 0, None, time.time()
    for ep in range(math.ceil(a.epochs)):
        erng = random.Random(a.seed * 1000 + ep)
        exs = [e for e in (encode(rd, r, erng, a.max_len) for r in rows) if e]
        print(f"epoch {ep}: encoded {len(exs)}/{len(rows)} rows (skipped {len(rows) - len(exs)} over --max-len "
              f"{a.max_len})", flush=True)
        order = list(range(len(exs)))
        erng.shuffle(order)
        if ep == math.ceil(a.epochs) - 1 and a.epochs < math.ceil(a.epochs):
            order = order[: int(len(order) * (a.epochs - ep))]
        batches = [order[i: i + a.batch] for i in range(0, len(order), a.batch)]
        if not batches:
            raise SystemExit(f"epoch {ep}: no rows to train on; check --data, --max-len and --epochs")
        if total_steps is None:
            total_steps = math.ceil(a.epochs * len(exs) / a.batch)
            print(f"rows {len(exs)} steps~{total_steps} trainable {ntrain / 1e6:.2f}M", flush=True)
        for batch in batches:
            lr = a.lr * min(1.0, (step + 1) / a.warmup) * 0.5 * (
                1 + math.cos(math.pi * min(1.0, step / max(1, total_steps))))
            opt.learning_rate = lr
            summed, total_loss = None, 0.0
            for i in batch:
                e = exs[i]
                loss, grads = loss_and_grad(model, mx.array(e["ids"]), mx.array(e["cand"]), mx.array(e["t"]),
                                            mx.array(e["w"]))
                summed = grads if summed is None else tree_map(mx.add, summed, grads)
                mx.eval(summed, loss)
                total_loss += loss.item()
            summed = tree_map(lambda g, n=len(batch): g / n, summed)
            summed, norm = optim.clip_grad_norm(summed, 1.0)
            opt.update(model, summed)
            mx.eval(model.trainable_parameters(), opt.state)
            step += 1
            if step == 1 or step % 10 == 0:
                record({"step": step, "ep": ep, "loss": total_loss / len(batch), "lr": lr, "gnorm": norm.item(),
                        "elapsed": round(time.time() - t0, 1)})
    evaluate("final", step)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (out / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "num_layers": len(model.layers), "lora_parameters": lora,
         "base_model": a.model, "revision": a.revision, "trainer": "examples/mlx/train_mlx.py"}, indent=1))
    log.close()
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
