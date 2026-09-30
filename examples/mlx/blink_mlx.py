#!/usr/bin/env python3
"""Run blink on a Mac with MLX.

blink reads a decision; it does not write one. It renders each typed question as one fixed prompt, runs a
single forward pass (prefill only), and reads the next-token scores of the offered option letters. This script
keeps every part of that except the forward pass: prompts, option labels, request checks and answers come from the
model repo's own `blink.py` (pinned revision), and MLX runs the model. It reads text and JSON state; screenshots
are not supported here.

  pip install "mlx-lm>=0.31.3"
  python blink_mlx.py                                  # the model-card example on thegovind/blink-4b
  python blink_mlx.py --request request.json           # {"state": ..., "questions": {...}}
  python blink_mlx.py --model thegovind/blink-mimo-9b  # its text side only
  python blink_mlx.py --check                          # compare with the Space's saved runs
  python blink_mlx.py --adapter runs/mine              # with a LoRA adapter from train_mlx.py
  python blink_mlx.py --model ./my-model               # a local folder that holds blink.py

In Python, with this file next to your code:

  from blink_mlx import load
  blink = load("thegovind/blink-4b")
  out = blink.decide(state, questions)                 # the same {answers, meta} as blink.decide
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

REVISION = "v1.4"
MODELS = ("thegovind/blink-4b", "thegovind/blink-mimo-9b", "thegovind/blink-27b")
SPACE = "thegovind/blink"
SPACE_REVISION = "03ff6a7bae855d38d948f3f9366695c2b1b52179"  # the Space commit whose saved runs --check reads
FILES = ["*.safetensors", "*.json", "*.jinja", "*.txt", "blink.py"]

EXAMPLE_STATE = "Order #4411 arrived with a cracked screen. I want my money back, not another one."
EXAMPLE_QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {"refund": "Money back", "replacement": "A new unit", "info": "Information only"},
    },
    "urgent": {"type": "noul", "instructions": "Does this need a reply today?"},
    "anger": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]},
}


def _import_blink(path: Path, model: str):
    """The model repo's blink.py, imported without torch: rendering, labels, checks and answers."""
    os.environ["BLINK_MODEL"] = model
    spec = importlib.util.spec_from_file_location("blink", path / "blink.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["blink"] = module
    spec.loader.exec_module(module)
    return module


def model_path(model: str, revision: str = REVISION) -> Path:
    """A local model folder as is, or the Hub repo's files at `revision`."""
    local = Path(model).expanduser()
    if local.is_dir():
        if not (local / "blink.py").is_file():
            raise SystemExit(f"{local} has no blink.py; copy it from the model repo you fine-tuned")
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision, allow_patterns=FILES))


def load_text_model(path: Path, adapter: str | None = None):
    """MLX's model for a blink checkpoint (text side only), with an MLX LoRA adapter folder if given."""
    from mlx_lm import load as mlx_load

    config = json.loads((path / "config.json").read_text())
    override = {"model_type": "qwen3_5"} if config.get("model_type") == "qwen3_5_text" else None
    model, _ = mlx_load(str(path), model_config=override, adapter_path=adapter)
    return model


def readout_head(model):
    """The output rows the offered letters are read from: the tied embeddings, or lm_head."""
    text = model.language_model
    return text.model.embed_tokens if text.args.tie_word_embeddings else text.lm_head


class MlxEngine:
    """blink.py's TorchEngine text path with MLX doing the forward pass."""

    name = "mlx"
    accepts_images = False

    def __init__(self, blink, path: Path, model_id: str, temperature: float, adapter: str | None = None):
        import mlx.core as mx
        from transformers import AutoTokenizer

        self.blink = blink
        self.model_id = model_id
        self.temperature = float(temperature)
        self.tok = AutoTokenizer.from_pretrained(str(path))
        model = load_text_model(path, adapter)
        self.backbone = model.language_model.model
        head = readout_head(model)
        if not hasattr(head, "weight") or getattr(head, "scales", None) is not None:
            raise blink.BlinkError("use the published bf16 weights; quantized checkpoints are not supported")
        self.head = head.weight
        self.mx = mx
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.labels, self.label_ids = blink.TorchEngine._verify_labels(self)
        self.last_model_ms = None

    def _wrap(self, user: str) -> str:
        return self.blink.TorchEngine._wrap(self, user)

    def render(self, state, questions: dict):
        return self.blink.TorchEngine.render(self, state, questions)

    def logits(self, state, questions: dict, bias: dict | None = None):
        """One prefill per question; the final hidden state times the offered letters' output rows, in FP32."""
        del bias
        mx = self.mx
        work = self.render(state, questions)
        rows = {}
        started = time.perf_counter()
        for item in work:
            hidden = self.backbone(mx.array([item["ids"]]))[0, -1].astype(mx.float32)
            rows[item["qkey"]] = (self.head[mx.array(item["cand"])].astype(mx.float32) @ hidden).tolist()
        self.last_model_ms = round((time.perf_counter() - started) * 1000, 1)
        return rows, sum(len(item["ids"]) for item in work)


class Blink:
    def __init__(self, model: str | None = None, revision: str = REVISION, temperature: float | None = None,
                 adapter: str | None = None):
        model = adapter_base(adapter, model) if adapter else (model or MODELS[0])
        path = model_path(model, revision)
        self.module = _import_blink(path, model)
        self.engine = MlxEngine(self.module, path, model,
                                self.module.TEMPERATURE if temperature is None else temperature, adapter)
        self.module._ENGINE = self.engine

    def decide(self, state, questions: dict, temperature: float | None = None) -> dict:
        out = self.module.decide(state, questions, temperature=temperature)
        out["meta"]["model_ms"] = self.engine.last_model_ms
        return out


def adapter_base(adapter: str, model: str | None) -> str:
    """The model an adapter from train_mlx.py was trained on; refuses a different explicit --model."""
    try:
        base = json.loads((Path(adapter) / "adapter_config.json").read_text()).get("base_model")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{adapter}: not an adapter folder from train_mlx.py ({exc})") from None
    if model and base and model != base:
        raise SystemExit(f"{adapter} was trained on {base}, not {model}; pass --model {base} or leave --model out")
    return model or base or MODELS[0]


def load(model: str | None = None, revision: str = REVISION, temperature: float | None = None,
         adapter: str | None = None) -> Blink:
    return Blink(model, revision, temperature, adapter)


def _softmax(xs):
    top = max(xs)
    exps = [math.exp(x - top) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def check(blink: Blink, reference: str | None = None) -> dict:
    """Every bundled Space example this model has a saved run for: same token count, same top answer, and
    the largest probability gap. `reference` is your own PyTorch recording in the Space's replay format
    (space/record_replay.py); without it, the Space's saved runs are used (blink-4b and blink-mimo-9b)."""
    from huggingface_hub import hf_hub_download

    name = blink.engine.model_id.split("/")[-1]
    replay = "replay.json" if name == "blink-4b" else f"replay-{name}.json"
    try:
        saved = json.loads(Path(reference).read_text() if reference else
                           Path(hf_hub_download(SPACE, replay, repo_type="space",
                                                revision=SPACE_REVISION)).read_text())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"no saved runs for {blink.engine.model_id} ({exc}); the Space has them for "
                         "blink-4b and blink-mimo-9b, and --reference takes your own")
    if saved.get("model") != blink.engine.model_id:
        raise SystemExit(f"the saved runs are {saved.get('model')!r}'s, not {blink.engine.model_id!r}'s")
    sys.path.insert(0, str(Path(hf_hub_download(SPACE, "examples.py", repo_type="space",
                                                revision=SPACE_REVISION)).parent))
    import examples

    requests = [(f"{case.key}/{ex.label}", ex.state, case.questions)
                for case in examples.USE_CASES for ex in case.examples]
    requests += [(f"playground/{label}", blink.module.as_state(text), json.loads(qs))
                 for label, text, qs in examples.PLAYGROUND_PRESETS]
    report = {"model": blink.engine.model_id, "requests": 0, "questions": 0, "top_agree": 0,
              "token_mismatches": 0, "max_abs_dp": 0.0, "flips": []}
    gaps, seen = [], set()
    for name_, state, questions in requests:
        key = blink.module.request_key(state, questions)
        hit = saved["requests"].get(key)
        if hit is None or key in seen:  # two Playground presets repeat use-case examples
            continue
        seen.add(key)
        raw, tokens = blink.engine.logits(state, questions)
        report["requests"] += 1
        report["token_mismatches"] += int(tokens != int(hit["input_tokens"]))
        for qkey, ours in raw.items():
            p, q = _softmax(ours), _softmax(hit["logits"][qkey])
            report["questions"] += 1
            gap = max(abs(a - b) for a, b in zip(p, q))
            gaps.append(gap)
            if p.index(max(p)) == q.index(max(q)):
                report["top_agree"] += 1
            else:
                ranked = sorted(q, reverse=True)
                report["flips"].append({"request": name_, "question": qkey,
                                        "saved_top_two_margin": round(ranked[0] - ranked[1], 4)})
    gaps.sort()
    report["max_abs_dp"] = round(gaps[-1], 4) if gaps else None
    report["median_max_abs_dp"] = round(gaps[len(gaps) // 2], 4) if gaps else None
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help=f"one of {', '.join(MODELS)}, or a local model folder (default: {MODELS[0]}, "
                                    "or the base model an --adapter was trained on)")
    ap.add_argument("--revision", default=REVISION, help="model repo revision (default: %(default)s)")
    ap.add_argument("--adapter", help="an MLX LoRA adapter folder from train_mlx.py")
    ap.add_argument("--temperature", type=float,
                    help="divide the letter scores by T before the softmax (fit T with lab/jevlab/fit_temp.py)")
    ap.add_argument("--request", help='JSON file with {"state": ..., "questions": {...}}')
    ap.add_argument("--check", action="store_true", help="compare with the Space's saved runs")
    ap.add_argument("--reference", help="with --check: your own recording (space/record_replay.py format)")
    a = ap.parse_args(argv)
    if a.reference and not a.check:
        ap.error("--reference needs --check")
    if a.temperature is not None and not a.temperature > 0:
        ap.error("--temperature must be positive")
    blink = load(a.model, a.revision, temperature=a.temperature, adapter=a.adapter)
    if a.check:
        print(json.dumps(check(blink, a.reference), indent=1))
        return
    if a.request:
        request = json.loads(Path(a.request).read_text())
        state, questions = request["state"], request["questions"]
    else:
        state, questions = EXAMPLE_STATE, EXAMPLE_QUESTIONS
    print(json.dumps(blink.decide(state, questions), indent=1))


if __name__ == "__main__":
    main()
