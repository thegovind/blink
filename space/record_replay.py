"""Record replay.json: the trained model's label logits for every bundled request.

Runs on a GPU host with the exact weights that are published:
    python record_replay.py --model /path/to/merged --out replay.json
Each entry is what TorchEngine returns for that request, plus the median end-to-end
latency of blink.decide over repeated runs on this GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import blink  # noqa: E402
import examples  # noqa: E402


def bundled_requests():
    """Every (state, questions) the interface can send without the user editing anything."""
    reqs = []
    for case in examples.USE_CASES:
        for ex in case.examples:
            reqs.append((f"{case.key}/{ex.label}", ex.state, case.questions))
    for label, state_text, questions_text in examples.PLAYGROUND_PRESETS:
        reqs.append((f"playground/{label}", blink.as_state(state_text), json.loads(questions_text)))
    return reqs


def weights_digest(path: str) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(path).glob("*.safetensors")):
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()


def main():
    # main() builds a TorchEngine directly; this is only for anything it calls later.
    os.environ["BLINK_ENGINE"] = "torch"
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local path of the published weights")
    ap.add_argument("--model-id", default=blink.MODEL_ID)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", default="replay.json")
    a = ap.parse_args()

    import torch

    eng = blink.TorchEngine(a.model, None, blink.TEMPERATURE)
    eng.model_id = a.model_id
    blink._ENGINE = eng
    reqs = bundled_requests()
    for _, state, qs in reqs[:3]:
        blink.decide(state, qs)  # warm-up
    cache = {}
    for name, state, qs in reqs:
        key = blink.request_key(state, qs)
        if key in cache:
            cache[key]["names"].append(name)
            continue
        raw, n_tokens = eng.logits(state, qs)
        lat = []
        for _ in range(a.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            blink.decide(state, qs)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
        cache[key] = {
            "names": [name],
            "logits": {k: [round(float(x), 5) for x in v] for k, v in raw.items()},
            "input_tokens": int(n_tokens),
            "latency_ms": round(statistics.median(lat), 1),
        }
        top = {k: blink.question_options(q)[max(range(len(raw[k])), key=raw[k].__getitem__)][0] for k, q in qs.items()}
        print(f"{name:40s} {n_tokens:6d} tok {cache[key]['latency_ms']:7.1f} ms  {top}")
    gpu = torch.cuda.get_device_name(0)
    out = {
        "model": a.model_id,
        "weights_sha256": weights_digest(a.model),
        "engine": "TorchEngine (bf16 weights, FP32 label projection)",
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requests": cache,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}: {len(cache)} requests")


if __name__ == "__main__":
    main()
