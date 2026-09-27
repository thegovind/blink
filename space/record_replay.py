"""Record replay.json: the trained model's label logits for every bundled request.

Runs on a GPU host with the exact weights that are published:
    python record_replay.py --model /path/to/merged --out replay.json
Each entry is what TorchEngine returns for that request, plus the median end-to-end
latency of blink.decide over repeated runs on this GPU.

The Screen click presets are screenshots, so they are recorded only by a model that reads
images with its own vision tower (blink-mimo-9b), and added to that model's recording:
    python record_replay.py --model /path/to/blink-mimo-9b --model-id thegovind/blink-mimo-9b \\
        --vision --screens-only --out replay-blink-mimo-9b.json
--screens-only keeps every text entry already in --out and refuses a file recorded from
other weights; without it, --vision records the text requests and the screenshots together.
The file records the SHA-256 of the blink.py that made it (screens_blink_py_sha256 for a
--screens-only merge), and each screenshot entry its median GPU time per Decide (model_ms).
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


def screen_requests():
    """The Screen click presets, exactly as the page sends them: a marked screenshot and three questions."""
    import screens

    return [(f"{screens.KEY}/{shot.key}", *screens.request(shot)) for shot in screens.presets()]


def weights_digest(path: str) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(path).glob("*.safetensors")):
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()


def _no_sync():
    return None


def file_digest(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def record(eng, reqs, repeats: int, sync=_no_sync) -> dict:
    """Each request's label logits from `eng`, and the median latency of blink.decide end to end.

    A screenshot request goes through the Space's own path, screens.forward: every question in
    one GPU call. Its entry also keeps the processed pixels and visual tokens the page shows
    beside a saved run, and the median GPU time per Decide (model_ms)."""
    import screens

    cache = {}
    for name, state, qs in reqs:
        key = blink.request_key(state, qs)
        if key in cache:
            cache[key]["names"].append(name)
            continue
        image = blink.contains_image_uri(state)
        if image:
            raw, n_tokens, facts = screens.forward(eng, state, qs)
        else:
            raw, n_tokens = eng.logits(state, qs)
        lat, gpu = [], []
        for _ in range(repeats):
            sync()
            t0 = time.perf_counter()
            if image:
                gpu.append(screens.forward(eng, state, qs)[2]["model_ms"])
            else:
                blink.decide(state, qs)
            sync()
            lat.append((time.perf_counter() - t0) * 1000)
        cache[key] = {
            "names": [name],
            "logits": {k: [round(float(x), 5) for x in v] for k, v in raw.items()},
            "input_tokens": int(n_tokens),
            "latency_ms": round(statistics.median(lat), 1),
        }
        if image:
            cache[key].update(image_pixels=facts["image_pixels"], visual_tokens=facts["visual_tokens"],
                              model_ms=round(statistics.median(gpu), 1), gpu_calls=1)
        top = {k: blink.question_options(q)[max(range(len(raw[k])), key=raw[k].__getitem__)][0] for k, q in qs.items()}
        print(f"{name:40s} {n_tokens:6d} tok {cache[key]['latency_ms']:7.1f} ms  {top}")
    return cache


def merge(existing: dict, cache: dict, model_id: str, digest: str, code: str = "") -> dict:
    """An existing recording with fresh entries added (same key: replaced). Only the same weights may mix.
    `code` is the SHA-256 of the blink.py that made the new entries."""
    if existing.get("model") != model_id:
        raise SystemExit(f"the recording is {existing.get('model')!r}'s, not {model_id!r}'s")
    if existing.get("weights_sha256") != digest:
        raise SystemExit("the recording was made from other weights; record everything again instead")
    out = dict(existing)
    out["requests"] = {**existing.get("requests", {}), **cache}
    out["screens_recorded_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if code:
        out["screens_blink_py_sha256"] = code
    return out


def main():
    # main() builds a TorchEngine directly; this is only for anything it calls later.
    os.environ["BLINK_ENGINE"] = "torch"
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local path of the published weights")
    ap.add_argument("--model-id", default=blink.MODEL_ID)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", default="replay.json")
    ap.add_argument("--vision", action="store_true",
                    help="read screenshots with the checkpoint's own vision tower and record the Screen click presets")
    ap.add_argument("--screens-only", action="store_true",
                    help="record only the screenshot presets and add them to --out, this model's existing recording")
    a = ap.parse_args()
    if a.screens_only and not a.vision:
        ap.error("--screens-only needs --vision")

    import torch

    name = a.model_id.rstrip("/").split("/")[-1]
    eng = blink.TorchEngine(a.model, None, blink.TEMPERATURE, vision=a.vision, model_name=name)
    eng.model_id = a.model_id
    blink._ENGINE = eng
    reqs = ([] if a.screens_only else bundled_requests()) + (screen_requests() if a.vision else [])
    for _, state, qs in reqs[:3]:
        blink.decide(state, qs)  # warm-up
    cache = record(eng, reqs, a.repeats, torch.cuda.synchronize)
    digest = weights_digest(a.model)
    code = file_digest(blink.__file__)  # the published blink.py this recording came from
    if a.screens_only:
        out = merge(json.loads(Path(a.out).read_text(encoding="utf-8")), cache, a.model_id, digest, code)
    else:
        out = {
            "model": a.model_id,
            "weights_sha256": digest,
            "blink_py_sha256": code,
            "engine": "TorchEngine (bf16 weights, FP32 label projection"
                      + (", own vision tower, first image layout)" if a.vision else ")"),
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "requests": cache,
        }
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}: {len(out['requests'])} requests ({len(cache)} recorded now)")


if __name__ == "__main__":
    main()
