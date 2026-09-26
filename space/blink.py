"""blink — one-pass typed decisions.

A request is a `state` plus a map of typed questions. Every question is rendered
independently with the fixed "semif" template the model was trained on, the questions run
as right-padded prefills (packed into batches of at most BLINK_TOKEN_BUDGET padded tokens,
all inside one GPU call), and the answer is read from the next-token distribution over the
offered option labels. No tokens are generated.

Engines (BLINK_ENGINE): "torch" runs the model; "replay" serves label logits the real
model produced for the bundled examples (hosting without the model); "hybrid" serves those
recordings for the bundled examples and runs the model for everything else; "mock" is a
deterministic stand-in with fabricated numbers (BLINK_MOCK=1 also selects it). By default:
hybrid when CUDA and a recording are both present, torch when only CUDA is, replay otherwise.

BLINK_MODELS="repo[@revision],repo2" serves several models side by side (decide(..., model=...));
the default model's recording is replay.json, another model's replay-<repo name>.json, and a
recording is only used by the model that made it.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import string
import time

# --- constants the deployment sets -------------------------------------------------

MODEL_ID = os.environ.get("BLINK_MODEL") or "thegovind/blink-4b"
MODEL_ID_27B = "thegovind/blink-27b"
MODEL_REVISION = os.environ.get("BLINK_REVISION") or None


def _model_specs() -> list[tuple[str, str | None]]:
    """BLINK_MODELS serves several models side by side: comma-separated "repo" or "repo@revision",
    the first one the default. Unset, the single BLINK_MODEL / BLINK_REVISION pair is served."""
    specs = []
    for item in os.environ.get("BLINK_MODELS", "").split(","):
        repo, _, rev = item.strip().partition("@")
        if repo.strip():
            specs.append((repo.strip(), rev.strip() or None))
    return specs or [(MODEL_ID, MODEL_REVISION)]


MODEL_SPECS = _model_specs()
MODEL_ID, MODEL_REVISION = MODEL_SPECS[0]

# Default release temperature: 1.0, not fitted; callers may override it.
TEMPERATURE = float(os.environ.get("BLINK_TEMPERATURE", "1.0"))

MAX_OPTIONS = 255
MAX_QUESTIONS = 512
MAX_INPUT_TOKENS = int(os.environ.get("BLINK_MAX_INPUT_TOKENS", "131072"))  # as evaluated
TOKEN_BUDGET = int(os.environ.get("BLINK_TOKEN_BUDGET", "32768"))  # padded tokens per forward
# Shared-prefix prefill (opt-in, BLINK_PREFIX_CACHE=1): the questions of one request share the system prompt and the
# evidence, so that prefix is encoded once and each question runs only its own tail against a copy of the cached
# prefix. Same tokens, same positions; only floating-point order differs. Measured 4.8-6.9x less model time on long
# multi-question documents, but on ~18k-token documents blink-4b's bf16 answers drift slightly further from an FP32
# reference than the plain path's do, so it is off by default (experiments/t5/PREREG.md, gate A').
PREFIX_CACHE = os.environ.get("BLINK_PREFIX_CACHE", "0").lower() in ("1", "true", "on", "yes")


def _prefix_setting(name: str, default: int) -> int:
    """A shared-prefix setting, read only when BLINK_PREFIX_CACHE is on: a stray value can't break the default path."""
    if not PREFIX_CACHE:
        return default
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


PREFIX_MIN_TOKENS = _prefix_setting("BLINK_PREFIX_MIN_TOKENS", 256)  # below this the plain path is as fast
PREFIX_KV_TOKENS = _prefix_setting("BLINK_PREFIX_KV_TOKENS", 131072)  # prefix copies x length held at once
# Readout implementation: "rows" (release: per-question gather, upcast and GEMV) or "batched" (one gather, one FP32
# matmul and one host transfer per forward; same arithmetic, different FP32 accumulation order). Opt-in under E1.
READOUT = os.environ.get("BLINK_READOUT", "rows").strip().lower()
GPU_DURATION = int(os.environ.get("BLINK_GPU_DURATION", "60"))

SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

LABEL_POOL = list(string.ascii_uppercase) + [
    "".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)
]
GENERIC_KEY = re.compile(r"^(?:[A-Za-z]{1,2}|option[_ ]?\d+|opt\d+|\d+)$")

MOCK = os.environ.get("BLINK_MOCK", "") not in ("", "0", "false", "False")
ENGINE = "mock" if MOCK else os.environ.get("BLINK_ENGINE", "").strip().lower()
HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY_PATH = os.environ.get("BLINK_REPLAY", os.path.join(HERE, "replay.json"))


def replay_path(model_id: str | None = None) -> str:
    """The default model's recording is replay.json (or BLINK_REPLAY); any other served model's
    is replay-<repo name>.json beside this file."""
    if model_id is None or model_id == MODEL_ID:
        return REPLAY_PATH
    return os.path.join(HERE, f"replay-{model_id.rstrip('/').split('/')[-1]}.json")


class BlinkError(ValueError):
    """A request that cannot be rendered."""


class ReplayMiss(BlinkError):
    """Replay mode has no recorded output for this request."""


# --- rendering (pure python; the model was trained on exactly this) -----------------


def text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, indent=2)


def option_text(key, desc) -> str:
    k = str(key)
    if desc is None or (isinstance(desc, str) and not desc.strip()):
        return k
    d = text(desc)
    if GENERIC_KEY.match(k) or k.strip().lower() == d.strip().lower():
        return d
    return f"{k}: {d}"


def question_options(q: dict) -> list[tuple[str, str]]:
    """[(option_key, display_text)] in request order."""
    qtype = q.get("type")
    crit = q.get("criteria")
    if qtype == "choice":
        if isinstance(crit, dict):
            items = [(str(k), option_text(k, v)) for k, v in crit.items()]
        elif isinstance(crit, list):
            items = [(str(k), str(k)) for k in crit]
        else:
            raise BlinkError("choice needs criteria")
    elif qtype == "noul":
        t = f = None
        if isinstance(crit, dict):
            t = crit.get("true", crit.get("yes"))
            f = crit.get("false", crit.get("no"))
        items = [
            ("yes", "Yes" + (f" — {text(t)}" if t else "")),
            ("no", "No" + (f" — {text(f)}" if f else "")),
        ]
    elif qtype == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
            raise BlinkError("a score takes 2 to 10 levels")
        items = [
            (str(i), f"Level {i}: {text(c)}" if c is not None else f"Level {i}")
            for i, c in enumerate(crit)
        ]
    else:
        raise BlinkError(f"unsupported question type {qtype!r}")
    keys = [k for k, _ in items]
    if len(set(keys)) != len(keys):
        raise BlinkError("duplicate option keys")
    if not 1 <= len(items) <= MAX_OPTIONS:
        raise BlinkError(f"{len(items)} options per choice; supported 1-{MAX_OPTIONS}")
    return items


def user_message(state, q: dict, labels: list[str], items: list[tuple[str, str]]) -> str:
    return json.dumps(
        {
            "evidence": state,
            "criterion": text(q.get("instructions")).strip(),
            "options": [
                {"letter": lab, "description": d} for lab, (_, d) in zip(labels, items)
            ],
        },
        ensure_ascii=False,
    )


def validate(questions) -> None:
    if not isinstance(questions, dict) or not questions:
        raise BlinkError("questions must be a non-empty object")
    if len(questions) > MAX_QUESTIONS:
        raise BlinkError(f"{len(questions)} questions; supported 1-{MAX_QUESTIONS}")
    for qkey, q in questions.items():
        if not isinstance(q, dict):
            raise BlinkError(f"question {qkey!r} must be an object")
        question_options(q)


def as_state(state):
    """A state that looks like JSON is passed through as JSON; anything else is text."""
    if not isinstance(state, str) or not state.strip().startswith(("{", "[")):
        return state
    try:
        return json.loads(state)
    except json.JSONDecodeError:
        return state


# --- answer assembly ----------------------------------------------------------------


def answer_for(q: dict, keys: list[str], probs: list[float]) -> dict:
    p = {k: v for k, v in zip(keys, probs)}
    z = sum(p.values()) or 1.0
    p = {k: v / z for k, v in p.items()}
    qtype = q["type"]
    if qtype == "choice":
        order = list(p)
        best = max(order, key=lambda k: (p[k], -order.index(k)))
        K = len(order)
        conf = (p[best] - 1 / K) / (1 - 1 / K) if K > 1 else 1.0
        return {"type": "choice", "choice": best, "probabilities": p, "confidence": conf}
    if qtype == "noul":
        return {"type": "noul", "noul": p["yes"], "probabilities": p}
    levels = [str(i) for i in range(len(q["criteria"]))]
    ev = sum(int(k) * p[k] for k in levels)
    top = max(levels, key=lambda k: (p[k], -int(k)))
    K = len(levels)
    return {
        "type": "score",
        "score": ev,
        "probabilities": {k: p[k] for k in levels},
        "legend": {k: text(q["criteria"][int(k)]) for k in levels},
        "choice": top,
        # added after the fields above, which are unchanged: the choice formula over the levels (TypeSafe-shaped)
        "confidence": min(1.0, max(0.0, (p[top] - 1 / K) / (1 - 1 / K))),
    }


def softmax(logits: list[float], temperature: float) -> list[float]:
    z = [v / temperature for v in logits]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


# --- mock engine --------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "a an the of to and or is are was were be been it its this that for in on at "
    "with as by from not no yes if then than there here we you they i".split()
)


def _terms(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 2}


class MockEngine:
    """Deterministic stand-in: lexical overlap plus a stable pseudo-random jitter.

    Values are fabricated. They exist so the interface can be exercised without a GPU.
    A caller may pass `bias` to shape a bundled demo; the real engine ignores it.
    """

    name = "mock"
    temperature = TEMPERATURE

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        # every served model gets its own fabricated numbers; the default keeps the historic ones
        self._salt = "" if model_id == MODEL_ID else f"|{model_id}"

    def logits(self, state, questions: dict, bias: dict | None = None) -> tuple[dict[str, list[float]], int]:
        evidence = _terms(text(state))
        bias = bias or {}
        out, n_tokens = {}, 0
        for qkey, q in questions.items():
            items = question_options(q)
            labels = LABEL_POOL[: len(items)]
            prompt = SYSTEM + user_message(state, q, labels, items)
            n_tokens += max(1, len(prompt) // 4)
            crit = _terms(text(q.get("instructions")))
            hint = bias.get(qkey) or {}
            row = []
            for i, (key, disp) in enumerate(items):
                opt = _terms(disp)
                overlap = len(opt & evidence) / (len(opt) ** 0.5 + 1.0)
                cue = len(opt & crit) / (len(crit) ** 0.5 + 1.0)
                seed = f"{text(state)}|{text(q.get('instructions'))}|{key}|{i}{self._salt}".encode()
                jitter = int.from_bytes(hashlib.blake2b(seed, digest_size=4).digest(), "big")
                row.append(
                    2.6 * overlap
                    + 1.1 * cue
                    + 1.9 * (jitter / 2**32)
                    - 0.5 * i / len(items)
                    + float(hint.get(key, 0.0))
                )
            out[qkey] = row
        return out, n_tokens


# --- replay engine ------------------------------------------------------------------


def request_key(state, questions: dict) -> str:
    """Stable key for a request. Question and option order are part of the request."""
    if isinstance(state, str):
        state = state.replace("\r\n", "\n").strip()
    blob = json.dumps({"state": state, "questions": questions}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class ReplayEngine:
    """Label logits the trained model produced for the bundled examples, recorded with
    TorchEngine. Temperature and answer assembly still run live."""

    name = "replay"

    def __init__(self, path: str = REPLAY_PATH, temperature: float = TEMPERATURE):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.model_id = data["model"]
        self.hardware = data.get("hardware", "")
        self.temperature = float(temperature)
        self.cache = data["requests"]
        self.last = None

    def logits(self, state, questions: dict, bias: dict | None = None) -> tuple[dict[str, list[float]], int]:
        del bias
        hit = self.cache.get(request_key(state, questions))
        if hit is None:
            raise ReplayMiss(
                "Edited input. This page only has saved runs for the built-in examples and presets. "
                "Pick one of those, or run blink.py yourself to decide on any text."
            )
        self.last = hit
        return {k: list(v) for k, v in hit["logits"].items()}, int(hit["input_tokens"])


# --- torch engine -------------------------------------------------------------------


def _gpu(duration: int):
    """spaces.GPU when running on ZeroGPU, a no-op decorator anywhere else."""
    try:
        import spaces
    except Exception:
        return lambda fn: fn
    try:
        return spaces.GPU(duration=duration)
    except Exception:
        return spaces.GPU


def _on_zero_gpu() -> bool:
    try:
        from spaces.config import Config

        return bool(Config.zero_gpu)
    except Exception:
        return os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true")


# Live engines by id. The GPU function looks its engine up here instead of receiving it as an
# argument: on ZeroGPU the call runs in a worker process where only module-level state carries the
# real weights, and an argument would arrive as a copy.
_LIVE: dict = {}


class TorchEngine:
    """Prefill-only readout over the offered labels, all questions of a request in one GPU call."""

    name = "torch"

    def __init__(self, model_id: str = MODEL_ID, revision=None, temperature: float = TEMPERATURE,
                 token_budget: int = TOKEN_BUDGET):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.temperature = float(temperature)
        self.token_budget = int(token_budget)
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        load = dict(revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa")
        if _on_zero_gpu():
            # ZeroGPU: load on the host, then place on cuda at module level (emulated until
            # a @spaces.GPU call attaches a real device).
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load).to("cuda")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, **load)
        self.model.eval()
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.labels, self.label_ids = self._verify_labels()
        self.key = id(self)
        _LIVE[self.key] = self

    def _wrap(self, user: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _verify_labels(self) -> tuple[list[str], list[int]]:
        """Keep labels that are one token at the assistant boundary and decode back to themselves."""
        probe = self._wrap("x")
        base = self.tok(probe, add_special_tokens=False)["input_ids"]
        labels, ids = [], []
        for lab in LABEL_POOL:
            t = self.tok(probe + lab, add_special_tokens=False)["input_ids"]
            if (
                len(t) == len(base) + 1
                and t[: len(base)] == base
                and self.tok.decode(t[-1:]) == lab
            ):
                labels.append(lab)
                ids.append(t[-1])
            if len(labels) >= MAX_OPTIONS:
                break
        if len(set(ids)) != len(ids):
            raise BlinkError("label token ids collide")
        if len(labels) < 26 or labels[:26] != list(string.ascii_uppercase):
            raise BlinkError(f"only {len(labels)} verified single-token labels")
        return labels, ids

    def render(self, state, questions: dict):
        work = []
        for qkey, q in questions.items():
            items = question_options(q)
            if len(items) > len(self.labels):
                raise BlinkError(f"{len(items)} options per choice; {len(self.labels)} labels verified")
            labels = self.labels[: len(items)]
            prompt = self._wrap(user_message(state, q, labels, items))
            ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > MAX_INPUT_TOKENS:
                raise BlinkError(
                    f"question {qkey!r} renders to {len(ids)} tokens, over the maximum context length "
                    f"of {MAX_INPUT_TOKENS}"
                )
            work.append(
                {
                    "qkey": qkey,
                    "keys": [k for k, _ in items],
                    "ids": ids,
                    "cand": self.label_ids[: len(items)],
                }
            )
        return work

    def logits(self, state, questions: dict, bias: dict | None = None) -> tuple[dict[str, list[float]], int]:
        del bias  # demo-only shaping; the trained model reads the evidence instead
        work = self.render(state, questions)
        seqs = [w["ids"] for w in work]
        rows, self.last_model_ms = _forward(self.key, seqs, [w["cand"] for w in work])
        self.last_prefill_tokens = _prefill_tokens(seqs, getattr(self, "prefix_cache", PREFIX_CACHE))
        return (
            {w["qkey"]: r for w, r in zip(work, rows)},
            sum(len(s) for s in seqs),
        )

    def logits_many(self, requests: list) -> list[tuple[dict[str, list[float]], int]]:
        """Several already-validated requests in one GPU call: every question of every request is packed into the
        same padded forwards, then split back per request. Cross-request batching for a queueing server (E1 arm H1);
        each request's rows come only from its own prompts."""
        return self.logits_rendered([self.render(state, questions) for state, questions in requests])

    def logits_rendered(self, works: list) -> list[tuple[dict[str, list[float]], int]]:
        """logits_many for requests already rendered (one render() list per request), so a caller can render each
        request on its own and keep one request's rendering error away from the others."""
        seqs = [w["ids"] for work in works for w in work]
        cands = [w["cand"] for work in works for w in work]
        rows, self.last_model_ms = _forward(self.key, seqs, cands) if seqs else ([], 0.0)
        out, k = [], 0
        for work in works:
            part = rows[k:k + len(work)]
            k += len(work)
            out.append(({w["qkey"]: r for w, r in zip(work, part)}, sum(len(w["ids"]) for w in work)))
        return out


def _batches(lengths: list[int], budget: int, max_rows: int | None = None):
    """Shortest first; a batch closes when (longest x count) would pass the padded-token budget, or at max_rows.
    A sequence longer than the budget runs alone."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batch, longest = [], 0
    for i in order:
        grown = max(longest, lengths[i])
        if batch and (grown * (len(batch) + 1) > budget or (max_rows and len(batch) >= max_rows)):
            yield batch
            batch, grown = [], lengths[i]
        batch.append(i)
        longest = grown
    if batch:
        yield batch


def _shared_prefix_len(seqs: list[list[int]]) -> int:
    """Length of the token prefix common to every sequence, leaving each at least one token of its own
    (the answer position must be computed in the per-question pass)."""
    if len(seqs) < 2:
        return 0
    lo, hi = min(seqs), max(seqs)  # the common prefix of all = the common prefix of the lexicographic extremes
    n = min(len(lo), len(hi))
    p = 0
    while p < n and lo[p] == hi[p]:
        p += 1
    return min(p, min(len(s) for s in seqs) - 1)


def _prefill_tokens(seqs: list[list[int]], prefix_cache: bool) -> int:
    """Tokens the model actually computes for these prompts: the shared prefix once when it is reused."""
    total = sum(len(s) for s in seqs)
    shared = _shared_prefix_len(seqs) if prefix_cache else 0
    return total - (len(seqs) - 1) * shared if shared >= PREFIX_MIN_TOKENS else total


def _expand_cache(cache, repeats: int):
    """A deep copy of a batch-1 cache repeated `repeats` times along the batch axis, for every tensor it holds:
    full-attention keys/values and the linear-attention conv and recurrent states (the library's own
    batch_repeat_interleave only covers keys/values)."""
    import copy

    import torch

    cache = copy.deepcopy(cache)
    if repeats == 1:
        return cache

    def rep(x):
        return x.repeat_interleave(repeats, dim=0) if torch.is_tensor(x) and x.numel() else x

    for layer in cache.layers:
        for name in ("keys", "values"):
            if torch.is_tensor(getattr(layer, name, None)):
                setattr(layer, name, rep(getattr(layer, name)))
        for name in ("conv_states", "recurrent_states"):
            held = getattr(layer, name, None)
            if isinstance(held, dict):
                setattr(layer, name, {k: rep(v) for k, v in held.items()})
            elif isinstance(held, (list, tuple)):
                setattr(layer, name, type(held)(rep(v) for v in held))
    return cache


def _read(model, head, h, lengths, cands, rows, out, device):
    import torch

    if READOUT == "batched":
        return _read_batched(head, h, lengths, cands, rows, out, device)
    last = torch.tensor([n - 1 for n in lengths], device=device)
    h = h[torch.arange(len(rows), device=device), last].float()
    for r, i in enumerate(rows):
        w = head[torch.tensor(cands[i], device=device)].float()
        out[i] = (w @ h[r]).tolist()


def _read_batched(head, h, lengths, cands, rows, out, device):
    """The same FP32 readout as _read, for a whole forward at once: the batch's distinct label rows are gathered
    and upcast once, one FP32 matmul scores every question against them, and one host transfer brings them back.
    Only the order of FP32 accumulation differs from the per-question path (BLINK_READOUT=batched; opt-in)."""
    import torch

    last = torch.tensor([n - 1 for n in lengths], device=device)
    hs = h[torch.arange(len(rows), device=device), last].float()
    uniq = sorted({t for i in rows for t in cands[i]})
    col = {t: j for j, t in enumerate(uniq)}
    z = (hs @ head[torch.tensor(uniq, device=device)].float().T).tolist()
    for r, i in enumerate(rows):
        out[i] = [z[r][col[t]] for t in cands[i]]


def _padded(seqs, rows, pad_id):
    import torch

    L = max(len(seqs[i]) for i in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    for r, i in enumerate(rows):
        ids[r, : len(seqs[i])] = torch.tensor(seqs[i], dtype=torch.long)
    return ids


def _forward_shared(model, head, seqs, cands, prefix_len: int, pad_id: int, budget: int, out, device) -> None:
    """Encode seqs[0][:prefix_len] once, then run every tail against an expanded copy of that cache."""
    import torch

    prefix = torch.tensor([seqs[0][:prefix_len]], dtype=torch.long, device=device)
    cache = model.model(input_ids=prefix, use_cache=True).past_key_values
    tails = [s[prefix_len:] for s in seqs]
    rows_cap = max(1, PREFIX_KV_TOKENS // (prefix_len + max(len(t) for t in tails)))
    for b in _batches([len(t) for t in tails], budget, max_rows=rows_cap):
        ids = _padded(tails, b, pad_id).to(device)
        h = model.model(input_ids=ids, past_key_values=_expand_cache(cache, len(b)), use_cache=True).last_hidden_state
        _read(model, head, h, [len(tails[i]) for i in b], cands, b, out, device)


@_gpu(GPU_DURATION)
def _forward(key: int, seqs: list[list[int]], cands: list[list[int]]):
    """Right-padded maskless prefill; every layer is causal, so padding cannot reach the
    last real position. Label rows of lm_head are applied in FP32. With several questions over
    a long shared prefix, the prefix is encoded once (see PREFIX_CACHE)."""
    import torch

    engine = _LIVE[key]
    model = engine.model
    device = next(model.parameters()).device
    head = model.lm_head.weight
    budget = getattr(engine, "token_budget", TOKEN_BUDGET)
    out: list = [None] * len(seqs)
    t0 = time.perf_counter()
    with torch.no_grad():
        shared = _shared_prefix_len(seqs) if getattr(engine, "prefix_cache", PREFIX_CACHE) else 0
        if shared > 0 and shared >= PREFIX_MIN_TOKENS:
            _forward_shared(model, head, seqs, cands, shared, engine.pad_id, budget, out, device)
        else:
            for b in _batches([len(s) for s in seqs], budget):
                ids = _padded(seqs, b, engine.pad_id).to(device)
                h = model.model(input_ids=ids, use_cache=False).last_hidden_state
                _read(model, head, h, [len(seqs[i]) for i in b], cands, b, out, device)
    return out, round((time.perf_counter() - t0) * 1000, 1)


# --- hybrid engine ------------------------------------------------------------------


class HybridEngine:
    """Recorded outputs for the bundled examples (instant, no GPU), the live model for
    everything else. The torch engine is built eagerly: ZeroGPU wants weights placed at startup."""

    name = "hybrid"

    def __init__(self, replay: "ReplayEngine | None", live: "TorchEngine"):
        self.replay, self.live = replay, live
        self.model_id = live.model_id
        self.temperature = live.temperature
        self.hardware = replay.hardware if replay is not None else ""
        self.last_source = None
        self.last = None

    def logits(self, state, questions: dict, bias: dict | None = None,
               prefer: str = "live") -> tuple[dict[str, list[float]], int]:
        if (
            prefer == "saved"
            and self.replay is not None
            and self.replay.cache.get(request_key(state, questions)) is not None
        ):
            self.last_source = "replay"
            out = self.replay.logits(state, questions)
            self.last = self.replay.last
            return out
        self.last_source = "torch"
        return self.live.logits(state, questions)


# --- public entry point -------------------------------------------------------------

_ENGINE = None  # the default model's engine (tests inject one here)
_ENGINES: dict = {}  # every other served model's engine, by id


def _cuda() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def engine_kind() -> str:
    if ENGINE in ("mock", "replay", "torch", "hybrid"):
        return ENGINE
    recorded = os.path.exists(REPLAY_PATH)
    if _cuda():
        return "hybrid" if recorded else "torch"
    return "replay" if recorded else "torch"


def models() -> list[str]:
    """Model ids this deployment serves, the default first."""
    return [m for m, _ in MODEL_SPECS]


def _matching_replay(model_id: str):
    """The model's recording, only if that same model made it: a saved run must be what the live
    model would answer."""
    path = replay_path(model_id)
    if not os.path.exists(path):
        return None
    rec = ReplayEngine(path, TEMPERATURE)
    return rec if rec.model_id == model_id else None


def _build(model_id: str):
    revision = dict(MODEL_SPECS).get(model_id)
    kind = engine_kind()
    if kind == "mock":
        return MockEngine(model_id)
    if kind == "replay":
        return ReplayEngine(replay_path(model_id), TEMPERATURE)
    if kind == "hybrid":
        return HybridEngine(_matching_replay(model_id), TorchEngine(model_id, revision, TEMPERATURE))
    return TorchEngine(model_id, revision, TEMPERATURE)


def engine(model: str | None = None):
    global _ENGINE
    mid = model or MODEL_ID
    if mid not in dict(MODEL_SPECS):
        raise BlinkError(f"unknown model {mid!r}; this deployment serves {', '.join(models())}")
    if mid == MODEL_ID:
        if _ENGINE is None:
            _ENGINE = _build(mid)
        return _ENGINE
    if mid not in _ENGINES:
        _ENGINES[mid] = _build(mid)
    return _ENGINES[mid]


def warm() -> list:
    """Build every served model's engine now: ZeroGPU wants weights placed at startup."""
    return [engine(m) for m in models()]


def decide(state, questions: dict, temperature: float | None = None, bias: dict | None = None,
           prefer: str = "live", model: str | None = None) -> dict:
    """{state, questions} -> {answers, meta}. One forward pass, zero generated tokens.

    `bias` shapes the mock engine so the bundled examples read realistically without a
    GPU. It is discarded by the trained model. `prefer="saved"` lets the hybrid engine answer
    a bundled example from its recording (used for the first page render); every other call
    runs the model. `model` picks one of models() (default: the first).
    """
    validate(questions)
    eng = engine(model)
    T = float(temperature if temperature is not None else eng.temperature)
    if T <= 0:
        raise BlinkError("temperature must be positive")
    t0 = time.perf_counter()
    if isinstance(eng, HybridEngine):
        raw, n_tokens = eng.logits(state, questions, bias, prefer=prefer)
    else:
        raw, n_tokens = eng.logits(state, questions, bias)
    answers = {}
    for qkey, q in questions.items():
        keys = [k for k, _ in question_options(q)]
        answers[qkey] = answer_for(q, keys, softmax(raw[qkey], T))
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    source = getattr(eng, "last_source", None) or eng.name
    meta = {
        "model": getattr(eng, "model_id", MODEL_ID),
        "engine": source,
        "temperature": T,
        "input_tokens": n_tokens,
        "generated_tokens": 0,
        "latency_ms": latency_ms,
    }
    if source == "replay":
        meta["latency_ms"] = float(eng.last["latency_ms"])
        if eng.hardware:
            meta["recorded_on"] = eng.hardware
    elif source == "torch":
        model_ms = getattr(getattr(eng, "live", eng), "last_model_ms", None)
        if model_ms is not None:
            meta["model_ms"] = model_ms  # forward passes only; excludes any wait for a device
        prefill = getattr(getattr(eng, "live", eng), "last_prefill_tokens", None)
        if prefill is not None:
            meta["prefill_tokens"] = prefill  # tokens computed; input_tokens counts every question's full prompt
    return {"answers": answers, "meta": meta}


def decide_many(requests: list, temperature: float | None = None, model: str | None = None) -> list:
    """Several {state, questions} requests in one GPU call (E1 arm H1). Returns one item per request, in order: the
    same {answers, meta} dict decide() returns, or the exception that request raised (a BlinkError when blink can't
    answer it). Each request is validated and rendered on its own, so one request's failure never reaches another;
    if the packed forward itself fails, every request is answered alone instead. Engines without logits_many
    answer one request at a time."""
    eng = engine(model)
    T = float(temperature if temperature is not None else eng.temperature)
    if T <= 0:
        raise BlinkError("temperature must be positive")
    live = getattr(eng, "live", eng)
    packed = hasattr(live, "logits_rendered")
    out: list = [None] * len(requests)
    todo, works = [], []
    for i, (state, questions) in enumerate(requests):
        try:
            validate(questions)
            if packed:
                works.append(live.render(state, questions))
            todo.append(i)
        except Exception as exc:  # noqa: BLE001 - this request's own error (422 for a BlinkError, else 500)
            out[i] = exc
    t0 = time.perf_counter()
    got: dict = {}
    if packed and todo:
        try:
            got = dict(zip(todo, live.logits_rendered(works)))
        except Exception:  # noqa: BLE001 - e.g. out of memory on the packed batch: answer each request alone below
            got = {}
    n_packed = len(got)
    for i in todo:
        if i not in got:
            try:
                got[i] = (live if packed else eng).logits(*requests[i])
            except Exception as exc:  # noqa: BLE001
                got[i] = exc
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    for i in todo:
        res = got[i]
        if isinstance(res, Exception):
            out[i] = res
            continue
        try:
            raw, n_tokens = res
            questions = requests[i][1]
            answers = {q: answer_for(spec, [k for k, _ in question_options(spec)], softmax(raw[q], T))
                       for q, spec in questions.items()}
            out[i] = {"answers": answers, "meta": {"model": getattr(eng, "model_id", MODEL_ID), "engine": eng.name,
                                                   "temperature": T, "input_tokens": n_tokens, "generated_tokens": 0,
                                                   "latency_ms": latency_ms, "batched_requests": n_packed or 1}}
        except Exception as exc:  # noqa: BLE001
            out[i] = exc
    return out


class BlinkBusy(RuntimeError):
    """The batching queue is full: the server is over capacity (HTTP 503; retry shortly)."""


class Batcher:
    """Cross-request batching for a serving process (E1 arm H1). One worker thread owns the engine: it takes the
    oldest waiting request, waits up to `window_s` for more (at most `max_requests` in all) and decides them together
    with decide_many. At most `max_queued` requests wait; past that, submit() raises BlinkBusy at once. submit()
    blocks the calling thread until its own result is ready and returns it, or raises that request's own error."""

    MAX_WINDOW_S = 1.0
    MAX_REQUESTS = 64  # the largest group E1 measured

    def __init__(self, window_s: float, max_requests: int = 16, model: str | None = None, max_queued: int = 64):
        import math
        import queue
        import threading

        window_s, max_requests, max_queued = float(window_s), int(max_requests), int(max_queued)
        if not (math.isfinite(window_s) and 0 < window_s <= self.MAX_WINDOW_S):
            raise ValueError(f"batch window must be more than 0 and at most {self.MAX_WINDOW_S:g} s")
        if not 1 <= max_requests <= self.MAX_REQUESTS:
            raise ValueError(f"max_requests must be 1-{self.MAX_REQUESTS}")
        if max_queued < 1:
            raise ValueError("max_queued must be at least 1")
        self.window_s, self.max_requests, self.max_queued, self.model = window_s, max_requests, max_queued, model
        self.q = queue.Queue(maxsize=max_queued)
        self.worker = threading.Thread(target=self._loop, name="blink-batcher", daemon=True)
        self.worker.start()

    def alive(self) -> bool:
        return self.worker.is_alive()

    def submit(self, state, questions: dict) -> dict:
        import concurrent.futures
        import queue

        validate(questions)  # a malformed request fails at once and never takes a place in the queue
        if not self.worker.is_alive():
            raise RuntimeError("the batching worker has stopped")
        fut = concurrent.futures.Future()
        try:
            self.q.put_nowait((state, questions, fut))
        except queue.Full:
            raise BlinkBusy(f"{self.max_queued} requests are already waiting; retry shortly") from None
        while not fut.done():
            concurrent.futures.wait([fut], timeout=1.0)
            if not fut.done() and not self.worker.is_alive():
                raise RuntimeError("the batching worker has stopped")
        return fut.result()

    def _loop(self) -> None:
        import queue

        while True:
            batch = [self.q.get()]
            results = None
            try:
                deadline = time.monotonic() + self.window_s
                while len(batch) < self.max_requests:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    try:
                        batch.append(self.q.get(timeout=left))
                    except queue.Empty:
                        break
                results = decide_many([(s, q) for s, q, _ in batch], model=self.model)
            except Exception as exc:  # noqa: BLE001 - a failure outside any one request (e.g. the engine): each caller gets it
                results = [exc] * len(batch)
            finally:
                if results is None or len(results) != len(batch):
                    results = [None] * len(batch)
                for (_, _, fut), res in zip(batch, results):
                    if res is None:
                        fut.set_exception(RuntimeError("the batching worker lost this request"))
                    elif isinstance(res, BaseException):
                        fut.set_exception(res)
                    else:
                        fut.set_result(res)
