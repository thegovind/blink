"""Free-form asks → one blink request. This is the only step on the page that generates text.

An untouched instruct model (default Qwen/Qwen3.5-4B, Apache-2.0) drafts a state and a typed question
as JSON; the draft is normalised and checked with blink.validate before blink decides in one pass.
BLINK_MOCK=1 swaps in a small rule-based drafter so the page and tests run without a model.
"""
from __future__ import annotations

import json
import os
import re
import time

import blink

AUTHOR_MODEL = os.environ.get("BLINK_AUTHOR", "Qwen/Qwen3.5-4B")
# pinned so every visitor gets the same drafter; override with BLINK_AUTHOR_REVISION
AUTHOR_REVISION = os.environ.get("BLINK_AUTHOR_REVISION") or (
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" if AUTHOR_MODEL == "Qwen/Qwen3.5-4B" else None)
MAX_ASK_CHARS = 4000
MAX_NEW_TOKENS = 256
MAX_DRAFT_OPTIONS = 26
GPU_SECONDS = int(os.environ.get("BLINK_AUTHOR_GPU_DURATION", "30"))


class AuthorError(ValueError):
    """The ask could not be turned into a valid request; the message is safe to show.

    `raw` keeps the drafter's text for logs; it is never shown to visitors."""

    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


SYSTEM = """You turn a user's free-form ask into ONE typed decision for a classifier that can only pick among the options you list. Reply with JSON only.

Format:
{"state": "<the text or facts to judge, copied from the ask>", "questions": {"<snake_case_name>": <question>}}

<question> is exactly one of:
{"type": "noul", "instructions": "<a yes/no question>"}
{"type": "choice", "instructions": "<the question>", "criteria": {"<snake_case_key>": "<short option>", ...}}
{"type": "score", "instructions": "<the question>", "criteria": ["<lowest level>", "...", "<highest level>"]}

Rules:
- noul for yes/no asks; choice to pick one answer (2 to 8 options covering every plausible answer, including the right one); score to rate on an ordered scale (3 to 5 levels, low to high).
- Never answer the question and never hint which option is right.
- Keep the user's own text in "state". If the ask is only a question, put the question in "state".
- List options in a neutral order.
- If the ask is not a question with a small set of answers (for example: write, translate, summarise, explain), reply {"error": "not a decision"}."""

SHOTS = [
    ('Is this email phishing? "Your mailbox is full. Log in here within 24 hours to keep receiving mail."',
     {"state": "Your mailbox is full. Log in here within 24 hours to keep receiving mail.",
      "questions": {"phishing": {"type": "noul", "instructions": "Is this email a phishing attempt?"}}}),
    ("Which planet is closest to the sun?",
     {"state": "Which planet is closest to the sun?",
      "questions": {"closest_planet": {"type": "choice", "instructions": "Which planet is closest to the sun?",
                                       "criteria": {"venus": "Venus", "mercury": "Mercury", "earth": "Earth",
                                                    "mars": "Mars"}}}}),
    ('How happy is this review: "Arrived late but works great, would buy again"',
     {"state": "Arrived late but works great, would buy again",
      "questions": {"happiness": {"type": "score", "instructions": "How happy is the reviewer?",
                                  "criteria": ["very unhappy", "unhappy", "neutral", "happy", "very happy"]}}}),
    ("Summarise this article about solar panels for me", {"error": "not a decision"}),
]


def messages(ask: str) -> list[dict]:
    m = [{"role": "system", "content": SYSTEM}]
    for user, answer in SHOTS:
        m += [{"role": "user", "content": user},
              {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]
    return m + [{"role": "user", "content": ask}]


# --- parsing and normalisation -------------------------------------------------------

_SLUG = re.compile(r"[^a-z0-9]+")


def slug(s, fallback: str = "option") -> str:
    out = _SLUG.sub("_", str(s).lower()).strip("_")[:40]
    return out or fallback


def _unquote(v: str) -> str:
    v = str(v).strip()
    return v[1:-1].strip() if len(v) > 1 and v[0] == v[-1] and v[0] in "'\"" else v


def _unique(keys: list[str]) -> list[str]:
    """Distinct keys in order: a repeat gets the first free `_2`, `_3`, … checked against every key already used,
    so `a`, `a!` (→ `a`) and `a_2` stay three keys instead of silently merging two options."""
    used, out = set(), []
    for k in keys:
        cand, n = k, 1
        while cand in used:
            n += 1
            cand = f"{k}_{n}"
        used.add(cand)
        out.append(cand)
    return out


def _closers(s: str) -> str | None:
    """Closing brackets missing at the end of s (outside strings), or None if s ends inside a string."""
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    return None if in_str else "".join(reversed(stack))


def first_json(text: str):
    """The first JSON object in text. A draft that stops before its last closing braces is closed and
    re-parsed (small models often drop one); anything else malformed returns None."""
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:].strip()
    try:
        obj, _ = json.JSONDecoder().raw_decode(s)
        return obj
    except json.JSONDecodeError:
        pass
    closers = _closers(s)
    if not closers:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s + closers)
        return obj
    except json.JSONDecodeError:
        return None


def _question(q) -> dict:
    if not isinstance(q, dict):
        raise AuthorError("The drafted question was invalid.")
    qtype = str(q.get("type", "")).strip().lower()
    qtype = {"yes/no": "noul", "yesno": "noul", "boolean": "noul", "bool": "noul"}.get(qtype, qtype)
    instructions = str(q.get("instructions") or "").strip()
    if not instructions:
        raise AuthorError("The drafted question was empty.")
    crit = q.get("criteria")
    if qtype == "noul":
        out = {"type": "noul", "instructions": instructions}
        if isinstance(crit, dict) and any(k in crit for k in ("true", "false", "yes", "no")):
            out["criteria"] = {"true": str(crit.get("true", crit.get("yes", ""))),
                               "false": str(crit.get("false", crit.get("no", "")))}
        return out
    if qtype == "choice":
        if isinstance(crit, dict):
            pairs = [(str(k), str(v)) for k, v in crit.items()]
        elif isinstance(crit, list):
            pairs = [(str(v), str(v)) for v in crit]
        else:
            raise AuthorError("The drafted choice had no options.")
        pairs = [(k, _unquote(v) or k) for k, v in pairs if str(v).strip() or str(k).strip()]
        seen = set()
        pairs = [p for p in pairs if not (p[1].lower() in seen or seen.add(p[1].lower()))]
        if not 2 <= len(pairs) <= MAX_DRAFT_OPTIONS:
            raise AuthorError(f"The draft gave {len(pairs)} options; a choice needs 2 to {MAX_DRAFT_OPTIONS}.")
        keys = _unique([slug(k) for k, _ in pairs])
        return {"type": "choice", "instructions": instructions, "criteria": dict(zip(keys, (v for _, v in pairs)))}
    if qtype == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
            raise AuthorError("The drafted score needs 2 to 10 levels.")
        return {"type": "score", "instructions": instructions, "criteria": [str(c) for c in crit]}
    raise AuthorError(f"The draft used an unsupported question type {qtype!r}.")


def normalise(obj, ask: str) -> dict:
    """{'state', 'questions'} ready for blink.decide, or AuthorError."""
    if isinstance(obj, dict) and obj.get("error"):
        raise AuthorError("blink answers questions with a few possible answers. Ask a question with clear options.")
    if not isinstance(obj, dict) or not isinstance(obj.get("questions"), dict) or not obj["questions"]:
        raise AuthorError("Couldn't draft a question with clear answers. Try rewording it.")
    if len(obj["questions"]) > 3:
        raise AuthorError("Ask one question at a time.")
    state = obj.get("state")
    state = state.strip() if isinstance(state, str) and state.strip() else ask.strip()
    keys = _unique([slug(k, "question") for k in obj["questions"]])
    questions = {k: _question(q) for k, q in zip(keys, obj["questions"].values())}
    try:
        blink.validate(questions)
    except blink.BlinkError as e:
        raise AuthorError(f"The draft isn't a valid request: {e}") from e
    return {"state": state, "questions": questions}


def check_ask(ask) -> str:
    ask = ask if isinstance(ask, str) else ""
    ask = ask.strip()
    if not ask:
        raise AuthorError("Type a question first.")
    if len(ask) > MAX_ASK_CHARS:
        raise AuthorError(f"That's {len(ask):,} characters; asks are limited to {MAX_ASK_CHARS:,}.")
    return ask


# --- engines --------------------------------------------------------------------------

_LIVE: dict = {}


class MockAuthor:
    """Rule-based stand-in for tests and local UI work. Drafts are simple and deterministic."""

    name = "mock"
    model_id = "mock"

    def generate(self, ask: str) -> tuple[str, int, float]:
        low = ask.lower().strip()
        if re.match(r"(write|translate|summari[sz]e|explain)\b", low):
            return json.dumps({"error": "not a decision"}), 0, 0.0
        word = re.search(r"\bin (?:the word )?['\"]?([a-z]+)['\"]?\s*\??$", low)
        if low.startswith("how many"):
            state = word.group(1) if word else ask
            q = {"type": "choice", "instructions": ask.rstrip("?") + "?",
                 "criteria": {str(n): str(n) for n in range(1, 5)}}
        elif " or " in low:
            head, _, tail = ask.rpartition(":")
            opts = [o.strip(" ?.!,") for o in re.split(r",\s*|\s+or\s+", tail or ask) if o.strip(" ?.!,")]
            q = {"type": "choice", "instructions": (head or ask).strip() or ask,
                 "criteria": {slug(o): o for o in opts[-6:]}}
            state = ask
        elif re.match(r"(how|rate)\b", low):
            q = {"type": "score", "instructions": ask, "criteria": ["not at all", "a little", "somewhat", "very", "extremely"]}
            state = ask
        else:
            q = {"type": "noul", "instructions": ask if ask.endswith("?") else ask + "?"}
            state = ask
        return json.dumps({"state": state, "questions": {"answer": q}}), 0, 0.0


class AuthorEngine:
    """Greedy JSON drafting with an untouched instruct model; weights placed like blink's TorchEngine."""

    name = "torch"

    def __init__(self, model_id: str = AUTHOR_MODEL, revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # the pin belongs to the configured drafter only
        revision = revision or (AUTHOR_REVISION if model_id == AUTHOR_MODEL else None)
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        load = dict(revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa")
        if blink._on_zero_gpu():
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load).to("cuda")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, **load)
        self.model.eval()
        self.key = id(self)
        _LIVE[self.key] = self

    def prompt(self, ask: str) -> str:
        return self.tok.apply_chat_template(messages(ask), tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)

    def generate(self, ask: str) -> tuple[str, int, float]:
        return _generate(self.key, self.prompt(ask))


@blink._gpu(GPU_SECONDS)
def _generate(key, prompt: str) -> tuple[str, int, float]:
    import torch

    eng = _LIVE[key]
    ids = eng.tok(prompt, return_tensors="pt").to(eng.model.device)
    pad = eng.tok.pad_token_id if eng.tok.pad_token_id is not None else eng.tok.eos_token_id
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        out = eng.model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=pad)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1000
    new = out[0, ids["input_ids"].shape[1]:]
    return eng.tok.decode(new, skip_special_tokens=True), int(new.shape[0]), ms


_ENGINE = None


def enabled() -> bool:
    """BLINK_AUTHOR=off hides free-form asks; mock mode always has them."""
    return os.environ.get("BLINK_MOCK") == "1" or AUTHOR_MODEL.lower() not in ("", "off", "none", "0")


def engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = MockAuthor() if os.environ.get("BLINK_MOCK") == "1" else AuthorEngine()
    return _ENGINE


def warm():
    """Load the drafter at startup (on ZeroGPU the weights must be placed at module level)."""
    return engine() if enabled() else None


def draft(ask) -> dict:
    """{'state', 'questions', 'author': {model, generated_tokens, model_ms}} or AuthorError."""
    ask = check_ask(ask)
    eng = engine()
    text, n_tokens, ms = eng.generate(ask)
    obj = first_json(text)
    if obj is None:
        raise AuthorError("Couldn't draft a question with clear answers. Try rewording it.", raw=text)
    try:
        req = normalise(obj, ask)
    except AuthorError as e:
        e.raw = text
        raise
    req["author"] = {"model": eng.model_id, "generated_tokens": n_tokens, "model_ms": round(ms, 1)}
    return req
