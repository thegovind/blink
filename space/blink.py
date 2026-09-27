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

import base64
import binascii
import hashlib
import io
import itertools
import json
import math
import os
import re
import secrets
import string
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from PIL import Image

# --- constants the deployment sets -------------------------------------------------

MODEL_ID = os.environ.get("BLINK_MODEL") or "thegovind/blink-4b"
MODEL_ID_27B = "thegovind/blink-27b"
MODEL_REVISION = os.environ.get("BLINK_REVISION") or None
VISION_BASES = {"blink-4b": "Qwen/Qwen3.5-4B", "blink-27b": "Qwen/Qwen3.8-27B"}
NATIVE_VISION_MODELS = {"blink-mimo-9b", "Qwen3.5-4B"}


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
VISION = os.environ.get("BLINK_VISION", "0").lower() in ("1", "true", "on", "yes")
VISION_TOWER = os.environ.get("BLINK_VISION_TOWER") or None
IMAGE_LAYOUTS = ("inline", "first")
IMAGE_LAYOUT_DEFAULT = "first"


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


class ImageError(BlinkError):
    """Invalid image input, with its location in the request body."""

    def __init__(self, message: str, loc: list):
        super().__init__(message)
        self.loc = loc


@dataclass(frozen=True)
class ImageLimits:
    max_images: int = 2
    max_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 1920 * 1088
    max_source_pixels: int = 20_000_000
    min_pixels: int = 65_536
    factor: int = 32


@dataclass(frozen=True)
class ImageRequest:
    state: object
    marked_state: object
    images: tuple[Image.Image, ...]
    markers: tuple[str, ...]
    limits: ImageLimits
    loc: list


_NO_IMAGES = object()


@dataclass(frozen=True)
class ImageSubmission:
    state: object
    images: object
    limits: ImageLimits
    loc: list


MAX_HEADER = 256
_IMAGE_PREFIX = "data:image/"
_CANONICAL_IMAGE_URI = re.compile(r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]*={0,2}",
                                  re.IGNORECASE)
_IMAGE_DATA = re.compile(r"data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/]+={0,2})\Z", re.IGNORECASE)
_IMAGE_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
_CANONICAL_IMAGE_HEADERS = {f"data:{mime};base64" for mime in _IMAGE_FORMATS}


def _scan_image_uris(value: str) -> Iterator[tuple[int, int, str | None]]:
    """Yield canonical URI spans or invalid-header errors with bounded header lookahead."""
    low = value.lower()
    if len(low) != len(value):
        # Unicode lowercasing can expand; retain the original offsets for payload extraction.
        low = "".join(char.lower() if len(char.lower()) == 1 else char for char in value)
    pos = 0
    while True:
        start = low.find(_IMAGE_PREFIX, pos)
        if start < 0:
            return
        comma = low.find(",", start, start + MAX_HEADER)
        if comma < 0:
            pos = start + len(_IMAGE_PREFIX)
            continue
        header = low[start:comma]
        compact = "".join(header.split())
        if not compact.endswith(";base64"):
            pos = start + len(_IMAGE_PREFIX)
            continue
        if compact not in _CANONICAL_IMAGE_HEADERS or header != compact:
            error = ("image data URI parameters or whitespace are not supported"
                     if header != compact or compact.count(";") > 1
                     else "image must be a data:image/png, image/jpeg, or image/webp URI with base64 data")
            yield start, comma + 1, error
            pos = comma + 1
            continue
        match = _CANONICAL_IMAGE_URI.match(value, start)
        if match is None:
            yield start, comma + 1, "image must be a data:image/png, image/jpeg, or image/webp URI with base64 data"
            pos = comma + 1
            continue
        yield start, match.end(), None
        pos = match.end()


def image_limits() -> ImageLimits:
    """Read image-only settings lazily, without changing the text-only path."""
    settings = {
        "MAX_IMAGES": ("max_images", 1, 8),
        "MAX_IMAGE_BYTES": ("max_bytes", 1024, 32 * 1024 * 1024),
        "MAX_IMAGE_PIXELS": ("max_pixels", 65_536, 16_777_216),
        "MAX_IMAGE_SOURCE_PIXELS": ("max_source_pixels", 65_536, 64_000_000),
    }
    defaults = ImageLimits()
    values = {}
    for key, (field, lo, hi) in settings.items():
        name = f"BLINK_{key}"
        try:
            n = int(os.environ.get(name, str(getattr(defaults, field))))
        except ValueError:
            raise BlinkError(f"{name} must be an integer from {lo} to {hi}") from None
        if not lo <= n <= hi:
            raise BlinkError(f"{name} must be from {lo} to {hi}")
        values[field] = n
    return replace(defaults, **values)


def vision_source(model_name: str, spec: str) -> tuple[str, str]:
    """Only graft a text checkpoint with its own base model's vision tower."""
    expected = VISION_BASES.get(model_name)
    if expected is None:
        raise BlinkError("--vision-tower is only supported for blink-4b and blink-27b")
    source, separator, revision = spec.rpartition("@")
    if not separator or not source or not revision:
        raise BlinkError("--vision-tower must be a repo@revision")
    if source != expected and not (
        os.path.isdir(source) and os.path.basename(os.path.normpath(source)) == expected.split("/")[-1]
    ):
        raise BlinkError(f"--vision-tower for {model_name} must use {expected}@revision")
    return source, revision


def _image_size(height: int, width: int, limits: ImageLimits) -> tuple[int, int]:
    """Use the vision processor's factor-aligned min/max pixel resize policy."""
    factor = limits.factor
    if max(height, width) / min(height, width) > 200:
        raise BlinkError("image aspect ratio must be at most 200:1")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > limits.max_pixels:
        scale = math.sqrt(height * width / limits.max_pixels)
        h = max(factor, math.floor(height / scale / factor) * factor)
        w = max(factor, math.floor(width / scale / factor) * factor)
    elif h * w < limits.min_pixels:
        scale = math.sqrt(limits.min_pixels / (height * width))
        h, w = math.ceil(height * scale / factor) * factor, math.ceil(width * scale / factor) * factor
    if not limits.min_pixels <= h * w <= limits.max_pixels:
        raise BlinkError(f"image cannot fit between {limits.min_pixels} and {limits.max_pixels} pixels")
    return h, w


def _decode_image(uri: str, index: int, loc: list, limits: ImageLimits) -> Image.Image:
    from PIL import Image, ImageOps, UnidentifiedImageError

    match = _IMAGE_DATA.fullmatch(uri)
    if match is None:
        raise ImageError("image must be a data:image/png, image/jpeg, or image/webp URI with base64 data", loc)
    mime, payload = match.group(1).lower(), match.group(2)
    if len(payload) > 4 * math.ceil(limits.max_bytes / 3):
        raise ImageError(f"image {index} exceeds {limits.max_bytes} decoded bytes", loc)
    try:
        raw = base64.b64decode(payload, validate=True)
    except binascii.Error:
        raise ImageError(f"image {index} has invalid base64 data", loc) from None
    if not raw or len(raw) > limits.max_bytes:
        raise ImageError(f"image {index} must contain 1-{limits.max_bytes} decoded bytes", loc)
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.format != _IMAGE_FORMATS[mime]:
                raise ImageError(f"image {index} does not match its {mime} MIME type", loc)
            width, height = opened.size
            if not width or not height or width * height > limits.max_source_pixels:
                raise ImageError(f"image {index} exceeds {limits.max_source_pixels} source pixels", loc)
            img = ImageOps.exif_transpose(opened)
            try:
                new_height, new_width = _image_size(img.height, img.width, limits)
            except BlinkError as exc:
                raise ImageError(f"image {index}: {exc}", loc) from exc
            img = img.convert("RGB")
            if img.size != (new_width, new_height):
                img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            return img
    except ImageError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError) as exc:
        raise ImageError(f"image {index} cannot be decoded: {exc}", loc) from exc


def contains_image_uri(value) -> bool:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str) and next(_scan_image_uris(item), None) is not None:
            return True
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return False


def _map_state_values(value, loc: list, transform):
    if not isinstance(value, (dict, list)):
        return transform(value, loc)
    result = {} if isinstance(value, dict) else [None] * len(value)

    def entries(node):
        return iter(node.items()) if isinstance(node, dict) else iter(enumerate(node))

    pending = [(entries(value), result, loc)]
    while pending:
        children, target, parent_loc = pending[-1]
        try:
            key, child = next(children)
        except StopIteration:
            pending.pop()
            continue
        child_loc = [*parent_loc, key]
        if isinstance(child, (dict, list)):
            nested = {} if isinstance(child, dict) else [None] * len(child)
            target[key] = nested
            pending.append((entries(child), nested, child_loc))
        else:
            target[key] = transform(child, child_loc)
    return result


def inspect_images(state, images=_NO_IMAGES, limits: ImageLimits | None = None) -> ImageSubmission | None:
    """Locate and count image submissions without decoding pixels or base64."""
    if images is _NO_IMAGES and not contains_image_uri(state):
        return None
    if images is not _NO_IMAGES and not isinstance(images, list):
        raise ImageError("images must be a list of data URIs", ["body", "images"])
    count, first_loc, resolved = 0, None, limits

    def record(loc: list) -> None:
        nonlocal count, first_loc, resolved
        if resolved is None:
            resolved = image_limits()
        count += 1
        if count > resolved.max_images:
            raise ImageError(f"at most {resolved.max_images} images per request", loc)
        if first_loc is None:
            first_loc = loc

    pending = [(state, ["body", "state"])]
    while pending:
        value, loc = pending.pop()
        if isinstance(value, str):
            for _, _, error in _scan_image_uris(value):
                if error is not None:
                    raise ImageError(error, loc)
                record(loc)
        elif isinstance(value, dict):
            pending.extend((child, [*loc, key]) for key, child in reversed(tuple(value.items())))
        elif isinstance(value, list):
            pending.extend((value[i], [*loc, i]) for i in range(len(value) - 1, -1, -1))
    if images is not _NO_IMAGES:
        for i, uri in enumerate(images):
            loc = ["body", "images", i]
            if not isinstance(uri, str):
                raise ImageError(f"image {count + 1} must be a data URI", loc)
            first = next(_scan_image_uris(uri), None)
            if first is None or first[0] != 0:
                raise ImageError("image must be a data:image/png, image/jpeg, or image/webp URI with base64 data", loc)
            if first[2] is not None:
                raise ImageError(first[2], loc)
            record(loc)
    return ImageSubmission(state, images, resolved, first_loc) if count else None


def extract_images(state, images=_NO_IMAGES, limits: ImageLimits | None = None) -> ImageRequest | None:
    """Lift inline or top-level data URIs; keep unique internal markers for exact image placement."""
    submission = inspect_images(state, images, limits)
    if submission is None:
        return None
    limits = submission.limits
    found: list[Image.Image] = []
    markers: list[str] = []
    first_loc = None
    nonce = secrets.token_hex(16)

    def add(uri: str, loc: list) -> str:
        nonlocal first_loc
        index = len(found) + 1
        if index > limits.max_images:
            raise ImageError(f"at most {limits.max_images} images per request", loc)
        decoded = _decode_image(uri, index, loc, limits)
        marker = f"BLINK_IMAGE_{nonce}_{index}_END"
        found.append(decoded)
        markers.append(marker)
        if first_loc is None:
            first_loc = loc
        return marker

    def lift(value, loc: list):
        if isinstance(value, str):
            parts = []
            offset = 0
            for start, end, error in _scan_image_uris(value):
                if error is not None:
                    raise ImageError(error, loc)
                parts.extend((value[offset:start], add(value[start:end], loc)))
                offset = end
            if parts:
                parts.append(value[offset:])
                return "".join(parts)
        return value

    marked = _map_state_values(state, ["body", "state"], lift)
    if images is not _NO_IMAGES:
        attachments = []
        for i, uri in enumerate(images):
            if not isinstance(uri, str):
                raise ImageError(f"image {len(found) + 1} must be a data URI", ["body", "images", i])
            attachments.append(add(uri, ["body", "images", i]))
        if attachments:
            if isinstance(marked, str):
                marked = marked + ("\n" if marked else "") + "\n".join(attachments)
            elif isinstance(marked, list):
                marked = [*marked, *attachments]
            else:
                marked = {"state": marked, "images": attachments}
    if not found:
        return None
    names = {marker: f"[image {i}]" for i, marker in enumerate(markers, 1)}

    def visible(value, _loc):
        if isinstance(value, str):
            for marker, placeholder in names.items():
                value = value.replace(marker, placeholder)
            return value
        return value

    return ImageRequest(_map_state_values(marked, ["body", "state"], visible),
                        marked, tuple(found), tuple(markers), limits, first_loc)


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
                 token_budget: int = TOKEN_BUDGET, vision: bool = False, vision_tower: str | None = None,
                 limits: ImageLimits | None = None, model_name: str | None = None,
                 image_layout: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.temperature = float(temperature)
        self.token_budget = int(token_budget)
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        load = dict(revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa")
        model_name = (model_name or model_id).rstrip("/").split("/")[-1]
        if vision_tower:
            source, source_rev = vision_source(model_name, vision_tower)
        if vision and not vision_tower and model_name not in NATIVE_VISION_MODELS:
            raise BlinkError("--vision needs a model with its own vision tower or --vision-tower for a text model")
        self.accepts_images = bool(vision or vision_tower)
        self.image_layout = IMAGE_LAYOUT_DEFAULT if image_layout is None else image_layout
        if self.image_layout not in IMAGE_LAYOUTS:
            raise BlinkError("image_layout must be inline or first")
        if self.accepts_images:
            self.image_timing = os.environ.get("BLINK_IMAGE_TIMING", "0").lower() in ("1", "true", "on", "yes")
        if not self.accepts_images:
            if _on_zero_gpu():
                # ZeroGPU: load on the host, then place on cuda at module level (emulated until
                # a @spaces.GPU call attaches a real device).
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **load).to("cuda")
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, **load)
        else:
            from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

            if not vision_tower:
                source, source_rev = model_id, revision
            config = AutoConfig.from_pretrained(source, revision=source_rev)
            if getattr(config, "vision_config", None) is None:
                raise BlinkError("the selected vision tower has no vision configuration")
            self.processor = AutoProcessor.from_pretrained(source, revision=source_rev)
            if vision_tower:
                text_config = AutoConfig.from_pretrained(model_id, revision=revision)
                self._validate_graft_config(text_config, config)
            image_processor = self.processor.image_processor
            size = image_processor.size
            self.image_limits = replace(limits or image_limits(), min_pixels=int(size["shortest_edge"]),
                                        factor=int(image_processor.patch_size * image_processor.merge_size))
            if self.image_limits.max_pixels < self.image_limits.min_pixels:
                raise BlinkError(f"max-image-pixels must be at least {self.image_limits.min_pixels}")
            vision_load = dict(dtype=torch.bfloat16, attn_implementation="sdpa", revision=source_rev)
            if _on_zero_gpu():
                self.model = AutoModelForImageTextToText.from_pretrained(source, **vision_load).to("cuda")
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = AutoModelForImageTextToText.from_pretrained(source, device_map=device, **vision_load)
            if vision_tower:
                self._graft_text(model_id, revision)
        self.model.eval()
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.labels, self.label_ids = self._verify_labels()
        if self.accepts_images:
            self._verify_image_labels()
        self.key = id(self)
        _LIVE[self.key] = self

    def _validate_graft_config(self, text_config, base_config) -> None:
        base_text = getattr(base_config, "text_config", None)
        fields = ("model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
                  "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
                  "tie_word_embeddings", "layer_types")
        for field in fields:
            text_value = getattr(text_config, field, None)
            if text_value is None or text_value != getattr(base_text, field, None):
                raise BlinkError(f"incompatible vision tower: text {field} differs from the checkpoint")
        if getattr(base_config.vision_config, "out_hidden_size", None) != text_config.hidden_size:
            raise BlinkError("incompatible vision tower: visual output does not match text hidden size")
        for token, field in (("<|image_pad|>", "image_token_id"),
                             ("<|vision_start|>", "vision_start_token_id"),
                             ("<|vision_end|>", "vision_end_token_id")):
            expected = getattr(base_config, field, None)
            if (not isinstance(expected, int) or not 0 <= expected < text_config.vocab_size
                    or self.tok.convert_tokens_to_ids(token) != expected
                    or self.processor.tokenizer.convert_tokens_to_ids(token) != expected):
                raise BlinkError(f"incompatible vision tower: {field} differs from the text tokenizer")

    def _graft_text(self, model_id: str, revision) -> None:
        from graft_keys import text_to_parent
        from transformers import AutoModelForCausalLM

        text_model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, device_map="cpu",
                                                          dtype=self.model.dtype, attn_implementation="sdpa")
        target = self.model.state_dict()
        copied = {}
        for name, value in text_model.state_dict().items():
            try:
                dest = text_to_parent(name)
            except ValueError as exc:
                raise BlinkError(str(exc)) from exc
            if dest not in target or target[dest].shape != value.shape or dest in copied:
                raise BlinkError(f"text weight {name!r} cannot be grafted onto the vision model")
            copied[dest] = value
        missing, unexpected = self.model.load_state_dict(copied, strict=False)
        unfilled = [name for name in missing if name.startswith("model.language_model.") or name == "lm_head.weight"]
        if unexpected or unfilled:
            raise BlinkError(f"incomplete text graft: {len(unfilled)} missing language weights, "
                             f"{len(unexpected)} unexpected weights")
        del text_model, copied, target

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

    def _verify_image_labels(self) -> None:
        prompt = self.processor.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [{"type": "text", "text": "x"}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        tok = self.processor.tokenizer
        base = tok(prompt, add_special_tokens=False)["input_ids"]
        for label, expected in zip(self.labels, self.label_ids):
            got = tok(prompt + label, add_special_tokens=False)["input_ids"]
            if got[:len(base)] != base or got[len(base):] != [expected]:
                raise BlinkError(f"vision processor does not preserve the option-label token for {label}")

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

    def render_images(self, request: ImageRequest, questions: dict):
        if not self.accepts_images:
            raise ImageError("this model reads text only", request.loc)
        # The training renderer also uses this method on an uninitialized engine.
        layout = getattr(self, "image_layout", IMAGE_LAYOUT_DEFAULT)
        if layout not in IMAGE_LAYOUTS:
            raise BlinkError("image_layout must be inline or first")
        started = time.perf_counter() if getattr(self, "image_timing", False) else None
        work = []
        split = re.compile("(" + "|".join(map(re.escape, request.markers)) + ")")
        lookup = dict(zip(request.markers, request.images))
        visible_markers = {marker: f"[image {i}]" for i, marker in enumerate(request.markers, 1)}
        for qkey, q in questions.items():
            items = question_options(q)
            if len(items) > len(self.labels):
                raise BlinkError(f"{len(items)} options per choice; {len(self.labels)} labels verified")
            labels = self.labels[: len(items)]
            user = user_message(request.marked_state, q, labels, items)
            if layout == "first":
                images = list(request.images)
                content = [{"type": "image", "image": image} for image in images]
                content.append({"type": "text", "text": split.sub(
                    lambda match: visible_markers[match.group()], user)})
            else:
                content, images = [], []
                for part in split.split(user):
                    if part in lookup:
                        content.append({"type": "image", "image": lookup[part]})
                        images.append(lookup[part])
                    elif part:
                        content.append({"type": "text", "text": part})
            prompt = self.processor.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = self.processor(text=[prompt], images=images, return_tensors="pt")
            ids = inputs["input_ids"]
            if ids.shape[1] > MAX_INPUT_TOKENS:
                raise BlinkError(f"question {qkey!r} renders to {ids.shape[1]} tokens, over the maximum context "
                                 f"length of {MAX_INPUT_TOKENS}")
            grids = inputs["image_grid_thw"]
            if len(grids) != len(images):
                raise BlinkError(f"question {qkey!r} has mismatched image patches")
            patch = int(self.processor.image_processor.patch_size)
            merge = int(self.processor.image_processor.merge_size)
            patches = [int(grid[0] * grid[1] * grid[2]) for grid in grids]
            if any(count % merge**2 for count in patches):
                raise BlinkError(f"question {qkey!r} has an unaligned image patch grid")
            if any(count * patch**2 > self.image_limits.max_pixels for count in patches):
                raise BlinkError(f"question {qkey!r} exceeds {self.image_limits.max_pixels} processed image pixels")
            work.append({"qkey": qkey, "keys": [k for k, _ in items], "inputs": inputs,
                         "cand": self.label_ids[: len(items)], "image_pixels": sum(patches) * patch**2,
                         "visual_tokens": sum(count // merge**2 for count in patches)})
        if started is not None:
            self.last_image_processor_ms = (time.perf_counter() - started) * 1000
        return work

    def logits_images(self, request: ImageRequest, questions: dict) -> tuple[dict[str, list[float]], int]:
        out, n_tokens = {}, 0
        model_ms = processor_ms = encoder_ms = lm_ms = 0.0
        for qkey, question in questions.items():
            work = self.render_images(request, {qkey: question})
            item = work[0]
            if not out:
                self.last_image_pixels = item["image_pixels"]
                self.last_visual_tokens = item["visual_tokens"]
            n_tokens += int(item["inputs"]["input_ids"].shape[1])
            if getattr(self, "image_timing", False):
                processor_ms += self.last_image_processor_ms
            rows, elapsed, encoder, prefill = _forward_images(self.key, work)
            out[qkey] = rows[0]
            model_ms += elapsed
            encoder_ms += encoder
            lm_ms += prefill
            del work, item, rows
        self.last_model_ms = round(model_ms, 1)
        self.last_image_processor_ms = processor_ms
        self.last_vision_encoder_ms = round(encoder_ms, 3)
        self.last_lm_prefill_ms = round(lm_ms, 3)
        self.last_prefill_tokens = n_tokens
        return out, n_tokens

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


def _forward_shared(model, head, seqs, cands, prefix_len: int, pad_id: int, budget: int, out, device,
                    backbone=None) -> None:
    """Encode seqs[0][:prefix_len] once, then run every tail against an expanded copy of that cache."""
    import torch

    backbone = backbone if backbone is not None else model.model
    prefix = torch.tensor([seqs[0][:prefix_len]], dtype=torch.long, device=device)
    cache = backbone(input_ids=prefix, use_cache=True).past_key_values
    tails = [s[prefix_len:] for s in seqs]
    rows_cap = max(1, PREFIX_KV_TOKENS // (prefix_len + max(len(t) for t in tails)))
    for b in _batches([len(t) for t in tails], budget, max_rows=rows_cap):
        ids = _padded(tails, b, pad_id).to(device)
        h = backbone(input_ids=ids, past_key_values=_expand_cache(cache, len(b)), use_cache=True).last_hidden_state
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
    backbone = model.model.language_model if getattr(engine, "accepts_images", False) else model.model
    budget = getattr(engine, "token_budget", TOKEN_BUDGET)
    out: list = [None] * len(seqs)
    t0 = time.perf_counter()
    with torch.no_grad():
        shared = _shared_prefix_len(seqs) if getattr(engine, "prefix_cache", PREFIX_CACHE) else 0
        if shared > 0 and shared >= PREFIX_MIN_TOKENS:
            _forward_shared(model, head, seqs, cands, shared, engine.pad_id, budget, out, device, backbone)
        else:
            for b in _batches([len(s) for s in seqs], budget):
                ids = _padded(seqs, b, engine.pad_id).to(device)
                h = backbone(input_ids=ids, use_cache=False).last_hidden_state
                _read(model, head, h, [len(seqs[i]) for i in b], cands, b, out, device)
    return out, round((time.perf_counter() - t0) * 1000, 1)


def _timed_image_model(model, inputs, device):
    import torch

    use_cuda = device.type == "cuda"
    starts, sections = {"encoder": [], "lm": []}, {"encoder": [], "lm": []}

    def begin(name):
        stamp = torch.cuda.Event(enable_timing=True) if use_cuda else time.perf_counter()
        if use_cuda:
            stamp.record()
        starts[name].append(stamp)

    def end(name):
        start = starts[name].pop()
        stamp = torch.cuda.Event(enable_timing=True) if use_cuda else time.perf_counter()
        if use_cuda:
            stamp.record()
        sections[name].append((start, stamp) if use_cuda else (stamp - start) * 1000)

    handles = [
        model.model.visual.register_forward_pre_hook(lambda _mod, _args: begin("encoder")),
        model.model.visual.register_forward_hook(lambda _mod, _args, _out: end("encoder")),
        model.model.language_model.register_forward_pre_hook(lambda _mod, _args: begin("lm")),
        model.model.language_model.register_forward_hook(lambda _mod, _args, _out: end("lm")),
    ]
    try:
        hidden = model.model(**inputs, use_cache=False).last_hidden_state
    finally:
        for handle in handles:
            handle.remove()
    if not sections["encoder"] or not sections["lm"]:
        raise RuntimeError("image forward did not run both visual and language modules")
    if use_cuda:
        torch.cuda.synchronize(device)
        elapsed = lambda name: sum(start.elapsed_time(end) for start, end in sections[name])
    else:
        elapsed = lambda name: sum(sections[name])
    return hidden, elapsed("encoder"), elapsed("lm")


@_gpu(GPU_DURATION)
def _forward_images(key: int, work: list):
    import torch

    engine = _LIVE[key]
    model = engine.model
    device = next(model.parameters()).device
    out = []
    t0 = time.perf_counter()
    encoder_ms = lm_ms = 0.0
    with torch.no_grad():
        for item in work:
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in item["inputs"].items()}
            if getattr(engine, "image_timing", False):
                hidden, encoder, lm = _timed_image_model(model, inputs, device)
                encoder_ms += encoder
                lm_ms += lm
            else:
                hidden = model.model(**inputs, use_cache=False).last_hidden_state
            length = (int(inputs["attention_mask"][0].sum().item()) if "attention_mask" in inputs
                      else int(inputs["input_ids"].shape[1]))
            row = [None]
            _read(model, model.lm_head.weight, hidden, [length], [item["cand"]], [0], row, device)
            out.append(row[0])
    return out, round((time.perf_counter() - t0) * 1000, 1), round(encoder_ms, 3), round(lm_ms, 3)

# --- hybrid engine ------------------------------------------------------------------


class HybridEngine:
    """Recorded outputs for the bundled examples (instant, no GPU), the live model for
    everything else. The torch engine is built eagerly: ZeroGPU wants weights placed at startup."""

    name = "hybrid"

    def __init__(self, replay: "ReplayEngine | None", live: "TorchEngine"):
        self.replay, self.live = replay, live
        self.model_id = live.model_id
        self.temperature = live.temperature
        self.accepts_images = getattr(live, "accepts_images", False)
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
    model_name = model_id.rstrip("/").split("/")[-1]
    vision = VISION and model_name in NATIVE_VISION_MODELS
    tower = VISION_TOWER if model_name in VISION_BASES else None
    options = {"vision": vision, "vision_tower": tower} if vision or tower else {}
    if kind == "hybrid":
        return HybridEngine(_matching_replay(model_id), TorchEngine(model_id, revision, TEMPERATURE, **options))
    return TorchEngine(model_id, revision, TEMPERATURE, **options)


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


def _prepare_images(state, images, eng, limits: ImageLimits | None):
    live = getattr(eng, "live", eng)
    if isinstance(state, ImageRequest):
        if not getattr(live, "accepts_images", False):
            raise ImageError("this model reads text only", state.loc)
        return state
    submission = (state if isinstance(state, ImageSubmission)
                  else inspect_images(state, images, limits or getattr(live, "image_limits", None)))
    if submission is None:
        return state
    if not getattr(live, "accepts_images", False):
        raise ImageError("this model reads text only", submission.loc)
    return extract_images(submission.state, submission.images, submission.limits)


def decide(state, questions: dict, temperature: float | None = None, bias: dict | None = None,
           prefer: str = "live", model: str | None = None, images=_NO_IMAGES,
           limits: ImageLimits | None = None) -> dict:
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
    if isinstance(state, (ImageRequest, ImageSubmission)) or images is not _NO_IMAGES or contains_image_uri(state):
        live = getattr(eng, "live", eng)
        decode_started = time.perf_counter() if getattr(live, "image_timing", False) else None
        prepared = _prepare_images(state, images, eng, limits)
        decode_ms = (time.perf_counter() - decode_started) * 1000 if decode_started is not None else 0.0
    else:
        prepared = state
    if isinstance(prepared, ImageRequest):
        live = getattr(eng, "live", eng)
        raw, n_tokens = live.logits_images(prepared, questions)
        if isinstance(eng, HybridEngine):
            eng.last_source = "torch"
    elif isinstance(eng, HybridEngine):
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
    if isinstance(prepared, ImageRequest):
        meta["image_pixels"] = live.last_image_pixels
        meta["visual_tokens"] = live.last_visual_tokens
        if getattr(live, "image_timing", False):
            meta["image_preprocess_ms"] = round(decode_ms + live.last_image_processor_ms, 3)
            meta["vision_encoder_ms"] = live.last_vision_encoder_ms
            meta["lm_prefill_ms"] = live.last_lm_prefill_ms
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
    if any(len(req) == 3 or isinstance(req[0], (ImageRequest, ImageSubmission))
           or contains_image_uri(req[0]) for req in requests):
        out = [None] * len(requests)
        text_indices = [i for i, req in enumerate(requests)
                        if len(req) == 2 and not isinstance(req[0], (ImageRequest, ImageSubmission))
                        and not contains_image_uri(req[0])]
        if text_indices:
            text_results = decide_many([requests[i] for i in text_indices], temperature, model)
            for i, result in zip(text_indices, text_results):
                out[i] = result
        text_indices = set(text_indices)
        for i, req in enumerate(requests):
            if i in text_indices:
                continue
            try:
                out[i] = decide(req[0], req[1], temperature=temperature, model=model,
                                **({"images": req[2]} if len(req) == 3 else {}))
            except Exception as exc:  # noqa: BLE001 - isolate each image request, as for text rendering
                out[i] = exc
        return out
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

    def submit(self, state, questions: dict, images=_NO_IMAGES, limits: ImageLimits | None = None) -> dict:
        import concurrent.futures
        import queue

        validate(questions)  # a malformed request fails at once and never takes a place in the queue
        if images is not _NO_IMAGES or isinstance(state, (ImageRequest, ImageSubmission)) or contains_image_uri(state):
            eng = engine(self.model)
            live = getattr(eng, "live", eng)
            pending = (state if isinstance(state, (ImageRequest, ImageSubmission))
                       else inspect_images(state, images, limits or getattr(live, "image_limits", None)))
            if pending is not None:
                if not getattr(live, "accepts_images", False):
                    raise ImageError("this model reads text only", pending.loc)
                state = pending
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
