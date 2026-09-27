"""Next click on a screen: a screenshot, numbered boxes over its candidate elements, and a task.

The presets are the local browser-test fixture pages, rendered by build_screens.py into
screens/*.png with their boxes in screens/screens.json. A visitor's own screenshot is marked
here the same way, with the boxes they draw or a numbered grid.

blink-mimo-9b reads a screenshot with its own vision tower. The Space turns that on with
BLINK_VISION=1, which reaches no other served model (blink.NATIVE_VISION_MODELS), so blink-4b
stays text-only and no second tower is downloaded. The saved runs are recorded by
record_replay.py --vision; mock mode fabricates numbers, as it does for text.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import inspect
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass

import blink

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "screens")
META = os.path.join(FOLDER, "screens.json")
KEY = "screen"
VISION_NAME = "blink-mimo-9b"

# The wording blink's screenshot development readouts used (cua/build_requests.py), so a
# preset asks what was measured; the risk question is this page's own addition.
ELEMENT_ASK = "Which marked element should the agent act on next to make progress on the task?"
GRID_ASK = "Which numbered cell holds the element the agent should act on next to make progress on the task?"
DONE_ASK = "Is the task already complete, without another element action?"
DONE_CRITERIA = {"true": "No more element action is needed", "false": "Another element action is needed"}
RISKY_ASK = "Would the next click for this task be risky or hard to undo?"
RISKY_CRITERIA = {
    "true": "It pays, sends, deletes or shares something that cannot be taken back",
    "false": "It only opens, selects, searches or filters",
}

# the decision rule's cut-offs (verdict below), shared with what the page highlights
DONE_AT = 0.60
RISKY_AT = 0.50
SURE_AT = 0.45

BLUE = (36, 71, 245)  # blink's cobalt
WHITE = (255, 255, 255)
MIN_BOXES = 2
MAX_BOXES = 20
MAX_TASK = 400
GRIDS = (3, 4)
# an upload is held to blink's own image defaults (blink.ImageLimits)
UPLOAD_BYTES = blink.ImageLimits.max_bytes
UPLOAD_SOURCE_PIXELS = blink.ImageLimits.max_source_pixels
WORK_PIXELS = blink.ImageLimits.max_pixels
FACTOR = blink.ImageLimits.factor  # both sides of what blink reads are a multiple of this
PREVIEW_WIDTH = 1280
THUMB_WIDTH = 640
UPLOAD_FORMATS = ("PNG", "JPEG", "WEBP")

# what a visitor is told when a screenshot or its boxes can't be used
PROBLEM = {
    "task": "Add a task.",
    "long": f"Keep the task under {MAX_TASK} characters.",
    "line": "Box line {n}: use x1, y1, x2, y2.",
    "small": "Make the box on line {n} larger.",
    "few": f"Draw at least {MIN_BOXES} boxes, or clear them to use the grid.",
    "many": f"Use up to {MAX_BOXES} boxes.",
    "format": "Use a PNG, JPEG or WebP screenshot.",
    "bytes": f"Use a screenshot under {UPLOAD_BYTES // (1024 * 1024)} MB.",
    "pixels": "Use a screenshot under 20 megapixels.",
    "shape": "Use a screenshot no more than 200 times wider than it is tall.",
    "none": "Upload a screenshot first.",
}


class ScreenError(blink.BlinkError):
    """A screenshot, box list or task this page can't send."""


class ScreenMiss(blink.ReplayMiss):
    """No saved run for this screenshot, and no live model to ask."""


# --- which model reads screenshots ------------------------------------------------


def vision_model() -> str | None:
    """The served model with its own vision tower (the Qwen base drafts text and is never served here)."""
    for model in blink.models():
        if model.rstrip("/").split("/")[-1] == VISION_NAME:
            return model
    return None


def available() -> bool:
    return vision_model() is not None


# --- screenshots and their boxes ----------------------------------------------------


@dataclass(frozen=True)
class Box:
    n: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in the pixels blink reads
    tag: tuple[int, int, int, int]  # where its number is drawn
    at: str = "above"  # the number sits above, left of or inside the box
    role: str = ""
    name: str = ""


@dataclass(frozen=True)
class Shot:
    """One screenshot as blink reads it: marked, with its task."""

    key: str
    label: str
    task: str
    png: bytes  # the marked screenshot sent to the model
    size: tuple[int, int]
    boxes: tuple[Box, ...]
    grid: int = 0  # 3 or 4 when the boxes are a numbered grid
    view: str = ""  # a lighter copy to show (an upload); empty shows `png`
    source_size: tuple[int, int] | None = None  # an upload's own pixels, the ones boxes are typed in
    expect: tuple = ()  # (question, option) pairs; read only by the mock engine

    @functools.cached_property
    def uri(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")

    @property
    def upload(self) -> bool:
        return self.key == "upload"


@dataclass(frozen=True)
class Upload:
    """A visitor's screenshot, cleaned: RGB, within blink's pixel budget, and a light copy to draw on."""

    png: bytes
    size: tuple[int, int]
    source_size: tuple[int, int]
    view: str


def style(width: int) -> dict:
    """Stroke, halo and number size, all in proportion to the screenshot."""
    stroke = max(2, round(width / 360))
    tag = max(16, round(width / 42))
    return {"stroke": stroke, "halo": max(1, stroke // 2), "tag": tag, "font": max(11, round(tag * 0.72))}


@functools.lru_cache(maxsize=16)
def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "LiberationSans-Bold.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow before 10.1
        return ImageFont.load_default()


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _place_tag(box, tw: int, th: int, size, taken: list, grid: bool) -> tuple[tuple, str]:
    """Above the box's corner, else to its left, else just inside: the first spot that covers nothing."""
    W, H = size
    x1, y1, x2, y2 = box
    spots = [] if grid else [
        ((x1, y1 - th, x1 + tw, y1), "above"),
        ((x1 - tw, y1, x1, y1 + th), "left"),
    ]
    spots.append(((x1, y1, x1 + tw, y1 + th), "inside"))
    for rect, at in spots:
        if rect[0] < 0 or rect[1] < 0 or rect[2] > W or rect[3] > H:
            continue
        if at == "inside" or not any(_overlaps(rect, t) for t in taken):
            return rect, at
    x = min(max(0, x1), W - tw)
    y = min(max(0, y1), H - th)
    return (x, y, x + tw, y + th), "inside"


def layout(boxes, size, grid: int = 0, meta: list[dict] | None = None) -> list[Box]:
    """Where each box's number goes, numbered in the order given. mark() draws exactly this,
    and the page lays its own marks on the same spots."""
    W, H = size
    s = style(W)
    font = _font(s["font"])
    padded = [(b[0] - s["stroke"], b[1] - s["stroke"], b[2] + s["stroke"], b[3] + s["stroke"]) for b in boxes]
    taken = [] if grid else list(padded)
    placed = []
    for n, (box, pad) in enumerate(zip(boxes, padded), 1):
        left, _, right, _ = font.getbbox(str(n))
        th = s["tag"]
        tw = max(th, right - left + th // 2)
        rect, at = _place_tag(box if grid else pad, tw, th, (W, H), taken, bool(grid))
        taken.append(rect)
        info = (meta[n - 1] if meta and n - 1 < len(meta) else {}) or {}
        placed.append(Box(n=n, box=tuple(int(v) for v in box), tag=tuple(int(v) for v in rect), at=at,
                          role=str(info.get("role", "")), name=str(info.get("name", ""))))
    return placed


def mark(image, boxes, grid: int = 0, meta: list[dict] | None = None):
    """Burn numbered boxes into a copy of the screenshot, the way blink will read it.

    Returns the marked image and each Box with the spot its number was drawn."""
    from PIL import ImageDraw

    img = image.convert("RGB")
    W, H = img.size
    s = style(W)
    draw = ImageDraw.Draw(img)
    font = _font(s["font"])
    placed = layout(boxes, (W, H), grid, meta)
    if grid:
        line = max(1, s["stroke"] // 2)
        for x in sorted({b[0] for b in boxes} - {0}):
            draw.line([(x, 0), (x, H)], fill=WHITE, width=line + 2)
            draw.line([(x, 0), (x, H)], fill=BLUE, width=line)
        for y in sorted({b[1] for b in boxes} - {0}):
            draw.line([(0, y), (W, y)], fill=WHITE, width=line + 2)
            draw.line([(0, y), (W, y)], fill=BLUE, width=line)
    else:
        for x1, y1, x2, y2 in boxes:
            h = s["halo"]
            draw.rectangle([x1 - s["stroke"] - h, y1 - s["stroke"] - h, x2 + s["stroke"] + h, y2 + s["stroke"] + h],
                           outline=WHITE, width=h)
            draw.rectangle([x1 - s["stroke"], y1 - s["stroke"], x2 + s["stroke"], y2 + s["stroke"]],
                           outline=BLUE, width=s["stroke"])
    for b in placed:
        # PIL fills both end rows and columns; the tag's far edge is exclusive, as on the page
        draw.rectangle([b.tag[0], b.tag[1], b.tag[2] - 1, b.tag[3] - 1], fill=BLUE)
        draw.text(((b.tag[0] + b.tag[2]) / 2, (b.tag[1] + b.tag[3]) / 2), str(b.n), fill=WHITE, font=font,
                  anchor="mm")
    return img, placed


def grid_boxes(width: int, height: int, k: int) -> list[tuple[int, int, int, int]]:
    """k x k cells, numbered left to right, then top to bottom."""
    if k not in GRIDS:
        raise ScreenError(f"grid must be one of {GRIDS}")
    xs = [round(width * i / k) for i in range(k + 1)]
    ys = [round(height * i / k) for i in range(k + 1)]
    return [(xs[c], ys[r], xs[c + 1], ys[r + 1]) for r in range(k) for c in range(k)]


_NUM = r"-?\d+(?:\.\d+)?"
_LINE = re.compile(rf"^\(?\s*({_NUM})\s*[,;\s]\s*({_NUM})\s*[,;\s]\s*({_NUM})\s*[,;\s]\s*({_NUM})\s*\)?$")


def parse_boxes(text, width: int, height: int) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """One box per line, x1, y1, x2, y2 in the screenshot's own pixels; corners in any order."""
    boxes, problems = [], []
    for i, raw in enumerate(str(text or "").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        match = _LINE.match(line)
        if not match:
            problems.append(PROBLEM["line"].format(n=i))
            continue
        x1, y1, x2, y2 = (float(v) for v in match.groups())
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            problems.append(PROBLEM["small"].format(n=i))
            continue
        boxes.append((round(x1), round(y1), round(x2), round(y2)))
    if len(boxes) > MAX_BOXES:
        problems.append(PROBLEM["many"])
    return boxes, problems


def box_line(box) -> str:
    return ", ".join(str(int(round(v))) for v in box)


def add_box(text, box, width: int, height: int) -> str:
    """The box list with one more line; a box that is not four numbers inside the screenshot is dropped."""
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return str(text or "")
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return str(text or "")
    x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return str(text or "")
    lines = [ln for ln in str(text or "").splitlines() if ln.strip()]
    if len(lines) >= MAX_BOXES:
        return "\n".join(lines)
    return "\n".join(lines + [box_line((x1, y1, x2, y2))])


def drop_box(text, n) -> str:
    """The box list without its n-th box (counted as the page numbers them)."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return str(text or "")
    lines = [ln for ln in str(text or "").splitlines() if ln.strip()]
    if 1 <= n <= len(lines):
        del lines[n - 1]
    return "\n".join(lines)


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _jpeg_uri(image, width: int) -> str:
    from PIL import Image

    if image.width > width:
        image = image.resize((width, max(1, round(image.height * width / image.width))),
                             Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=84, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def load_upload(path) -> Upload:
    """A visitor's file -> an Upload, or a ScreenError that says what to change."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not path:
        raise ScreenError(PROBLEM["none"])
    try:
        nbytes = os.path.getsize(path)
    except (OSError, TypeError):
        raise ScreenError(PROBLEM["format"]) from None
    if nbytes > UPLOAD_BYTES:
        raise ScreenError(PROBLEM["bytes"])
    try:
        with Image.open(path) as opened:
            if opened.format not in UPLOAD_FORMATS:
                raise ScreenError(PROBLEM["format"])
            w, h = opened.size
            if not w or not h or w * h > UPLOAD_SOURCE_PIXELS:
                raise ScreenError(PROBLEM["pixels"])
            if max(w, h) / min(w, h) > 200:
                raise ScreenError(PROBLEM["shape"])
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except ScreenError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError):
        raise ScreenError(PROBLEM["format"]) from None
    source = image.size
    # blink's own resize rule, applied here once: the marks are then drawn on exactly the pixels it reads
    try:
        height, width = blink._image_size(source[1], source[0], blink.ImageLimits())
    except blink.BlinkError:
        raise ScreenError(PROBLEM["shape"]) from None
    if (width, height) != source:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return Upload(png=_png(image), size=image.size, source_size=source, view=_jpeg_uri(image, PREVIEW_WIDTH))


def _open(png: bytes):
    from PIL import Image

    with Image.open(io.BytesIO(png)) as opened:
        return opened.convert("RGB")


def upload_shot(upload: Upload | None, task: str, boxes_text: str = "", grid: int = 3) -> tuple[Shot | None, list[str]]:
    """The visitor's screenshot, marked with their boxes or, with none, a numbered grid."""
    if upload is None:
        return None, [PROBLEM["none"]]
    boxes, problems = parse_boxes(boxes_text, *upload.source_size)
    if problems:
        return None, problems
    if boxes and len(boxes) < MIN_BOXES:
        return None, [PROBLEM["few"]]
    work, grid = _work_boxes(upload, boxes, grid)
    marked, placed = mark(_open(upload.png), work, grid=grid)
    return Shot(key="upload", label="", task=str(task or "").strip(), png=_png(marked), size=upload.size,
                boxes=tuple(placed), grid=grid, view=upload.view, source_size=upload.source_size), []


def _work_boxes(upload: Upload, boxes, grid: int) -> tuple[list, int]:
    """Boxes typed in the upload's own pixels -> the pixels blink reads; none -> the grid's cells."""
    if not boxes:
        grid = grid if grid in GRIDS else GRIDS[0]
        return grid_boxes(*upload.size, grid), grid
    fx, fy = upload.size[0] / upload.source_size[0], upload.size[1] / upload.source_size[1]
    return [(round(x1 * fx), round(y1 * fy), round(x2 * fx), round(y2 * fy)) for x1, y1, x2, y2 in boxes], 0


def draft_shot(upload: Upload, boxes_text: str = "", grid: int = 3) -> tuple[Shot, list[str]]:
    """What the drawing surface shows before anything is sent: the boxes as typed, nothing burned in."""
    boxes, problems = parse_boxes(boxes_text, *upload.source_size)
    work, grid = _work_boxes(upload, boxes[:MAX_BOXES], grid)
    return Shot(key="upload", label="", task="", png=b"", size=upload.size,
                boxes=tuple(layout(work, upload.size, grid)), grid=grid, view=upload.view,
                source_size=upload.source_size), problems


def thumb(shot: Shot) -> str:
    """A small copy of exactly what blink reads, marks and all."""
    return _jpeg_uri(_open(shot.png), THUMB_WIDTH)


# --- the presets ------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _meta() -> dict:
    with open(META, encoding="utf-8") as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=1)
def presets() -> tuple[Shot, ...]:
    """The fixture screenshots, checked against the digests build_screens.py wrote."""
    shots = []
    for p in _meta()["presets"]:
        with open(os.path.join(FOLDER, p["image"]), "rb") as fh:
            png = fh.read()
        if hashlib.sha256(png).hexdigest() != p["sha256"]:
            raise ScreenError(f"screens/{p['image']} does not match screens.json; rerun build_screens.py")
        boxes = tuple(Box(n=b["n"], box=tuple(b["box"]), tag=tuple(b["tag"]), at=b["at"],
                          role=b.get("role", ""), name=b.get("name", "")) for b in p["boxes"])
        shots.append(Shot(key=p["key"], label=p["label"], task=p["task"], png=png, size=tuple(p["size"]),
                          boxes=boxes, expect=tuple(sorted((p.get("expect") or {}).items()))))
    return tuple(shots)


def preset(key: str) -> Shot:
    for shot in presets():
        if shot.key == key:
            return shot
    raise KeyError(key)


def with_task(shot: Shot, task: str) -> Shot:
    """The same screenshot asked about a different task (an edited preset)."""
    from dataclasses import replace

    return replace(shot, task=str(task or "").strip())


def task_problems(task) -> list[str]:
    task = str(task or "").strip()
    if not task:
        return [PROBLEM["task"]]
    if len(task) > MAX_TASK:
        return [PROBLEM["long"]]
    return []


# --- the request ----------------------------------------------------------------------


def questions(n: int, grid: int = 0) -> dict:
    """A choice over the numbered marks, then two yes/no checks; a new dict every call."""
    unit = "cell" if grid else "box"
    return {
        "element": {
            "type": "choice",
            "instructions": GRID_ASK if grid else ELEMENT_ASK,
            "criteria": {str(i): f"{unit} {i}" for i in range(1, n + 1)},
        },
        "done": {"type": "noul", "instructions": DONE_ASK, "criteria": dict(DONE_CRITERIA)},
        "risky": {"type": "noul", "instructions": RISKY_ASK, "criteria": dict(RISKY_CRITERIA)},
    }


def note(shot: Shot) -> str:
    n = len(shot.boxes)
    if shot.grid:
        return f"Use the attached screenshot with a numbered {shot.grid}x{shot.grid} grid, cells 1-{n}."
    return f"Use the attached screenshot with boxes 1-{n}."


def state(shot: Shot) -> dict:
    """The evaluated screenshot request's shape: the task, no history, a note and the image itself."""
    return {"task": shot.task, "previous_actions": [], "screenshot_note": note(shot), "screenshot": shot.uri}


def request(shot: Shot) -> tuple[dict, dict]:
    return state(shot), questions(len(shot.boxes), shot.grid)


def request_key(shot: Shot) -> str:
    """The replay key record_replay.py stores this screenshot's saved run under."""
    return blink.request_key(*request(shot))


def mock_bias(shot: Shot) -> dict:
    """Demo shaping for the mock engine, which cannot see; the trained model reads the pixels."""
    bias = {}
    for question, option in shot.expect:
        bias.setdefault(question, {})[str(option)] = 3.2
    return bias


# --- asking blink -----------------------------------------------------------------------


def _recording(eng):
    if getattr(eng, "name", "") == "replay":
        return eng
    return getattr(eng, "replay", None)


def _answers(raw: dict, qs: dict, temperature: float) -> dict:
    return {qkey: blink.answer_for(q, [k for k, _ in blink.question_options(q)],
                                   blink.softmax([float(x) for x in raw[qkey]], temperature))
            for qkey, q in qs.items()}


def forward(live, st: dict, qs: dict) -> tuple[dict, int, dict]:
    """Every question about one screenshot in a single GPU call: blink's own image rendering and
    forward, handed all three questions at once instead of one call each (on ZeroGPU each call
    attaches a device). Each question is still its own forward inside the call, so the numbers
    are those blink.decide gives. -> (label logits by question, prompt tokens, image facts)."""
    lifted = blink.extract_images(st, limits=getattr(live, "image_limits", None))
    work = live.render_images(lifted, qs)
    rows, model_ms, _encoder_ms, _prefill_ms = blink._forward_images(live.key, work)
    raw = {item["qkey"]: row for item, row in zip(work, rows)}
    tokens = sum(int(item["inputs"]["input_ids"].shape[1]) for item in work)
    return raw, tokens, {"model_ms": model_ms, "image_pixels": int(work[0]["image_pixels"]),
                         "visual_tokens": int(work[0]["visual_tokens"])}


def _live(eng, st: dict, qs: dict) -> dict:
    live = getattr(eng, "live", eng)
    temperature = float(eng.temperature)
    t0 = time.perf_counter()
    raw, tokens, facts = forward(live, st, qs)
    return {"answers": _answers(raw, qs, temperature),
            "meta": {"model": getattr(eng, "model_id", None), "engine": "torch", "temperature": temperature,
                     "input_tokens": tokens, "generated_tokens": 0,
                     "latency_ms": round((time.perf_counter() - t0) * 1000, 1), **facts}}


def _from_recording(rec, hit: dict, qs: dict, temperature: float) -> dict:
    answers = _answers(hit["logits"], qs, temperature)
    meta = {"model": rec.model_id, "engine": "replay", "temperature": temperature,
            "input_tokens": int(hit["input_tokens"]), "generated_tokens": 0,
            "latency_ms": float(hit["latency_ms"])}
    for extra in ("image_pixels", "visual_tokens"):
        if extra in hit:
            meta[extra] = int(hit[extra])
    if getattr(rec, "hardware", ""):
        meta["recorded_on"] = rec.hardware
    return {"answers": answers, "meta": meta}


def _mock(shot: Shot, st: dict, qs: dict, model: str) -> dict:
    """Fabricated numbers on the real request: the image is checked and decoded exactly as the
    live path does, then a text-only stand-in answers with the image as its placeholder."""
    lifted = blink.extract_images(st)
    out = blink.decide(lifted.state, qs, bias=mock_bias(shot), model=model)
    pixels = sum(img.width * img.height for img in lifted.images)
    out["meta"]["image_pixels"] = pixels
    out["meta"]["visual_tokens"] = pixels // 1024
    return out


def decide(shot: Shot, model: str | None = None, prefer: str = "live") -> dict:
    """{answers, meta} for this screenshot, from blink-mimo-9b.

    prefer="saved" is the first render: a recording when there is one, never a live run on a
    host that attaches its device per call. Replay-only hosting answers from recordings alone."""
    model = model or vision_model()
    if model is None:
        raise ScreenError("no served model reads screenshots")
    st, qs = request(shot)
    eng = blink.engine(model)
    kind = getattr(eng, "name", "")
    rec = _recording(eng)
    if kind == "mock":
        return _mock(shot, st, qs, model)
    if rec is not None and (prefer == "saved" or kind == "replay"):
        hit = rec.cache.get(blink.request_key(st, qs))
        if hit is not None:
            return _from_recording(rec, hit, qs, float(eng.temperature))
    if kind == "replay" or (prefer == "saved" and kind == "hybrid"):
        raise ScreenMiss("No saved run for this screenshot.")
    return _live(eng, st, qs)


def accepts_images(model: str | None = None) -> bool:
    """Whether a live run can read a screenshot here (mock mode always can)."""
    model = model or vision_model()
    if model is None:
        return False
    eng = blink.engine(model)
    if getattr(eng, "name", "") in ("mock", "replay"):
        return True
    return bool(getattr(getattr(eng, "live", eng), "accepts_images", False))


def rule_source() -> str:
    """The decision rule as the card shows it: the cut-offs, then verdict itself."""
    cuts = "".join(f"{name} = {value:g}\n" for name, value in
                   (("DONE_AT", DONE_AT), ("RISKY_AT", RISKY_AT), ("SURE_AT", SURE_AT)))
    return cuts + "\n" + inspect.getsource(verdict).rstrip()


def verdict(a: dict) -> tuple:
    done = a["done"]["noul"]
    risky = a["risky"]["noul"]
    element = a["element"]
    top = max(element["probabilities"].values())
    if done >= DONE_AT:
        return "go", "done", "Task looks done", f"Nothing left to click ({done:.0%} yes)."
    if risky >= RISKY_AT:
        return "hold", "ask", f"Ask before clicking {element['choice']}", f"Risky click ({risky:.0%} yes)."
    if top < SURE_AT:
        return "hold", "unsure", "Let a person take over", f"Top mark {top:.0%}."
    return "go", "click", f"Click {element['choice']}", f"Top mark {top:.0%}; not done yet ({done:.0%} yes)."
