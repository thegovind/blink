"""Burn the Space's original marks on the exact screenshot sent to blink."""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# screens.py imports its sibling blink.py, as it does in space/build_screens.py.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "space") not in sys.path:
    sys.path.insert(0, str(_ROOT / "space"))
from space import screens  # noqa: E402

from .dom import Candidate  # noqa: E402

PAD_CSS = 6


@dataclass(frozen=True)
class Marked:
    png: bytes
    boxes: tuple[screens.Box, ...]
    size: tuple[int, int]

    def shot(self, task: str) -> screens.Shot:
        return screens.Shot(key="scenario", label="", task=task, png=self.png,
                            size=self.size, boxes=self.boxes)

    def trace_candidates(self, candidates: list[Candidate]) -> list[dict]:
        return [{"n": candidate.n, "box": list(box.box), "role": candidate.role,
                 "name": candidate.name, "selector": candidate.selector, "target": candidate.target}
                for candidate, box in zip(candidates, self.boxes)]

    def overlay_marks(self, scale: float) -> list[dict]:
        return [{"n": box.n, "box": [v / scale for v in box.box],
                 "tag": [v / scale for v in box.tag]} for box in self.boxes]


def mark_screenshot(png: bytes, candidates: list[Candidate], *,
                    scale: float, viewport: dict[str, int]) -> Marked:
    with Image.open(io.BytesIO(png)) as opened:
        image = opened.convert("RGB")
    size = (round(viewport["width"] * scale), round(viewport["height"] * scale))
    if image.size != size:
        raise ValueError(f"Screenshot is {image.size}, expected {size} at scale {scale}")
    rectangles = []
    for candidate in candidates:
        r = candidate.rect
        x1 = max(0, r["x"] - PAD_CSS)
        y1 = max(0, r["y"] - PAD_CSS)
        x2 = min(viewport["width"], r["x"] + r["w"] + PAD_CSS)
        y2 = min(viewport["height"], r["y"] + r["h"] + PAD_CSS)
        rectangles.append(tuple(round(n * scale) for n in (x1, y1, x2, y2)))
    marked, boxes = screens.mark(image, rectangles,
                                 meta=[{"role": c.role, "name": c.name} for c in candidates])
    output = io.BytesIO()
    marked.save(output, format="PNG", optimize=True)
    return Marked(png=output.getvalue(), boxes=tuple(boxes), size=size)
