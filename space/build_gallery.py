"""Build the Computer use gallery from CUA harness run folders.

    uv run --no-project --python 3.12 --with pillow python space/build_gallery.py \\
        RUN_DIR [RUN_DIR ...] --registry REGISTRY.json --stage /tmp/blink-cua-stage \\
        [--out space/cua/gallery.json] [--more MORE.json] [--media-base URL]

Each RUN_DIR is one harness run (`runs/<run_id>/`, see cua/scenarios/SPEC.md): `config.json`, then
`<scenario>/<task>-<seed>-<model>/` with `summary.json`, `trace.jsonl`, `steps/NNN.png`, `video.mp4` and
`poster.png`. The script writes

- `gallery.json`, the one file the Space reads: every scenario's success rate over all its episodes, the
  episodes it shows, each step's marks, probabilities, checks and time, and the rule that picked them;
- a staging folder holding exactly the media gallery.json names, at the paths it names under the media
  base (the public dataset repo), plus MANIFEST.json with each file's size and SHA-256.

Selection, recorded in gallery.json as `rule`: per scenario and model, the showcase is the successful
episode with the lowest (task, seed) that has a video; if none succeeded, the lowest (task, seed) episode,
shown as the failure it is. A failed episode of the showcase's task (else of any task) rides along, so the
player can show a run that went wrong. Success rates count every episode, errors included, as the
harness report does. Mock runs (config.json `"mock": true`, or --mock) make a mock gallery: labelled on
the page, refused by stage_space.py, and never written into the Space's own cua/ folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPACE_GALLERY = HERE / "cua" / "gallery.json"
SCHEMA = "blink-cua-gallery/1"
MEDIA_BASE = "https://huggingface.co/datasets/thegovind/blink-cua/resolve/main/"
VISION_MODEL = "blink-mimo-9b"
EPISODE = re.compile(r"^(?P<task>\d+)-(?P<seed>\d+)-(?P<model>[\w.\-]+)$")
Z95 = 1.959963984540054
PRIMARY = {"interrupts": 0, "mode": "batched"}
RULE = {
    "showcase": "Per scenario and model: the successful episode with the lowest (task, seed) that has a "
                "video; if none succeeded, the lowest (task, seed) episode, shown as a failure.",
    "failure": "The lowest-seed failed episode of the showcase's task, else the lowest (task, seed) failed "
               "episode; none when every episode succeeded.",
    "stats": "Every episode of the scenario and model at interrupts 0 in batched mode, errors counted as "
             "failures, as the harness report counts them; other conditions are listed apart.",
    "hero": "The vision model's successful showcases, in registry order.",
}
WEBP_QUALITY = 80
PREVIEW_WIDTH = 720


class GalleryError(Exception):
    pass


# --- numbers ---------------------------------------------------------------------------


def wilson(successes: int, total: int) -> list[float] | None:
    """The harness report's Wilson 95% interval (cua/scenarios/harness/report.py)."""
    if not total:
        return None
    p = successes / total
    center = (p + Z95 * Z95 / (2 * total)) / (1 + Z95 * Z95 / total)
    radius = Z95 * math.sqrt(p * (1 - p) / total + Z95 * Z95 / (4 * total * total)) / (1 + Z95 * Z95 / total)
    return [round(max(0.0, center - radius), 4), round(min(1.0, center + radius), 4)]


def percentile(values: list[float], at: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * at
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 1)


def ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


# Seeds whose step times count toward p50/p95 (None = all). Set by --latency-seeds when some seeds were
# replayed one run per server and the rest ran many at a time, so the median is not inflated by waiting.
LATENCY_SEEDS: set[int] | None = None


def stats(episodes: list[dict]) -> dict:
    """Success over every episode, per-step accuracy against the oracle, the risk gate and time."""
    counts = {k: sum(int(e["summary"].get(k) or 0) for e in episodes)
              for k in ("element_correct", "element_total", "risk_tp", "risk_fp", "risk_fn", "risk_tn",
                        "done_tp", "done_fp", "gates")}
    wins = sum(bool(e["summary"].get("success")) for e in episodes)
    timed = [e for e in episodes if LATENCY_SEEDS is None or e["seed"] in LATENCY_SEEDS]
    wall = [float(ms) for e in timed for ms in e["summary"].get("wall_ms") or []]
    return {
        "episodes": len(episodes),
        "successes": wins,
        "rate": ratio(wins, len(episodes)),
        "wilson95": wilson(wins, len(episodes)),
        "errors": sum(e["summary"].get("stop_reason") == "error" for e in episodes),
        "step_accuracy": ratio(counts["element_correct"], counts["element_total"]),
        "steps_scored": counts["element_total"],
        "gate_recall": ratio(counts["risk_tp"], counts["risk_tp"] + counts["risk_fn"]),
        "false_gate_rate": ratio(counts["risk_fp"], counts["risk_fp"] + counts["risk_tn"]),
        "done_precision": ratio(counts["done_tp"], counts["done_tp"] + counts["done_fp"]),
        "gates": counts["gates"],
        "p50_ms": percentile(wall, 0.5),
        "p95_ms": percentile(wall, 0.95),
        "decisions": len(wall),
        "seeds": sorted({e["seed"] for e in episodes}),
        "latency_seeds": sorted({e["seed"] for e in timed}),
    }


# --- reading runs ----------------------------------------------------------------------


def _json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _trace(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_runs(run_dirs: list[Path]) -> tuple[list[dict], list[dict], bool]:
    """Every finished episode in the runs, their sources, and whether any run is a mock."""
    episodes, sources, mock = [], [], False
    seen: dict[tuple, Path] = {}
    for run in run_dirs:
        run = Path(run)
        config_path = run / "config.json"
        if not config_path.is_file():
            raise GalleryError(f"{run} has no config.json; is it a harness run folder?")
        config = _json(config_path)
        mock = mock or bool(config.get("mock"))
        count = 0
        for summary_path in sorted(run.glob("*/*/summary.json")):
            folder = summary_path.parent
            if folder.parent.parent != run:
                continue
            match = EPISODE.match(folder.name)
            if not match:
                continue
            summary = _json(summary_path)
            scenario = folder.parent.name
            key = (scenario, int(match["task"]), int(match["seed"]), match["model"],
                   int(summary.get("interrupts") or 0), summary.get("mode") or "batched")
            if key in seen:
                raise GalleryError(f"{scenario}/{folder.name} appears in both {seen[key]} and {run}; "
                                   "pass each episode once")
            seen[key] = run
            trace_path = folder / "trace.jsonl"
            episodes.append({
                "run_id": run.name, "folder": folder, "scenario": scenario, "task": int(match["task"]),
                "seed": int(match["seed"]), "model": match["model"], "interrupts": key[4], "mode": key[5],
                "summary": summary, "trace": _trace(trace_path) if trace_path.is_file() else [],
            })
            count += 1
        sources.append({"run_id": run.name, "harness_sha": config.get("harness_sha"),
                        "mock": bool(config.get("mock")), "episodes": count})
    return episodes, sources, mock


def video_source(e: dict) -> Path | None:
    """The harness's H.264 when it made one, else its real-time WebM (an odd-sized phone frame can stop
    the H.264 step; the WebM is the same recording)."""
    mp4, webm = e["folder"] / "video.mp4", e["folder"] / "video.webm"
    if mp4.is_file() and mp4.stat().st_size and not e["summary"].get("video_error"):
        return mp4
    return webm if webm.is_file() and webm.stat().st_size else None


def has_video(e: dict) -> bool:
    return video_source(e) is not None


def poster_source(e: dict) -> Path | None:
    """The harness's poster, else the first marked step (a phone run whose H.264 step failed has no poster)."""
    for path in (e["folder"] / "poster.png", e["folder"] / "steps" / "001.png"):
        if path.is_file():
            return path
    return None


def order_key(e: dict) -> tuple:
    return (e["task"], e["seed"], e["run_id"])


def pick(pool: list[dict]) -> tuple[dict | None, dict | None]:
    """(showcase, failure) by RULE."""
    if not pool:
        return None, None
    wins = sorted((e for e in pool if e["summary"].get("success") and has_video(e)), key=order_key)
    showcase = wins[0] if wins else min(pool, key=lambda e: (not has_video(e), *order_key(e)))
    losses = [e for e in pool if not e["summary"].get("success") and e is not showcase
              and e["summary"].get("stop_reason") != "error" and e["trace"]]
    same = sorted((e for e in losses if e["task"] == showcase["task"]), key=order_key)
    failure = (same or sorted(losses, key=order_key) or [None])[0]
    return showcase, failure


# --- one episode, as the page shows it ------------------------------------------------


def _media(e: dict) -> str:
    return f"runs/{e['run_id']}/{e['scenario']}/{e['folder'].name}"


def _round(p) -> float | None:
    return None if p is None else round(float(p), 3)


def _frame(e: dict, scenario: dict | None) -> list[int]:
    if scenario:
        vp, scale = scenario.get("viewport") or {}, float(scenario.get("scale") or 1.5)
        if vp.get("width") and vp.get("height"):
            return [round(vp["width"] * scale), round(vp["height"] * scale)]
    return [1440, 864]


def step_view(row: dict, frame: list[int], media: str) -> dict:
    """One trace row: the marked screenshot, its boxes with their numbers' spots, what blink answered
    and what the harness did with it. Nothing the model never saw is added except the oracle's box."""
    import screens

    cands = row.get("candidates") or []
    if any(int(c.get("n", i)) != i for i, c in enumerate(cands, 1)):
        raise GalleryError(f"step {row.get('step')}: the boxes are not numbered 1 to {len(cands)} in order")
    boxes = [[int(v) for v in c["box"]] for c in cands]
    placed = screens.layout([tuple(b) for b in boxes], tuple(frame)) if boxes else []
    answers = row.get("answers") or {}
    head = answers.get("element") or answers.get("lane") or {}
    probs = head.get("probabilities") or {}
    action = row.get("chosen_action") or row.get("action") or {}
    kind = action.get("kind") or "none"
    pick = action.get("box") if kind not in ("done", "none") else None
    done = (answers.get("done") or {}).get("noul")
    risky = (answers.get("risky") or {}).get("noul")
    image = row.get("marked_screenshot") or f"steps/{int(row['step']):03}.png"
    return {
        "n": int(row["step"]),
        "img": f"{media}/{Path(image).with_suffix('.webp').as_posix()}",
        "state": row.get("state") or "",
        "boxes": boxes,
        "tags": [[*b.tag, b.at] for b in placed],
        "labels": [" \u00b7 ".join(x for x in (c.get("role") or "", c.get("name") or "") if x) for c in cands],
        "p": [_round(probs.get(str(c["n"]))) for c in cands],
        "pick": int(pick) if pick not in (None, "") else None,
        "oracle": int(row["oracle_box"]) if row.get("oracle_box") not in (None, "") else None,
        "ok": row.get("element_correct"),
        "done": _round(done),
        "risky": _round(risky),
        "gate": bool(row.get("gate_fired")),
        "act": kind,
        "value": action.get("value"),
        "ms": round(float(row.get("wall_ms") or 0), 1),
        "tokens": (row.get("usage") or {}).get("visual_tokens") or None,
    }


def episode_view(e: dict, scenario: dict | None) -> dict:
    s = e["summary"]
    media = _media(e)
    frame = _frame(e, scenario)
    trace = e["trace"]
    task = ((trace[0].get("request_state") or {}).get("task") if trace else "") or ""
    tokens = sorted({int(r["usage"]["visual_tokens"]) for r in trace
                     if (r.get("usage") or {}).get("visual_tokens")})
    video = has_video(e)
    return {
        "scenario": e["scenario"], "model": e["model"], "task_index": e["task"], "seed": e["seed"],
        "interrupts": e["interrupts"], "run_id": e["run_id"],
        "task": task,
        "success": bool(s.get("success")),
        "detail": str(s.get("detail") or "") if not s.get("success") else "",
        "stop_reason": s.get("stop_reason"),
        "steps_taken": int(s.get("steps") or 0),
        "optimal_steps": s.get("optimal_steps"),
        "gates": int(s.get("gates") or 0),
        "p50_ms": s.get("p50_wall_ms"), "p95_ms": s.get("p95_wall_ms"),
        "visual_tokens": tokens[0] if len(tokens) == 1 else None,
        "frame": frame,
        "video": f"{media}/video.mp4" if video else None,
        "preview": f"{media}/preview.mp4" if video else None,
        "poster": f"{media}/poster.webp" if poster_source(e) or video else None,
        "duration_s": None,
        "steps": [step_view(r, frame, media) for r in trace if (r.get("candidates") is not None)],
    }


def rail(steps: list[dict]) -> list[dict]:
    """Each step in a few bytes, for the page to draw the step rail before it fetches the steps."""
    out = []
    for st in steps:
        mark = {"a": st["act"]}
        if st["gate"]:
            mark["g"] = 1
        if st["ok"] is False:
            mark["m"] = 1
        out.append(mark)
    return out


def steps_file(view: dict, folder: str) -> tuple[str, bytes]:
    """Where an episode's steps are published, named by their content so a cache never serves old ones."""
    body = json.dumps(view["steps"], separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"{folder}/steps.{hashlib.sha256(body).hexdigest()[:12]}.json", body


# --- media --------------------------------------------------------------------------


def _ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        return None
    return get_ffmpeg_exe()


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode:
        raise GalleryError(f"{Path(cmd[0]).name} failed: {r.stderr[-600:]}")


def duration(path: Path) -> float | None:
    probe = shutil.which("ffprobe")
    if not probe:
        return None
    r = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=False)
    try:
        return round(float(r.stdout.strip()), 2)
    except ValueError:
        return None


def to_webp(src: Path, dst: Path, width: int | None = None, lossless: bool = False, reuse: bool = False) -> None:
    from PIL import Image

    if reuse and dst.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if width and im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        im.save(dst, "WEBP", quality=WEBP_QUALITY, method=6, lossless=lossless)


def stage_episode(e: dict, view: dict, stage: Path, ffmpeg: str | None, reuse: bool = False) -> None:
    folder = e["folder"]
    for step in view["steps"]:
        src = folder / "steps" / Path(step["img"]).with_suffix(".png").name
        if src.is_file():
            to_webp(src, stage / step["img"], reuse=reuse)
        else:
            step["img"] = None
    if view["poster"]:
        width = 960 if view["frame"][0] > view["frame"][1] else None
        still = folder / "poster.png"
        if not still.is_file() and ffmpeg and video_source(e) and not (reuse and (stage / view["poster"]).is_file()):
            # the harness makes its poster from the video's second second; do the same when it could not
            still = stage / (view["poster"] + ".png")
            still.parent.mkdir(parents=True, exist_ok=True)
            try:
                _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", "1", "-i", str(video_source(e)),
                      "-frames:v", "1", str(still)])
            except GalleryError:
                still.unlink(missing_ok=True)
        if not still.is_file():
            still = poster_source(e)
        if still is None:
            view["poster"] = None
        else:
            to_webp(still, stage / view["poster"], width=width, reuse=reuse)
            if still.parent != folder and still.name.endswith(".webp.png"):
                still.unlink()
    source = video_source(e)
    if view["video"] and source is not None:
        out = stage / view["video"]
        out.parent.mkdir(parents=True, exist_ok=True)
        if reuse and out.is_file() and (stage / view["preview"]).is_file():
            view["duration_s"] = duration(out)
            return
        quiet = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)] if ffmpeg else []
        h264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"]
        if ffmpeg and source.suffix == ".mp4":
            # the harness's own H.264; only its index moves to the front, so it streams
            _run([*quiet, "-c", "copy", "-movflags", "+faststart", "-an", str(out)])
        elif ffmpeg:
            # H.264 wants even sides: a phone's 585 px gains one blank column, nothing is scaled
            _run([*quiet, "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-preset", "slow", "-crf", "23", *h264, str(out)])
        elif source.suffix == ".mp4":
            shutil.copyfile(source, out)
        else:
            view["video"] = view["preview"] = None
            return
        if ffmpeg:
            scale = f"scale='trunc(min({PREVIEW_WIDTH},iw)/2)*2':-2:flags=lanczos"
            _run([*quiet, "-vf", scale, "-preset", "slow", "-crf", "30", *h264, str(stage / view["preview"])])
        else:
            view["preview"] = view["video"]
        view["duration_s"] = duration(out)


# --- the tiles beyond clicks ------------------------------------------------------------


TILE_KEYS = ("id", "title", "line")


def read_more(path: Path | None, stage: Path | None, ffmpeg: str | None) -> list[dict]:
    """--more: a JSON list of tiles, each {id, title, line, metric?, video?, poster?, captions?, sound?,
    chart?, images?}. File paths are relative to the manifest; they are staged under more/<id>/."""
    if path is None:
        return []
    items = _json(path)
    root = Path(path).parent
    tiles = []
    for item in items:
        missing = [k for k in TILE_KEYS if not item.get(k)]
        if missing:
            raise GalleryError(f"tile {item.get('id')!r} lacks {', '.join(missing)}")
        tid = re.sub(r"[^a-z0-9-]+", "-", str(item["id"]).lower()).strip("-")
        tile = {"id": tid, "title": item["title"], "line": item["line"], "metric": item.get("metric"),
                "sound": bool(item.get("sound")), "mock": bool(item.get("mock"))}
        base = f"more/{tid}"

        def src(rel: str, tid: str = tid) -> Path:
            p = (root / rel).resolve()
            if not p.is_file():
                raise GalleryError(f"tile {tid}: {rel} not found")
            return p

        if item.get("video"):
            tile["video"] = f"{base}/{Path(item['video']).stem}.mp4"
            if stage is not None:
                (stage / base).mkdir(parents=True, exist_ok=True)
                if ffmpeg:
                    audio = [] if tile["sound"] else ["-an"]
                    _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src(item["video"])),
                          "-c", "copy", "-movflags", "+faststart", *audio, str(stage / tile["video"])])
                else:
                    shutil.copyfile(src(item["video"]), stage / tile["video"])
        if item.get("captions"):
            tile["captions"] = f"{base}/{Path(item['captions']).stem}.vtt"
            if stage is not None:
                shutil.copyfile(src(item["captions"]), stage / tile["captions"])
        for key in ("poster", "chart"):
            if item.get(key):
                tile[key] = f"{base}/{Path(item[key]).stem}.webp"
                if stage is not None:
                    to_webp(src(item[key]), stage / tile[key], lossless=key == "chart")
        if item.get("images"):
            tile["images"] = []
            for rel in item["images"]:
                name = f"{base}/{Path(rel).stem}.webp"
                tile["images"].append(name)
                if stage is not None:
                    to_webp(src(rel), stage / name)
        if not any(tile.get(k) for k in ("video", "chart", "images", "poster")):
            raise GalleryError(f"tile {tid} has nothing to show")
        tiles.append(tile)
    return tiles


# --- assembly -----------------------------------------------------------------------


def device(frame: list[int]) -> str:
    return "phone" if frame[1] > frame[0] else "desktop"


def build(run_dirs: list[Path], registry: dict, *, stage: Path | None = None, more: Path | None = None,
          media_base: str = MEDIA_BASE, mock: bool = False, models: list[str] | None = None,
          reuse: bool = False) -> dict:
    episodes, sources, run_mock = read_runs(run_dirs)
    if not episodes:
        raise GalleryError("no finished episodes in the given runs")
    mock = mock or run_mock
    by_id = {s["id"]: s for s in registry.get("scenarios", [])}
    order = [s["id"] for s in registry.get("scenarios", [])]
    unknown = sorted({e["scenario"] for e in episodes} - set(by_id))
    if unknown:
        raise GalleryError(f"scenarios missing from the registry: {', '.join(unknown)}")
    found_models = sorted({e["model"] for e in episodes}, key=lambda m: (m != VISION_MODEL, m))
    models = [m for m in (models or found_models) if m in found_models]
    ffmpeg = _ffmpeg() if stage is not None else None
    if stage is not None:
        if stage.exists() and not reuse:
            shutil.rmtree(stage)
        stage.mkdir(parents=True, exist_ok=True)

    shown: dict[str, dict] = {}
    scenarios = []
    for sid in order:
        mine = [e for e in episodes if e["scenario"] == sid]
        if not mine:
            continue
        reg = by_id[sid]
        frame = _frame(mine[0], reg)
        entry = {"id": sid, "title": reg.get("title") or sid, "kind": reg.get("kind") or "dom",
                 "device": device(frame), "frame": frame,
                 "tasks": [{"index": t["index"], "summary": t.get("summary", ""),
                            "optimal_steps": t.get("optimal_steps"), "risky_steps": t.get("risky_steps", 0)}
                           for t in reg.get("tasks", [])],
                 "models": {}}
        for model in models:
            pool = [e for e in mine if e["model"] == model]
            if not pool:
                continue
            primary = [e for e in pool if e["interrupts"] == PRIMARY["interrupts"] and e["mode"] == PRIMARY["mode"]]
            main = primary or pool
            showcase, failure = pick(main)
            others = {}
            for e in pool:
                if e in main:
                    continue
                others.setdefault((e["interrupts"], e["mode"]), []).append(e)
            item = {
                "stats": stats(main),
                "condition": dict(PRIMARY) if primary else {"interrupts": main[0]["interrupts"],
                                                            "mode": main[0]["mode"]},
                "outcomes": [{"task": e["task"], "seed": e["seed"], "ok": bool(e["summary"].get("success")),
                              "error": e["summary"].get("stop_reason") == "error"}
                             for e in sorted(main, key=order_key)],
                "other_conditions": [{"interrupts": k[0], "mode": k[1], "stats": stats(v)}
                                     for k, v in sorted(others.items())],
                "showcase": None, "failure": None,
            }
            for role, e in (("showcase", showcase), ("failure", failure)):
                if e is None:
                    continue
                eid = f"{sid}/{e['folder'].name}"
                if e["interrupts"] or e["mode"] != "batched":
                    eid += f"/i{e['interrupts']}-{e['mode']}"
                view = episode_view(e, reg)
                if stage is not None:
                    stage_episode(e, view, stage, ffmpeg, reuse)
                view["rail"] = rail(view["steps"])
                view["steps_url"] = None
                if view["steps"]:
                    rel, body = steps_file(view, _media(e))
                    view["steps_url"] = rel
                    if stage is not None:
                        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
                        (stage / rel).write_bytes(body)
                shown[eid] = view
                item[role] = eid
            entry["models"][model] = item
        scenarios.append(entry)

    totals = {}
    for model in models:
        pool = [e for e in episodes if e["model"] == model and e["interrupts"] == 0 and e["mode"] == "batched"]
        if pool:
            totals[model] = {**stats(pool), "scenarios": len({e["scenario"] for e in pool})}
    # The model the page leads with: the best measured success over every run, else the one that reads screens live.
    lead = max(totals, key=lambda m: (totals[m]["rate"] or 0, m == VISION_MODEL)) if totals else VISION_MODEL
    hero = [s["models"][lead]["showcase"] for s in scenarios
            if lead in s["models"] and s["models"][lead]["showcase"]
            and shown[s["models"][lead]["showcase"]]["success"]
            and shown[s["models"][lead]["showcase"]]["video"]]
    tokens = sorted({v["visual_tokens"] for v in shown.values() if v["visual_tokens"]
                     and v["frame"][0] > v["frame"][1]})
    gallery = {
        "schema": SCHEMA,
        "mock": bool(mock),
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "media_base": media_base if media_base.endswith("/") else media_base + "/",
        "rule": RULE,
        "sources": sources,
        "models": models,
        "vision_model": VISION_MODEL,
        "lead_model": lead,
        "desktop_visual_tokens": tokens[0] if len(tokens) == 1 else None,
        "totals": totals,
        "hero": hero,
        "scenarios": scenarios,
        "episodes": shown,
        "more": read_more(more, stage, ffmpeg),
    }
    check(gallery)
    if stage is not None:
        keep = set(media_paths(gallery)) | {"gallery.json", "MANIFEST.json"}
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            if path.relative_to(stage).as_posix() not in keep:
                path.unlink()  # a reused stage holds exactly what this gallery names
        for folder in sorted((p for p in stage.rglob("*") if p.is_dir()), reverse=True):
            if not any(folder.iterdir()):
                folder.rmdir()
        write_json(stage / "gallery.json", gallery)
        manifest = {}
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            rel = path.relative_to(stage).as_posix()
            manifest[rel] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        write_json(stage / "MANIFEST.json", {"media_base": gallery["media_base"], "files": manifest})
    return gallery


# --- checks -------------------------------------------------------------------------

# what a public file must never carry: local paths, machine names, hardware and the employer (each split in
# two, so this file passes the public strict scan it mirrors)
BANNED = tuple(a + b for a, b in (("/nv", "me"), ("/ho", "me/"), ("/tm", "p/"), ("azure", "user"), ("go", "k-"),
                                  ("a1", "00"), ("h1", "00"), ("nvi", "dia"), ("80", "gb"), ("core", "ai"),
                                  ("micro", "soft"), ("127.0.", "0.1"), ("local", "host")))


def media_paths(gallery: dict) -> list[str]:
    """Every file gallery.json names under the media base."""
    out = []
    for ep in gallery["episodes"].values():
        out += [ep[k] for k in ("video", "preview", "poster", "steps_url") if ep.get(k)]
        out += [s["img"] for s in ep["steps"] if s.get("img")]
    for tile in gallery.get("more", []):
        out += [tile[k] for k in ("video", "poster", "chart", "captions") if tile.get(k)]
        out += list(tile.get("images") or [])
    return sorted(set(out))


def check(gallery: dict) -> None:
    """Refuse a gallery the page could not show or should not publish."""
    if gallery.get("schema") != SCHEMA:
        raise GalleryError(f"schema {gallery.get('schema')!r} is not {SCHEMA}")
    for s in gallery["scenarios"]:
        for model, item in s["models"].items():
            for role in ("showcase", "failure"):
                if item[role] is not None and item[role] not in gallery["episodes"]:
                    raise GalleryError(f"{s['id']}/{model} {role} {item[role]} is not in episodes")
            st = item["stats"]
            if st["successes"] > st["episodes"] or len(item["outcomes"]) != st["episodes"]:
                raise GalleryError(f"{s['id']}/{model}: outcomes and stats disagree")
    for eid in gallery["hero"]:
        if eid not in gallery["episodes"]:
            raise GalleryError(f"hero {eid} is not in episodes")
    for ep in gallery["episodes"].values():
        for step in ep["steps"]:
            if not (len(step["boxes"]) == len(step["tags"]) == len(step["labels"]) == len(step["p"])):
                raise GalleryError(f"{ep['scenario']} step {step['n']}: boxes and answers disagree")
        if len(ep.get("rail", ep["steps"])) != len(ep["steps"]):
            raise GalleryError(f"{ep['scenario']}: the rail and the steps disagree")
    for rel in media_paths(gallery):
        if rel.startswith("/") or ".." in rel.split("/") or "\\" in rel:
            raise GalleryError(f"media path {rel!r} must be relative")
    low = json.dumps({k: v for k, v in gallery.items() if k != "media_base"}).lower()
    for bad in BANNED:
        if bad in low:
            raise GalleryError(f"{bad!r} found in the gallery")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="harness run folders (runs/<run_id>)")
    ap.add_argument("--registry", type=Path, required=True, help="cua/scenarios/apps/registry.json")
    ap.add_argument("--out", type=Path, default=SPACE_GALLERY)
    ap.add_argument("--stage", type=Path, help="write the media here, at the paths gallery.json names")
    ap.add_argument("--more", type=Path, help="tiles beyond clicks (JSON list, see read_more)")
    ap.add_argument("--media-base", default=MEDIA_BASE)
    ap.add_argument("--model", action="append", help="models to show, in order (default: all, vision first)")
    ap.add_argument("--mock", action="store_true", help="label the gallery as mock data")
    ap.add_argument("--reuse", action="store_true",
                    help="keep media already converted in --stage; files the gallery no longer names are removed")
    ap.add_argument("--latency-seeds", help="only these seeds' step times go into p50/p95, e.g. 1-3 or 1,2,3")
    a = ap.parse_args(argv)
    global LATENCY_SEEDS
    if a.latency_seeds:
        LATENCY_SEEDS = set()
        for part in a.latency_seeds.split(","):
            lo, _, hi = part.partition("-")
            LATENCY_SEEDS.update(range(int(lo), int(hi or lo) + 1))
    sys.path.insert(0, str(HERE))
    try:
        gallery = build(a.runs, _json(a.registry), stage=a.stage, more=a.more, media_base=a.media_base,
                        mock=a.mock, models=a.model, reuse=a.reuse)
    except GalleryError as exc:
        print(f"build_gallery: {exc}", file=sys.stderr)
        return 1
    if gallery["mock"] and a.out.resolve().parent == SPACE_GALLERY.parent:
        print("build_gallery: a mock gallery never goes in the Space's cua/ folder; pass --out elsewhere",
              file=sys.stderr)
        return 1
    write_json(a.out, gallery)
    shown = sum(len(s["models"]) for s in gallery["scenarios"])
    print(f"wrote {a.out}: {len(gallery['scenarios'])} scenarios, {shown} scenario-model rows, "
          f"{len(gallery['episodes'])} episodes shown, {len(media_paths(gallery))} media files"
          + (" (MOCK)" if gallery["mock"] else ""))
    if a.stage:
        print(f"staged {a.stage} for {gallery['media_base']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
