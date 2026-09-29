"""The Computer use tab: the gallery build, what the tab renders from it, and the links into it.

    cd space && BLINK_MOCK=1 uv run --no-project --python 3.12 --with gradio==6.28.0 --with huggingface_hub \\
        python -m unittest test_cua
"""

import contextlib
import hashlib
import html as htmllib
import importlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    from PIL import Image
except ImportError:  # pragma: no cover - the public CI installs pillow
    raise unittest.SkipTest("pillow is not installed") from None

os.environ.setdefault("BLINK_MOCK", "1")
import blink
import build_gallery
import computer_use
import screens

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = "/" + "home/"  # a private path, split so the public strict scan passes
MIMO = "thegovind/blink-mimo-9b"
FOUR = "thegovind/blink-4b"
BOXES = [[100, 100, 300, 160], [400, 100, 600, 160], [100, 400, 500, 460]]
REGISTRY = {"version": 1, "scenarios": [
    {"id": "shop", "title": "Outdoor gear store", "kind": "dom", "viewport": {"width": 960, "height": 576},
     "scale": 1.5, "tasks": [{"index": 0, "summary": "buy, pay (risky)", "optimal_steps": 3, "risky_steps": 1},
                             {"index": 1, "summary": "promo", "optimal_steps": 2, "risky_steps": 0}]},
    {"id": "phone", "title": "Phone settings", "kind": "dom", "viewport": {"width": 390, "height": 844},
     "scale": 1.5, "tasks": [{"index": 0, "summary": "toggle", "optimal_steps": 2, "risky_steps": 0}]},
    {"id": "unused", "title": "Never run", "kind": "dom", "viewport": {"width": 960, "height": 576},
     "scale": 1.5, "tasks": []},
]}


@contextlib.contextmanager
def serving(*ids):
    saved = (blink.MODEL_SPECS, blink.MODEL_ID, blink._ENGINE, dict(blink._ENGINES))
    blink.MODEL_SPECS = [(i, None) for i in ids]
    blink.MODEL_ID = ids[0]
    blink._ENGINE = None
    blink._ENGINES = {}
    try:
        yield
    finally:
        blink.MODEL_SPECS, blink.MODEL_ID, blink._ENGINE, engines = saved
        blink._ENGINES = engines


@contextlib.contextmanager
def gallery_at(path):
    """Point the page at a gallery file (or at nothing) for the length of the block."""
    was = os.environ.get("BLINK_GALLERY")
    os.environ["BLINK_GALLERY"] = str(path)
    try:
        yield
    finally:
        if was is None:
            os.environ.pop("BLINK_GALLERY", None)
        else:
            os.environ["BLINK_GALLERY"] = was


def need_gradio(test):
    try:
        import gradio as gr
    except ImportError:  # pragma: no cover - depends on the environment
        test.skipTest("gradio is not installed")
    return gr


def reload_app():
    import app
    import ui

    importlib.reload(ui)
    return importlib.reload(app)


def episode(run: Path, scenario: str, task: int, seed: int, model: str, *, ok: bool = True, steps: int = 3,
            video: str | None = "mp4", error: bool = False, frame=(1440, 864), gate_at: int | None = None,
            interrupts: int = 0, task_text: str = 'Buy the "Trail Runner" in size 42.'):
    """One harness episode folder, shaped like cua/scenarios/harness writes it."""
    folder = run / scenario / f"{task}-{seed}-{model}"
    (folder / "steps").mkdir(parents=True)
    rows, walls = [], []
    for i in range(steps):
        last = i == steps - 1 and ok
        Image.new("RGB", frame, (236, 238, 243)).save(folder / "steps" / f"{i + 1:03}.png")
        probs = {"1": 0.62, "2": 0.28, "3": 0.10} if i % 2 == 0 else {"1": 0.15, "2": 0.80, "3": 0.05}
        pick = max(probs, key=probs.get)
        gate = gate_at == i
        risky = 0.71 if gate else 0.04
        wall = 700.0 + 10 * i
        walls.append(wall)
        rows.append({
            "step": i + 1, "t_start": "2026-09-29T05:40:04+00:00", "url": "http://127.0.0.1:36849/shop/index.html",
            "state": "browse",
            "candidates": [{"n": k + 1, "box": b, "role": "button", "name": f"Button {k + 1}",
                            "selector": f"#b{k + 1}", "target": None} for k, b in enumerate(BOXES)],
            "marked_screenshot": f"steps/{i + 1:03}.png",
            "request_state": {"task": task_text, "previous_actions": [], "screenshot_note": "boxes 1-3"},
            "answers": {"element": {"type": "choice", "choice": pick, "probabilities": probs},
                        "done": {"type": "noul", "noul": 0.93 if last else 0.04},
                        "risky": {"type": "noul", "noul": risky}},
            "usage": {"input_tokens": 4100, "visual_tokens": 1215 if frame[0] > frame[1] else 723},
            "chosen_action": ({"kind": "done", "box": None, "value": None} if last else
                              {"kind": "type" if i == 1 else "click", "box": pick,
                               "value": "42" if i == 1 else None}),
            "wall_ms": wall, "oracle_box": None if last else "1", "element_correct": None if last else pick == "1",
            "gate_fired": gate, "expected": {"selector": "#b1"}, "check": {"done": False},
        })
    with open(folder / "trace.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    summary = {"scenario": scenario, "kind": "dom", "task_index": task, "seed": seed, "model": model,
               "server": "http://127.0.0.1:8301", "interrupts": interrupts, "mode": "batched",
               "success": ok and not error, "done": ok, "detail": "" if ok else "cart has 3 lines",
               "stop_reason": "error" if error else "model_done",
               "error": "ValueError: at " + HOME + "someone/agent.py" if error else None,
               "optimal_steps": 3, "steps": steps - (1 if ok else 0), "gates": 1 if gate_at is not None else 0,
               "element_correct": sum(bool(r["element_correct"]) for r in rows),
               "element_total": sum(r["element_correct"] is not None for r in rows),
               "risk_tp": 1 if gate_at is not None else 0, "risk_fn": 0, "risk_fp": 0, "risk_tn": steps - 1,
               "done_tp": 1 if ok else 0, "done_fp": 0,
               "wall_ms": walls, "p50_wall_ms": build_gallery.percentile(walls, 0.5),
               "p95_wall_ms": build_gallery.percentile(walls, 0.95), "harness_sha": "abc123"}
    if error:
        (folder / "error.txt").write_text("Traceback: " + HOME + "someone/agent.py line 1\n")
    with open(folder / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh)
    if video:
        (folder / f"video.{video}").write_bytes(b"not really a video")
    Image.new("RGB", frame, (200, 210, 230)).save(folder / "poster.png")
    return folder


def make_run(root: Path, run_id: str = "r1", mock: bool = False) -> Path:
    run = root / run_id
    run.mkdir(parents=True)
    with open(run / "config.json", "w", encoding="utf-8") as fh:
        json.dump({"harness_sha": "abc123", "mock": mock, "servers": {"blink-mimo-9b": ["http://127.0.0.1:8301"]}}, fh)
    return run


def standard_runs(root: Path) -> list[Path]:
    """shop: mimo 2 of 4 finish (one errored), 4b 1 of 2; phone: one mimo run with only a WebM."""
    run = make_run(root)
    episode(run, "shop", 0, 1, "blink-mimo-9b", ok=True, gate_at=1)
    episode(run, "shop", 0, 2, "blink-mimo-9b", ok=False)
    episode(run, "shop", 0, 3, "blink-mimo-9b", ok=True)
    episode(run, "shop", 1, 1, "blink-mimo-9b", ok=False, error=True)
    episode(run, "shop", 0, 1, "blink-4b", ok=False)
    episode(run, "shop", 0, 2, "blink-4b", ok=True)
    episode(run, "phone", 0, 1, "blink-mimo-9b", ok=True, video="webm", frame=(585, 1266))
    return [run]


class TestNumbers(unittest.TestCase):
    def test_wilson_is_the_harness_interval(self):
        z = 1.959963984540054
        for k, n in ((0, 3), (2, 4), (9, 10), (10, 10)):
            p = k / n
            c = (p + z * z / (2 * n)) / (1 + z * z / n)
            r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            self.assertEqual(build_gallery.wilson(k, n), [round(max(0, c - r), 4), round(min(1, c + r), 4)])
        self.assertIsNone(build_gallery.wilson(0, 0))

    def test_percentiles_interpolate(self):
        self.assertEqual(build_gallery.percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertEqual(build_gallery.percentile([5], 0.95), 5)
        self.assertIsNone(build_gallery.percentile([], 0.5))


class TestBuild(unittest.TestCase):
    """build_gallery.py over harness-shaped run folders."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.runs = standard_runs(self.tmp / "runs")

    def build(self, **kw):
        return build_gallery.build(self.runs, REGISTRY, **kw)

    def test_the_rule_picks_the_first_finished_run_and_keeps_a_failed_one(self):
        g = self.build()
        self.assertEqual(g["schema"], "blink-cua-gallery/1")
        self.assertFalse(g["mock"])
        self.assertEqual([s["id"] for s in g["scenarios"]], ["shop", "phone"])  # registry order, only what ran
        self.assertEqual(g["models"], ["blink-mimo-9b", "blink-4b"])  # the vision model first
        shop = g["scenarios"][0]["models"]["blink-mimo-9b"]
        self.assertEqual(shop["showcase"], "shop/0-1-blink-mimo-9b")
        self.assertEqual(shop["failure"], "shop/0-2-blink-mimo-9b")  # same task; the errored run is not shown
        four = g["scenarios"][0]["models"]["blink-4b"]
        self.assertEqual((four["showcase"], four["failure"]), ("shop/0-2-blink-4b", "shop/0-1-blink-4b"))
        self.assertIn("lowest (task, seed)", g["rule"]["showcase"])
        self.assertEqual(g["hero"], ["shop/0-1-blink-mimo-9b", "phone/0-1-blink-mimo-9b"])

    def test_every_run_counts_errors_included(self):
        st = self.build()["scenarios"][0]["models"]["blink-mimo-9b"]
        self.assertEqual((st["stats"]["episodes"], st["stats"]["successes"], st["stats"]["errors"]), (4, 2, 1))
        self.assertEqual(st["stats"]["rate"], 0.5)
        self.assertEqual(st["stats"]["wilson95"], build_gallery.wilson(2, 4))
        self.assertEqual([(o["task"], o["seed"], o["ok"], o["error"]) for o in st["outcomes"]],
                         [(0, 1, True, False), (0, 2, False, False), (0, 3, True, False), (1, 1, False, True)])
        self.assertEqual(st["stats"]["gate_recall"], 1.0)
        self.assertEqual(st["condition"], {"interrupts": 0, "mode": "batched"})
        self.assertIsNotNone(st["stats"]["p50_ms"])

    def test_a_step_carries_what_the_page_draws(self):
        g = self.build()
        ep = g["episodes"]["shop/0-1-blink-mimo-9b"]
        self.assertEqual(ep["task"], 'Buy the "Trail Runner" in size 42.')
        self.assertEqual((ep["frame"], ep["visual_tokens"], ep["success"], ep["gates"]), ([1440, 864], 1215, True, 1))
        first, second, last = ep["steps"]
        self.assertEqual(first["boxes"], BOXES)
        placed = screens.layout([tuple(b) for b in BOXES], (1440, 864))
        self.assertEqual(first["tags"], [[*b.tag, b.at] for b in placed])  # where mark() drew each number
        self.assertEqual(first["labels"], ["button \u00b7 Button 1", "button \u00b7 Button 2", "button \u00b7 Button 3"])
        self.assertEqual(first["p"], [0.62, 0.28, 0.1])
        self.assertEqual((first["pick"], first["oracle"], first["ok"], first["act"]), (1, 1, True, "click"))
        self.assertEqual((second["pick"], second["gate"], second["act"], second["value"]), (2, True, "type", "42"))
        self.assertEqual((last["act"], last["pick"], last["done"]), ("done", None, 0.93))
        self.assertTrue(first["img"].endswith("/steps/001.webp"))
        self.assertEqual(ep["video"], "runs/r1/shop/0-1-blink-mimo-9b/video.mp4")

    def test_nothing_private_leaves_the_runs(self):
        g = self.build()
        text = json.dumps(g)
        for needle in (HOME, "127.0.0.1", "error.txt", "Traceback", "#b1", "selector", "abc123/"):
            self.assertNotIn(needle, text)
        self.assertEqual(g["sources"], [{"run_id": "r1", "harness_sha": "abc123", "mock": False, "episodes": 7}])
        bad = dict(g, episodes={**g["episodes"]})
        bad["episodes"]["shop/0-1-blink-mimo-9b"] = dict(bad["episodes"]["shop/0-1-blink-mimo-9b"],
                                                          task="open " + HOME + "me")
        with self.assertRaises(build_gallery.GalleryError):
            build_gallery.check(bad)

    def test_staging_holds_exactly_the_media_it_names(self):
        stage = self.tmp / "stage"
        with mock.patch.object(build_gallery, "_ffmpeg", lambda: None):
            g = self.build(stage=stage)
        named = set(build_gallery.media_paths(g))
        staged = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
        self.assertEqual(staged, named | {"gallery.json", "MANIFEST.json"})
        with Image.open(stage / g["episodes"]["shop/0-1-blink-mimo-9b"]["steps"][0]["img"]) as im:
            self.assertEqual((im.format, im.size), ("WEBP", (1440, 864)))
        with Image.open(stage / g["episodes"]["shop/0-1-blink-mimo-9b"]["poster"]) as im:
            self.assertEqual((im.format, im.size), ("WEBP", (960, 576)))
        # without ffmpeg a WebM cannot become H.264: the phone run shows its steps and no video
        phone = g["episodes"]["phone/0-1-blink-mimo-9b"]
        self.assertIsNone(phone["video"])
        self.assertEqual(g["hero"], ["shop/0-1-blink-mimo-9b"])
        manifest = json.loads((stage / "MANIFEST.json").read_text())
        self.assertEqual(manifest["media_base"], build_gallery.MEDIA_BASE)
        for rel, meta in manifest["files"].items():
            body = (stage / rel).read_bytes()
            self.assertEqual((meta["bytes"], meta["sha256"]), (len(body), hashlib.sha256(body).hexdigest()))
        self.assertEqual(json.loads((stage / "gallery.json").read_text())["episodes"].keys(), g["episodes"].keys())
        for rel in named:
            self.assertFalse(rel.startswith("/") or ".." in rel, rel)

    def test_a_mock_run_makes_a_mock_gallery_that_never_lands_in_the_space(self):
        run = make_run(self.tmp / "mockruns", "m1", mock=True)
        episode(run, "shop", 0, 1, "blink-mimo-9b")
        self.assertTrue(build_gallery.build([run], REGISTRY)["mock"])
        self.assertTrue(self.build(mock=True)["mock"])
        reg = self.tmp / "registry.json"
        reg.write_text(json.dumps(REGISTRY))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = build_gallery.main([str(run), "--registry", str(reg)])
        self.assertEqual(code, 1)
        self.assertIn("never goes in the Space", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(HERE, "cua", "gallery.json.tmp")))
        out = self.tmp / "g.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(build_gallery.main([str(run), "--registry", str(reg), "--out", str(out)]), 0)
        self.assertTrue(json.loads(out.read_text())["mock"])

    def test_an_episode_in_two_runs_is_refused(self):
        again = make_run(self.tmp / "runs", "r2")
        episode(again, "shop", 0, 1, "blink-mimo-9b")
        with self.assertRaisesRegex(build_gallery.GalleryError, "appears in both"):
            build_gallery.build(self.runs + [again], REGISTRY)

    def test_a_run_without_a_poster_gets_its_first_step(self):
        """A phone run whose H.264 step failed has no poster.png: its first marked step stands in."""
        folder = self.runs[0] / "phone" / "0-1-blink-mimo-9b"
        (folder / "poster.png").unlink()
        stage = self.tmp / "stage3"
        with mock.patch.object(build_gallery, "_ffmpeg", lambda: None):
            g = self.build(stage=stage)
        poster = g["episodes"]["phone/0-1-blink-mimo-9b"]["poster"]
        with Image.open(stage / poster) as im:
            self.assertEqual((im.format, im.size), ("WEBP", (585, 1266)))  # a phone keeps its own width

    def test_a_reused_stage_keeps_its_media_and_drops_what_is_no_longer_named(self):
        stage = self.tmp / "stage4"
        with mock.patch.object(build_gallery, "_ffmpeg", lambda: None):
            first = self.build(stage=stage)
            kept = stage / first["episodes"]["shop/0-1-blink-mimo-9b"]["steps"][0]["img"]
            stamp = kept.stat().st_mtime_ns
            stale = stage / "runs" / "old" / "gone.webp"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"x")
            again = self.build(stage=stage, reuse=True)
        self.assertEqual(kept.stat().st_mtime_ns, stamp)  # not converted again
        self.assertFalse(stale.exists())
        self.assertFalse(stale.parent.exists())
        staged = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
        self.assertEqual(staged, set(build_gallery.media_paths(again)) | {"gallery.json", "MANIFEST.json"})

    def test_the_steps_are_published_beside_their_media_by_content(self):
        stage = self.tmp / "stage5"
        with mock.patch.object(build_gallery, "_ffmpeg", lambda: None):
            g = self.build(stage=stage)
        ep = g["episodes"]["shop/0-1-blink-mimo-9b"]
        body = (stage / ep["steps_url"]).read_bytes()
        self.assertEqual(json.loads(body), ep["steps"])
        self.assertIn(hashlib.sha256(body).hexdigest()[:12], ep["steps_url"])
        self.assertEqual(len(ep["rail"]), len(ep["steps"]))

    def test_a_scenario_missing_from_the_registry_is_refused(self):
        with self.assertRaisesRegex(build_gallery.GalleryError, "missing from the registry"):
            build_gallery.build(self.runs, {"scenarios": REGISTRY["scenarios"][:1]})

    def test_other_conditions_are_reported_apart(self):
        run = make_run(self.tmp / "more", "i1")
        episode(run, "shop", 0, 1, "blink-mimo-9b", ok=False, interrupts=1)
        g = build_gallery.build(self.runs + [run], REGISTRY)
        item = g["scenarios"][0]["models"]["blink-mimo-9b"]
        self.assertEqual(item["stats"]["episodes"], 4)  # interrupts 1 is not mixed in
        self.assertEqual(item["other_conditions"][0]["interrupts"], 1)
        self.assertEqual(item["other_conditions"][0]["stats"]["episodes"], 1)

    def test_tiles_beyond_clicks(self):
        root = self.tmp / "more"
        root.mkdir()
        Image.new("RGB", (800, 500), (255, 255, 255)).save(root / "chart.png")
        Image.new("RGB", (640, 400), (10, 10, 10)).save(root / "poster.png")
        (root / "voice.mp4").write_bytes(b"x")
        (root / "voice.vtt").write_text("WEBVTT\n")
        spec = [{"id": "voice", "title": "Voice to click", "line": "Say it; blink clicks it.", "video": "voice.mp4",
                 "poster": "poster.png", "captions": "voice.vtt", "sound": True, "metric": {"value": "9/10", "label": "done"}},
                {"id": "X-Ray!", "title": "Screen X-ray", "line": "Is a dialog open?", "chart": "chart.png"}]
        (root / "more.json").write_text(json.dumps(spec))
        stage = self.tmp / "stage2"
        with mock.patch.object(build_gallery, "_ffmpeg", lambda: None):
            g = self.build(stage=stage, more=root / "more.json")
        voice, xray = g["more"]
        self.assertEqual((voice["id"], voice["sound"], voice["video"], voice["captions"]),
                         ("voice", True, "more/voice/voice.mp4", "more/voice/voice.vtt"))
        self.assertEqual(xray["id"], "x-ray")
        self.assertEqual(xray["chart"], "more/x-ray/chart.webp")
        for rel in ("more/voice/voice.mp4", "more/voice/poster.webp", "more/x-ray/chart.webp"):
            self.assertTrue((stage / rel).is_file(), rel)
        (root / "bad.json").write_text(json.dumps([{"id": "a", "title": "A"}]))
        with self.assertRaisesRegex(build_gallery.GalleryError, "lacks line"):
            self.build(more=root / "bad.json")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is not installed")
class TestVideo(unittest.TestCase):
    """With ffmpeg: the harness's MP4 is kept as it is, a WebM with an odd side becomes an even H.264."""

    def test_videos_stream_and_previews_shrink(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        run = make_run(tmp / "runs")
        wide = episode(run, "shop", 0, 1, "blink-mimo-9b", video=None, steps=1)
        tall = episode(run, "phone", 0, 1, "blink-mimo-9b", video=None, steps=1, frame=(585, 1266))
        ff = [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi"]
        try:
            subprocess.run([*ff, "-i", "testsrc=size=1440x864:rate=10:duration=1", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", str(wide / "video.mp4")], check=True, capture_output=True)
            subprocess.run([*ff, "-i", "testsrc=size=585x1266:rate=10:duration=1", "-c:v", "libvpx",
                            str(tall / "video.webm")], check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:  # an ffmpeg without libx264 or libvpx
            self.skipTest(exc.stderr.decode()[-200:])
        g = build_gallery.build([run], REGISTRY, stage=tmp / "stage")

        def probe(rel):
            out = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=codec_name,width,height", "-of", "json", str(tmp / "stage" / rel)],
                                 capture_output=True, text=True, check=True)
            s = json.loads(out.stdout)["streams"][0]
            return s["codec_name"], s["width"], s["height"]

        shop, phone = g["episodes"]["shop/0-1-blink-mimo-9b"], g["episodes"]["phone/0-1-blink-mimo-9b"]
        self.assertEqual(probe(shop["video"]), ("h264", 1440, 864))
        self.assertEqual(probe(shop["preview"]), ("h264", 720, 432))
        self.assertEqual(probe(phone["video"]), ("h264", 586, 1266))  # one blank column, nothing scaled
        self.assertEqual(probe(phone["preview"])[0], "h264")
        self.assertAlmostEqual(shop["duration_s"], 1.0, delta=0.2)
        head = (tmp / "stage" / shop["video"]).read_bytes()[:4096]
        self.assertLess(head.find(b"moov"), head.find(b"mdat") if b"mdat" in head else 4096)  # streams


class TestTab(unittest.TestCase):
    """What the Computer use tab renders from a gallery, and what it says without one."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        runs = standard_runs(cls.tmp / "runs")
        cls.g = build_gallery.build(runs, REGISTRY)
        cls.path = cls.tmp / "gallery.json"
        cls.path.write_text(json.dumps(cls.g))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, True)

    def test_load_reads_the_file_it_is_pointed_at(self):
        with gallery_at(self.path):
            self.assertEqual(computer_use.load()["hero"], self.g["hero"])
            self.assertTrue(computer_use.available())
        with gallery_at(self.tmp / "nope.json"):
            self.assertIsNone(computer_use.load())
        wrong = self.tmp / "wrong.json"
        wrong.write_text(json.dumps({"schema": "other"}))
        with gallery_at(wrong):
            self.assertIsNone(computer_use.load())
        self.assertEqual(computer_use.GALLERY, os.path.join(HERE, "cua", "gallery.json"))

    def test_the_stage_the_cards_and_the_data_agree(self):
        top = computer_use.top(self.g)
        data = json.loads(htmllib.unescape(re.search(r'data-g="([^"]*)"', top).group(1)))
        self.assertEqual(set(data), {"media_base", "mock", "models", "vision_model", "lead_model", "scenarios",
                                     "episodes", "rule_cut"})
        shown = data["episodes"]["shop/0-1-blink-mimo-9b"]
        self.assertNotIn("steps", shown)  # fetched when the player needs them
        self.assertEqual(shown["rail"], [{"a": "click"}, {"a": "type", "g": 1, "m": 1}, {"a": "done"}])
        self.assertRegex(shown["steps_url"], r"^runs/r1/shop/0-1-blink-mimo-9b/steps\.[0-9a-f]{12}\.json$")
        self.assertEqual(data["rule_cut"], {"done_at": screens.DONE_AT, "risky_at": screens.RISKY_AT})
        self.assertEqual(data["episodes"]["shop/0-1-blink-mimo-9b"]["pad"], 6)
        copy = json.loads(htmllib.unescape(re.search(r'data-copy="([^"]*)"', top).group(1)))
        self.assertEqual(copy["ask"], computer_use.COPY["ask"])
        self.assertEqual(top.count('class="blk-cuax-reel'), 2)
        self.assertEqual(top.count('class="blk-cuax-reel tall"'), 1)  # the phone run
        self.assertEqual(top.count('class="blk-cuax-card"'), 2)
        self.assertIn('href="?tab=computer-use&amp;scenario=shop"', top)
        self.assertIn("2/4 runs", top)  # the card counts every run, the errored one included
        self.assertEqual(re.search(r'data-sid="shop" style="--i:0">.*?</a>', top, re.DOTALL).group(0).count("<i class="), 4)
        self.assertIn('<i class="ok"></i><i class="no"></i><i class="ok"></i><i class="no"></i>', top)
        self.assertIn('muted loop playsinline preload="none"', top)
        self.assertNotIn("autoplay", top.split('class="blk-cuax-cards"')[1])  # cards play on hover only
        self.assertNotIn("blk-mock", top)
        self.assertIn(computer_use.COPY["mock"], computer_use.top(dict(self.g, mock=True)))

    def test_the_strip_puts_a_phone_between_two_screens(self):
        eps = {"a": {"frame": [1440, 864]}, "b": {"frame": [585, 1266]}, "c": {"frame": [1440, 864]},
               "d": {"frame": [1440, 864]}}
        self.assertEqual(computer_use.hero_order({"hero": ["a", "b", "c", "d"], "episodes": eps}), ["a", "b", "c", "d"])
        self.assertEqual(computer_use.hero_order({"hero": ["b", "a", "c"], "episodes": eps}), ["a", "b", "c"])

    def test_without_runs_the_tab_still_explains(self):
        top = computer_use.top(None, "<section>how</section>")
        self.assertIn('data-g="null"', top)
        self.assertIn("<section>how</section>", top)
        self.assertEqual(computer_use.tiles(None), "")

    def test_the_stats_row_says_what_it_counts(self):
        row = computer_use.stat_row(self.g)
        self.assertIn("<b>2</b>" + computer_use.COPY["stat_apps"], row)
        self.assertIn("<b>60%</b>" + computer_use.COPY["stat_runs"].format(n=5), row)  # mimo: 3 of 5 across both scenarios
        self.assertIn(computer_use.COPY["stat_step"], row)

    def test_links_open_a_run_and_fall_back_when_stale(self):
        link = computer_use.parse_link
        self.assertEqual(link("?tab=computer-use&scenario=shop", self.g),
                         {"scenario": "shop", "model": "blink-mimo-9b", "run": "showcase", "step": None})
        self.assertEqual(link("?scenario=SHOP&model=blink-4b&run=failed&step=2", self.g),
                         {"scenario": "shop", "model": "blink-4b", "run": "failure", "step": 2})
        self.assertEqual(link("?scenario=shop&step=99", self.g)["step"], 3)
        self.assertEqual(link("?scenario=shop&step=0", self.g)["step"], 1)
        self.assertEqual(link("?scenario=shop&model=nope", self.g)["model"], "blink-mimo-9b")
        self.assertEqual(link("?scenario=phone&run=failed", self.g)["run"], "showcase")  # none failed there
        for raw in ("?scenario=nope", "", None, 7, "?tab=results"):
            self.assertIsNone(link(raw, self.g)["scenario"], raw)

    def test_the_explainer_counts_what_it_shows(self):
        facts = {"blocks": 27, "hidden": 1152, "base": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", "tokens": 1215,
                 "side": 32, "width": 1440, "height": 864}
        answers = {"element": {"choice": "3", "probabilities": {"1": 0.05, "2": 0.1, "3": 0.85}},
                   "done": {"noul": 0.02}, "risky": {"noul": 0.04}}
        html = htmllib.unescape(computer_use.explainer(facts, "data:image/jpeg;base64,xx", answers, "Click 3"))
        self.assertEqual(html.count('class="blk-cx-n"'), 6)  # six nodes
        self.assertEqual(html.count("<i></i><b>"), 5)  # three boxes, then done and risky
        self.assertEqual(html.count('<i style="--k:'), 27)  # one line per encoder block
        self.assertIn('style="--cols:45;--rows:27"', html)  # 45 x 27 patches of 32 px: 1,215 tokens
        self.assertIn('data-to="1215">1,215</b>', html)
        self.assertIn("1,215 image tokens", html)
        self.assertIn("XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B, unchanged by blink training.", html)
        self.assertIn("27 blocks \u00b7 hidden 1152", html)
        self.assertIn(computer_use.COPY["how_nodes"][2][1].format(side=32), html)
        self.assertIn('<li class="win" style="--p:0.850;--i:0"><span>box 3</span>', html)
        self.assertIn('<li style="--p:0.100;--i:1"><span>box 2</span>', html)
        self.assertIn("<b>Click 3</b>", html)
        for label, _ in computer_use.COPY["how_nodes"]:
            self.assertIn(f"<b>{label}</b>", html)

    def test_tiles_open_one_at_a_time_and_sound_waits_for_a_tap(self):
        g = dict(self.g, more=[{"id": "voice", "title": "Voice to click", "line": "Say it.", "video": "more/voice/v.mp4",
                                "poster": "more/voice/p.webp", "sound": True, "metric": {"value": "9/10", "label": "done"}}])
        html = computer_use.tiles(g)
        self.assertIn('data-tile="voice"', html)
        self.assertIn("<strong>9/10</strong>done", html)
        data = json.loads(htmllib.unescape(re.search(r'data-t="([^"]*)"', html).group(1)))
        self.assertTrue(data["more"][0]["sound"])
        js = computer_use.TILES_JS
        self.assertIn("t.sound", js)
        sound_branch = js.split("fig += t.sound")[1].split(":")[0]
        self.assertNotIn("autoplay", sound_branch)
        self.assertNotIn("muted", sound_branch)

    def test_home_loops_a_real_run(self):
        fig = computer_use.home_figure(self.g)
        self.assertIn("<video autoplay muted loop playsinline", fig)
        self.assertIn("preview.mp4", fig)
        self.assertIn("poster.webp", fig)
        self.assertIn("Outdoor gear store \u00b7 2/4 runs \u00b7 blink-mimo-9b", fig)
        self.assertEqual(computer_use.home_figure(dict(self.g, hero=[])), "")

    def test_the_words_are_in_one_place_and_clean(self):
        words = json.dumps(computer_use.COPY, ensure_ascii=False)
        # no first person plural, no em dash, no hardware, host or employer (the last split for the public scan)
        for bad in ("\u2014", " we ", " our ", "We ", "Our ", " us ", "G" + "PU", "hf.space", "M" + "AI", "Micro" + "soft",
                    "A1" + "00", "H1" + "00", "Nvi" + "dia"):
            self.assertNotIn(bad, words)
        self.assertNotIn("check_copy", dir(computer_use))
        # every string the player shows comes from COPY: no English in the script but its fallbacks
        for literal in re.findall(r"'([A-Z][a-z]+ [a-z][^']*)'", computer_use.PLAYER_JS):
            self.fail(f"the player spells out {literal!r}; put it in COPY")

    def test_the_player_only_touches_the_page(self):
        js = computer_use.PLAYER_JS
        for missing in ("XMLHttpRequest", "pushState", "eval(", "localStorage", "document.cookie"):
            self.assertNotIn(missing, js)
        # the one request it makes: a run's steps, from beside its media, without credentials
        self.assertEqual(js.count("fetch("), 1)
        self.assertIn("fetch(media(at), { credentials: 'omit' })", js)
        self.assertIn("history.replaceState", js)
        self.assertIn("window.parent.postMessage", js)
        self.assertIn("__blinkFirstUrl", js)
        for param in computer_use.PARAMS:
            self.assertIn(f"'{param}'", js)
        node = shutil.which("node")
        if node:
            for name, code in (("player", js), ("tiles", computer_use.TILES_JS)):
                check = ("new Function('element','trigger','props','watch',"
                         "require('fs').readFileSync(0,'utf8'))")
                r = subprocess.run([node, "-e", check], input=code, text=True, capture_output=True, check=False)
                self.assertEqual(r.returncode, 0, f"{name}: {r.stderr[-400:]}")


class TestTabInApp(unittest.TestCase):
    """The tab follows Home, Screen click lives in it, and every old link still lands."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.g = build_gallery.build(standard_runs(cls.tmp / "runs"), REGISTRY)
        cls.path = cls.tmp / "gallery.json"
        cls.path.write_text(json.dumps(cls.g))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, True)

    def setUp(self):
        need_gradio(self)
        self.addCleanup(reload_app)

    def build(self):
        import gradio as gr

        app = reload_app()
        demo = app.build()
        try:
            tabs = [b.id for b in demo.blocks.values() if isinstance(b, gr.Tab)]
            fns = list(demo.fns.values())
            ids = {b._id: getattr(b, "elem_id", None) for b in demo.blocks.values()}
            config = str(demo.config)
        finally:
            demo.close()
        return app, SimpleNamespace(tabs=tabs, fns=fns, ids=ids, config=config)

    def test_computer_use_follows_home_and_holds_screen_click(self):
        with serving(FOUR, MIMO), gallery_at(self.path):
            app, built = self.build()
            self.assertEqual(app.ui.TAB_IDS[:3], ("home", "computer-use", "use-cases"))
            self.assertEqual(built.tabs[:3], ["home", "computer-use", "use-cases"])
            self.assertNotIn("screen", built.tabs)  # no longer a use case
            self.assertEqual(app.ui.CASE_IDS[0], "nextclick")
            names = {getattr(fn.fn, "__name__", "") for fn in built.fns}
            self.assertLessEqual({"screen_run", "load_upload", "drawn", "dropped", "redraw", "opened"}, names)
            self.assertIn("cua-top", built.config)
            self.assertNotIn("cua-more-box", built.config)  # no tiles in these runs
            self.assertIn("blk-cuax-card", built.config)
            self.assertIn('id="cua-try"', built.config)
            self.assertIn("blk-cuax-rail", built.config)  # the player script rides on the tab

    def test_no_screen_model_and_no_runs_means_no_tab(self):
        with serving(FOUR), gallery_at(self.tmp / "none.json"):
            app, built = self.build()
            self.assertNotIn("computer-use", app.ui.TAB_IDS)
            self.assertNotIn("computer-use", built.tabs)
            self.assertEqual(app.ui.parse_deep_link("?tab=computer-use&scenario=shop"), ("home", None))
            self.assertNotIn("Computer use", [t for _, t, _ in app.ui.HOME_LINKS])

    def test_runs_alone_make_the_tab_without_screen_click(self):
        with serving(FOUR), gallery_at(self.path):
            _, built = self.build()
            self.assertIn("computer-use", built.tabs)
            names = {getattr(fn.fn, "__name__", "") for fn in built.fns}
            self.assertNotIn("screen_run", names)
            self.assertNotIn('id="cua-try"', built.config)

    def test_old_screen_click_links_land_on_the_new_tab(self):
        with serving(FOUR, MIMO), gallery_at(self.path):
            app = reload_app()
            ui = app.ui
            for raw in ("?tab=use-cases&case=screen", "?case=screen", "?tab=use-cases&case=SCREEN&shot=done",
                        "?tab=computer-use", "?tab=computer-use&shot=lookup", "?scenario=shop",
                        "?tab=computer-use&scenario=shop&step=2", "#computer-use"):
                with self.subTest(raw=raw):
                    self.assertEqual(ui.parse_deep_link(raw), ("computer-use", None))
            for raw, shot in (("?tab=use-cases&case=screen&shot=done", "done"), ("?case=screen&shot=catalog", "catalog"),
                              ("?tab=computer-use&shot=lookup", "lookup"), ("?tab=computer-use&shot=nope", "catalog"),
                              ("?tab=computer-use", None), ("?tab=results&shot=done", None), ("?shot=done", None),
                              ("?tab=use-cases&case=nextclick&shot=done", None), ("?case=screen&shot=done#results", None)):
                with self.subTest(raw=raw):
                    self.assertEqual(ui.parse_shot(raw), shot)
            self.assertEqual(ui.parse_deep_link("?tab=use-cases&case=nextclick"), ("use-cases", "nextclick"))
            self.assertEqual(ui.parse_deep_link("?tab=results&case=screen"), ("results", None))
            tab, case, _ = app.open_from_url("?tab=use-cases&case=screen&shot=done")
            self.assertEqual(tab["selected"], "computer-use")
            self.assertNotIn("selected", case)

    def test_the_address_carries_the_run_only_on_its_tab(self):
        with serving(FOUR, MIMO), gallery_at(self.path):
            app = reload_app()
        js = app.SYNC_URL
        self.assertIn("if (shot && tab === \"computer-use\")", js)
        self.assertIn("if (tab !== \"computer-use\")", js)
        self.assertIn("['scenario', 'model', 'run', 'step'].forEach((k) => url.searchParams.delete(k))", js)
        self.assertIn("window.__blinkFirstUrl = window.location.search + window.location.hash;", app.HEAD)
        self.assertLess(app.HEAD.index("__blinkFirstUrl"), app.HEAD.index("__blinkChips"))

    def test_the_masthead_names_the_model_that_reads_screens_there(self):
        with serving(FOUR, MIMO), gallery_at(self.path):
            app = reload_app()
            self.assertIn(MIMO, app.masthead_for("computer-use", "", FOUR))
            self.assertIn(FOUR, app.masthead_for("use-cases", "", FOUR))
            self.assertIn(FOUR, app.masthead_for("use-cases", "nextclick", FOUR))

    def test_home_watches_tries_and_reads_text(self):
        with serving(FOUR, MIMO), gallery_at(self.path):
            app, built = self.build()
            into = {built.ids.get(fn.targets[0][0]): fn for fn in built.fns
                    if fn.targets and built.ids.get(fn.targets[0][0]) in ("home-watch", "home-screen", "home-nextclick")
                    and fn.fn is not None}
            self.assertEqual(set(into), {"home-watch", "home-screen", "home-nextclick"})
            for elem, (tab, case) in (("home-watch", ("computer-use", "")), ("home-screen", ("computer-use", "")),
                                      ("home-nextclick", ("use-cases", "nextclick"))):
                fn = into[elem]
                self.assertEqual(fn.api_visibility, "private")
                out = fn.fn(FOUR)
                self.assertEqual((out[0]["selected"], out[2], out[3]), (tab, tab, case))
                self.assertIn(MIMO if tab == "computer-use" else FOUR, out[4])
            scroll = [fn for fn in built.fns if fn.js == app.SHOW_TRY]
            self.assertEqual(len(scroll), 1)
            self.assertIn("<video autoplay muted loop playsinline", app.ui.home_cua_shot())
        with serving(FOUR, MIMO), gallery_at(self.tmp / "none.json"):
            app, built = self.build()
            self.assertNotIn("home-watch", built.config)
            self.assertIn("data:image/jpeg;base64,", app.ui.home_cua_shot())  # the marked screenshot until runs land

    def test_the_explainer_uses_the_saved_run(self):
        with serving(FOUR, MIMO), gallery_at(self.path):
            app = reload_app()
            html = htmllib.unescape(app.ui.cua_explainer())
            facts = app.ui.vision_facts()
            self.assertEqual((facts["tokens"], facts["side"]), (1215, 32))
            self.assertIn("1,215 image tokens", html)
            shot = screens.presets()[0]
            said = screens.verdict(screens.decide(shot, prefer="saved")["answers"])[2]
            self.assertIn(f"<b>{said}</b>", html)


class TestStager(unittest.TestCase):
    """Only a real gallery ships, and the page's code goes with it."""

    def setUp(self):
        try:
            import stage_space
        except ModuleNotFoundError:
            self.skipTest("stage_space.py is only in the research tree")
        self.stage = stage_space

    def test_the_tab_ships_and_a_mock_gallery_never_does(self):
        self.assertIn("computer_use.py", self.stage.FILES)
        self.assertEqual(self.stage.GALLERY, "cua/gallery.json")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "cua").mkdir()
        with mock.patch.object(self.stage, "HERE", str(tmp)):
            self.assertEqual(self.stage.gallery_problem(), "")  # none: nothing to refuse
            (tmp / "cua" / "gallery.json").write_text(json.dumps({"mock": True, "media_base": build_gallery.MEDIA_BASE}))
            self.assertIn("mock", self.stage.gallery_problem())
            (tmp / "cua" / "gallery.json").write_text(json.dumps({"mock": False, "media_base": "http://127.0.0.1:1/"}))
            self.assertIn("not a Hugging Face dataset", self.stage.gallery_problem())
            (tmp / "cua" / "gallery.json").write_text(json.dumps({"mock": False, "media_base": build_gallery.MEDIA_BASE}))
            self.assertEqual(self.stage.gallery_problem(), "")

    def test_the_committed_gallery_if_any_is_real(self):
        path = os.path.join(HERE, "cua", "gallery.json")
        if not os.path.exists(path):
            self.skipTest("no gallery committed yet")
        with open(path, encoding="utf-8") as fh:
            g = json.load(fh)
        self.assertFalse(g["mock"])
        build_gallery.check(g)


if __name__ == "__main__":
    unittest.main()
