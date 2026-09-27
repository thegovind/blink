"""Screen click: the fixture presets, their marks, the screenshot request and where it is answered.

    cd space && BLINK_MOCK=1 uv run --no-project --python 3.12 --with gradio==6.28.0 --with huggingface_hub \\
        python -m unittest test_screens
"""

import contextlib
import hashlib
import html as htmllib
import importlib
import io
import json
import os
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from PIL import Image

os.environ.setdefault("BLINK_MOCK", "1")
import blink  # noqa: E402
import screens  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MIMO = "thegovind/blink-mimo-9b"
FOUR = "thegovind/blink-4b"
BLUE = screens.BLUE
# Recorded on a GPU with record_replay.py --vision --screens-only (replay-blink-mimo-9b.json).
SCREEN_REPLAY_PENDING = False


@contextlib.contextmanager
def serving(*ids):
    """Pretend this deployment serves these models, with fresh engines, then put everything back."""
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


def png_file(width=1280, height=800, kind="PNG", color=(244, 246, 251)) -> str:
    fd, path = tempfile.mkstemp(suffix="." + kind.lower())
    os.close(fd)
    Image.new("RGB", (width, height), color).save(path, format=kind)
    return path


def pixel(png: bytes, x: int, y: int):
    with Image.open(io.BytesIO(png)) as im:
        return im.convert("RGB").getpixel((x, y))


def _stage_space():
    try:
        import stage_space
    except ModuleNotFoundError:
        raise unittest.SkipTest("stage_space.py (the Space stager) is only in the research tree") from None
    return stage_space


class TestPresets(unittest.TestCase):
    """The bundled screenshots are the fixture pages, marked where the metadata says."""

    def setUp(self):
        with open(screens.META, encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.shots = screens.presets()

    def test_four_presets_from_the_three_fixture_pages(self):
        self.assertEqual([s.key for s in self.shots], ["catalog", "directory", "lookup", "done"])
        self.assertEqual([s.label for s in self.shots],
                         ["Open the result", "Filter first", "Pick a topic", "Already done"])
        self.assertEqual(set(self.meta["fixtures"]), {"catalog.html", "directory.html", "lookup.html", "style.css"})
        for digest in self.meta["fixtures"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual({p["page"] for p in self.meta["presets"]}, {"catalog.html", "directory.html", "lookup.html"})

    def test_the_tasks_are_the_browser_tests_own(self):
        import build_screens

        tasks = {p["key"]: p["task"] for p in self.meta["presets"]}
        self.assertEqual(tasks["catalog"], build_screens.CATALOG)
        self.assertEqual(tasks["done"], build_screens.CATALOG)
        self.assertEqual(tasks["directory"], build_screens.DIRECTORY)
        self.assertEqual(tasks["lookup"], build_screens.LOOKUP)

    def test_each_png_is_small_and_matches_its_digest(self):
        for p, shot in zip(self.meta["presets"], self.shots):
            with self.subTest(p["key"]):
                path = os.path.join(screens.FOLDER, p["image"])
                raw = open(path, "rb").read()
                self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertLess(len(raw), 300 * 1024)
                self.assertEqual(len(raw), p["bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), p["sha256"])
                with Image.open(path) as im:
                    self.assertEqual(list(im.size), p["size"])
                    self.assertEqual(im.size, (1440, 864))
                    self.assertEqual((im.width % screens.FACTOR, im.height % screens.FACTOR), (0, 0))
                self.assertLessEqual(1440 * 864, blink.ImageLimits().max_pixels)
                self.assertEqual(shot.png, raw)

    def test_a_changed_png_is_refused(self):
        with open(os.path.join(screens.FOLDER, "catalog.png"), "rb") as fh:
            raw = fh.read()
        real_open = open

        def tampered(path, mode="r", *a, **k):
            if str(path).endswith("catalog.png") and "b" in mode:
                return io.BytesIO(raw + b"x")
            return real_open(path, mode, *a, **k)

        screens.presets.cache_clear()
        try:
            with mock.patch("builtins.open", tampered), self.assertRaises(screens.ScreenError):
                screens.presets()
        finally:
            screens.presets.cache_clear()

    def test_boxes_are_numbered_in_order_inside_the_image(self):
        for shot in self.shots:
            with self.subTest(shot.key):
                W, H = shot.size
                self.assertEqual([b.n for b in shot.boxes], list(range(1, len(shot.boxes) + 1)))
                self.assertGreaterEqual(len(shot.boxes), 2)
                for b in shot.boxes:
                    x1, y1, x2, y2 = b.box
                    self.assertTrue(0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, b)
                    tx1, ty1, tx2, ty2 = b.tag
                    self.assertTrue(0 <= tx1 < tx2 <= W and 0 <= ty1 < ty2 <= H, b)
                    self.assertIn(b.at, ("above", "left", "inside"))
                    self.assertTrue(b.role and b.name)

    def test_no_number_covers_another_box(self):
        s = screens.style(1440)
        for shot in self.shots:
            pads = [(b.box[0] - s["stroke"], b.box[1] - s["stroke"], b.box[2] + s["stroke"], b.box[3] + s["stroke"])
                    for b in shot.boxes]
            for b in shot.boxes:
                if b.at == "inside":
                    continue
                for other, pad in zip(shot.boxes, pads):
                    with self.subTest(shot=shot.key, tag=b.n, box=other.n):
                        self.assertFalse(screens._overlaps(b.tag, pad))

    def test_the_marks_are_burned_in_where_the_metadata_says(self):
        """Pixels, not just numbers: each number's plate and each box's stroke are blink's cobalt."""
        s = screens.style(1440)
        for shot in self.shots:
            for b in shot.boxes:
                with self.subTest(shot=shot.key, n=b.n):
                    tx1, ty1, tx2, ty2 = b.tag
                    self.assertEqual(pixel(shot.png, tx1 + 1, ty1 + 1), BLUE)
                    self.assertEqual(pixel(shot.png, tx2 - 2, ty2 - 2), BLUE)
                    x1, y1, x2, y2 = b.box
                    mid = (y1 + y2) // 2
                    self.assertEqual(pixel(shot.png, x2 + s["stroke"] // 2, mid), BLUE)
                    self.assertEqual(pixel(shot.png, x2 + s["stroke"] + s["halo"] // 2, mid), screens.WHITE)

    def test_the_expected_answers_name_real_options(self):
        for shot in self.shots:
            expect = dict(shot.expect)
            self.assertIn(expect.get("done"), ("yes", "no"))
            self.assertEqual(expect.get("risky"), "no")  # none of the fixture pages can buy, send or delete
            if "element" in expect:
                self.assertIn(expect["element"], screens.questions(len(shot.boxes))["element"]["criteria"])
        self.assertEqual(dict(screens.preset("done").expect)["done"], "yes")

    def test_the_metadata_carries_no_local_paths(self):
        with open(screens.META, encoding="utf-8") as fh:
            text = fh.read()
        for bad in ("/" + "home/", "/" + "nvme", "runs/", "\\\\"):  # split so the public strict scan passes
            self.assertNotIn(bad, text)


class TestRequest(unittest.TestCase):
    """What the page sends: the evaluated screenshot request's shape, with one inline PNG."""

    def setUp(self):
        self.shot = screens.preset("directory")

    def test_the_state_is_the_task_a_note_and_the_image(self):
        st = screens.state(self.shot)
        self.assertEqual(list(st), ["task", "previous_actions", "screenshot_note", "screenshot"])
        self.assertEqual(st["task"], "Filter workshops to Remote and Morning only, then open Blue Workshop.")
        self.assertEqual(st["previous_actions"], [])
        self.assertEqual(st["screenshot_note"], "Use the attached screenshot with boxes 1-4.")
        self.assertTrue(st["screenshot"].startswith("data:image/png;base64,"))
        self.assertEqual(st["screenshot"], self.shot.uri)

    def test_the_questions_are_the_measured_wording(self):
        qs = screens.questions(4)
        self.assertEqual(list(qs), ["element", "done", "risky"])
        self.assertEqual(qs["element"]["type"], "choice")
        self.assertEqual(qs["element"]["criteria"], {"1": "box 1", "2": "box 2", "3": "box 3", "4": "box 4"})
        self.assertEqual((qs["done"]["type"], qs["risky"]["type"]), ("noul", "noul"))
        blink.validate(qs)
        # the element and done wording is cua/build_requests.py's, word for word
        path = os.path.join(HERE, "..", "cua", "build_requests.py")
        if not os.path.exists(path):
            self.skipTest("cua/build_requests.py is not in this checkout")
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn(f'INSTRUCTION = "{qs["element"]["instructions"]}"', src)
        self.assertIn(f'DONE = "{qs["done"]["instructions"]}"', src)
        self.assertIn(json.dumps(qs["done"]["criteria"]["true"]), src)

    def test_the_options_read_as_their_numbers(self):
        items = blink.question_options(screens.questions(3)["element"])
        self.assertEqual(items, [("1", "box 1"), ("2", "box 2"), ("3", "box 3")])
        grid = blink.question_options(screens.questions(9, grid=3)["element"])
        self.assertEqual(grid[-1], ("9", "cell 9"))
        self.assertIn("cell", screens.questions(9, grid=3)["element"]["instructions"])

    def test_every_call_gets_its_own_questions(self):
        a = screens.questions(3)
        a["element"]["criteria"]["1"] = "changed"
        a["done"]["criteria"]["true"] = "changed"
        self.assertEqual(screens.questions(3)["element"]["criteria"]["1"], "box 1")
        self.assertEqual(screens.DONE_CRITERIA["true"], "No more element action is needed")

    def test_blink_finds_one_image_and_reads_it_unresized(self):
        st, _ = screens.request(self.shot)
        found = blink.inspect_images(st)
        self.assertIsNotNone(found)
        self.assertEqual(found.loc, ["body", "state", "screenshot"])
        lifted = blink.extract_images(st)
        self.assertEqual(len(lifted.images), 1)
        self.assertEqual(lifted.images[0].size, (1440, 864))  # read exactly, never resampled
        self.assertEqual(lifted.state["screenshot"], "[image 1]")
        self.assertEqual(lifted.state["task"], st["task"])

    def test_the_replay_key_follows_the_task_and_the_pixels(self):
        key = screens.request_key(self.shot)
        self.assertEqual(key, screens.request_key(screens.preset("directory")))
        self.assertEqual(key, blink.request_key(*screens.request(self.shot)))
        self.assertNotEqual(key, screens.request_key(screens.with_task(self.shot, "Open Green Workshop.")))
        keys = {screens.request_key(s) for s in screens.presets()}
        self.assertEqual(len(keys), 4)

    def test_a_task_is_required_and_short(self):
        self.assertEqual(screens.task_problems(""), [screens.PROBLEM["task"]])
        self.assertEqual(screens.task_problems("   "), [screens.PROBLEM["task"]])
        self.assertEqual(screens.task_problems("x" * (screens.MAX_TASK + 1)), [screens.PROBLEM["long"]])
        self.assertEqual(screens.task_problems("Open it."), [])

    def test_the_mock_bias_points_at_the_expected_answer(self):
        self.assertEqual(screens.mock_bias(screens.preset("directory")),
                         {"element": {"2": 3.2}, "done": {"no": 3.2}, "risky": {"no": 3.2}})
        self.assertNotIn("element", screens.mock_bias(screens.preset("done")))


class TestMarks(unittest.TestCase):
    """Boxes and grids burned into a screenshot the way the presets are."""

    def test_mark_numbers_boxes_and_keeps_numbers_clear_of_them(self):
        with Image.open(png_file()) as im:
            boxes = [(60, 300, 380, 380), (420, 300, 740, 380), (60, 160, 620, 240)]
            marked, placed = screens.mark(im, boxes)
        self.assertEqual(marked.size, (1280, 800))
        self.assertEqual([b.n for b in placed], [1, 2, 3])
        self.assertEqual([b.box for b in placed], boxes)
        s = screens.style(1280)
        for b in placed:
            self.assertEqual(marked.getpixel((b.tag[0] + 1, b.tag[1] + 1)), BLUE)
            self.assertEqual(marked.getpixel((b.box[0] - s["stroke"] // 2 - 1, (b.box[1] + b.box[3]) // 2)), BLUE)
            if b.at != "inside":
                for other in placed:
                    self.assertFalse(screens._overlaps(b.tag, other.box))

    def test_the_drawing_surface_lays_out_what_mark_draws(self):
        """The page's own marks on an upload and the ones burned into what blink reads are one layout."""
        up = screens.load_upload(png_file())
        text = "60, 300, 380, 380\n420, 300, 740, 380"
        draft, problems = screens.draft_shot(up, text)
        shot, more = screens.upload_shot(up, "Save it.", text)
        self.assertEqual(problems + more, [])
        self.assertEqual(draft.boxes, shot.boxes)
        grid_draft, _ = screens.draft_shot(up, "", 4)
        grid_shot, _ = screens.upload_shot(up, "Save it.", "", 4)
        self.assertEqual(grid_draft.boxes, grid_shot.boxes)

    def test_a_grid_tiles_the_screenshot_left_to_right_then_down(self):
        for k in screens.GRIDS:
            cells = screens.grid_boxes(1283, 797, k)
            self.assertEqual(len(cells), k * k)
            self.assertEqual(sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in cells), 1283 * 797)
            self.assertEqual(cells[0][:2], (0, 0))
            self.assertEqual(cells[-1][2:], (1283, 797))
            self.assertLess(cells[0][0], cells[1][0])  # 2 is right of 1
            self.assertEqual(cells[k][0], 0)  # k + 1 starts the next row
        with self.assertRaises(screens.ScreenError):
            screens.grid_boxes(100, 100, 5)

    def test_box_lines_parse_in_the_screenshots_own_pixels(self):
        boxes, problems = screens.parse_boxes("10, 20, 110, 70\n(300 40 200 90)\n\n 5;5;60;60 ", 400, 300)
        self.assertEqual(problems, [])
        self.assertEqual(boxes, [(10, 20, 110, 70), (200, 40, 300, 90), (5, 5, 60, 60)])
        clamped, _ = screens.parse_boxes("-20, -5, 900, 900", 400, 300)
        self.assertEqual(clamped, [(0, 0, 400, 300)])
        _, problems = screens.parse_boxes("10, 20, 110\nhello\n1, 1, 3, 3", 400, 300)
        self.assertEqual(problems, [screens.PROBLEM["line"].format(n=1), screens.PROBLEM["line"].format(n=2),
                                    screens.PROBLEM["small"].format(n=3)])
        many = "\n".join(f"{i}, 0, {i + 10}, 10" for i in range(0, 10 * (screens.MAX_BOXES + 1), 10))
        _, problems = screens.parse_boxes(many, 1000, 100)
        self.assertIn(screens.PROBLEM["many"], problems)

    def test_a_drawn_box_is_checked_before_it_is_kept(self):
        self.assertEqual(screens.add_box("", [30.4, 60, 10, 20], 100, 100), "10, 20, 30, 60")
        self.assertEqual(screens.add_box("1, 1, 9, 9", [20, 20, 40, 40], 100, 100), "1, 1, 9, 9\n20, 20, 40, 40")
        self.assertEqual(screens.add_box("", [0, 0, 500, 500], 100, 100), "0, 0, 100, 100")
        for bad in (None, "x", [1, 2, 3], [0, 0, float("nan"), 5], [1, 1, 3, 3], ["a", 1, 2, 3]):
            with self.subTest(bad=bad):
                self.assertEqual(screens.add_box("5, 5, 50, 50", bad, 100, 100), "5, 5, 50, 50")
        full = "\n".join(["0, 0, 10, 10"] * screens.MAX_BOXES)
        self.assertEqual(screens.add_box(full, [20, 20, 40, 40], 100, 100).count("\n"), screens.MAX_BOXES - 1)
        self.assertEqual(screens.drop_box("a\nb\nc", 2), "a\nc")
        self.assertEqual(screens.drop_box("a\nb", 9), "a\nb")
        self.assertEqual(screens.drop_box("a\nb", "x"), "a\nb")


class TestUpload(unittest.TestCase):
    """A visitor's screenshot is held to blink's own image limits and marked like a preset."""

    def test_png_jpeg_and_webp_are_read(self):
        for kind in ("PNG", "JPEG", "WEBP"):
            with self.subTest(kind):
                up = screens.load_upload(png_file(640, 400, kind))
                self.assertEqual((up.size, up.source_size), ((640, 384), (640, 400)))  # blink's 32-pixel rule
                self.assertTrue(up.png.startswith(b"\x89PNG"))
                self.assertTrue(up.view.startswith("data:image/jpeg;base64,"))

    def test_what_cannot_be_read_says_what_to_change(self):
        cases = [(png_file(64, 64, "GIF"), "format"), (png_file(6000, 4000), "pixels"), (png_file(4000, 10), "shape")]
        noise = tempfile.mkstemp(suffix=".png")[1]
        with open(noise, "wb") as fh:
            fh.write(b"not an image")
        cases += [(noise, "format"), ("", "none"), ("/nonexistent/shot.png", "format")]
        for path, why in cases:
            with self.subTest(why=why, path=path), self.assertRaises(screens.ScreenError) as caught:
                screens.load_upload(path)
            self.assertEqual(str(caught.exception), screens.PROBLEM[why])
        with mock.patch.object(os.path, "getsize", return_value=screens.UPLOAD_BYTES + 1):
            with self.assertRaises(screens.ScreenError) as caught:
                screens.load_upload(png_file(64, 64))
        self.assertEqual(str(caught.exception), screens.PROBLEM["bytes"])

    def test_a_large_screenshot_is_shrunk_to_blinks_pixel_budget(self):
        up = screens.load_upload(png_file(2880, 1800))
        self.assertEqual(up.source_size, (2880, 1800))
        self.assertLessEqual(up.size[0] * up.size[1], screens.WORK_PIXELS)
        self.assertLess(abs(up.size[0] / up.size[1] - 1.6), 0.05)
        self.assertEqual((up.size[0] % screens.FACTOR, up.size[1] % screens.FACTOR), (0, 0))

    def test_blink_reads_the_marked_upload_pixel_for_pixel(self):
        for w, h in ((2880, 1800), (1366, 768), (390, 844), (100, 100)):
            with self.subTest(size=(w, h)):
                up = screens.load_upload(png_file(w, h))
                shot, _ = screens.upload_shot(up, "Save it.", "")
                read = blink.extract_images(screens.state(shot)).images[0]
                self.assertEqual(read.size, up.size)
                with Image.open(io.BytesIO(shot.png)) as marked:
                    self.assertEqual(read.tobytes(), marked.convert("RGB").tobytes())

    def test_no_boxes_means_a_numbered_grid(self):
        up = screens.load_upload(png_file())
        for k in screens.GRIDS:
            shot, problems = screens.upload_shot(up, "Save it.", "", k)
            self.assertEqual(problems, [])
            self.assertEqual((shot.grid, len(shot.boxes)), (k, k * k))
            qs = screens.questions(len(shot.boxes), shot.grid)
            self.assertEqual(len(qs["element"]["criteria"]), k * k)
            self.assertIn(f"{k}x{k} grid", screens.note(shot))
        shot, _ = screens.upload_shot(up, "Save it.", "", 7)
        self.assertEqual(shot.grid, screens.GRIDS[0])

    def test_boxes_typed_in_source_pixels_are_scaled_to_what_blink_reads(self):
        up = screens.load_upload(png_file(2880, 1800))
        shot, problems = screens.upload_shot(up, "Save it.", "0, 0, 1440, 900\n1440, 900, 2880, 1800")
        self.assertEqual(problems, [])
        W, H = up.size
        self.assertEqual(shot.boxes[0].box, (0, 0, round(W / 2), round(H / 2)))
        self.assertEqual(shot.boxes[1].box[2:], (W, H))
        self.assertEqual(shot.source_size, (2880, 1800))
        self.assertNotEqual(shot.png, up.png)  # the marks are burned in
        lifted = blink.extract_images(screens.state(shot))
        self.assertLessEqual(lifted.images[0].width * lifted.images[0].height, blink.ImageLimits().max_pixels)

    def test_one_box_or_a_bad_line_is_refused(self):
        up = screens.load_upload(png_file())
        self.assertEqual(screens.upload_shot(up, "t", "0, 0, 50, 50")[1], [screens.PROBLEM["few"]])
        self.assertEqual(screens.upload_shot(up, "t", "0, 0, 50")[1], [screens.PROBLEM["line"].format(n=1)])
        self.assertEqual(screens.upload_shot(None, "t")[1], [screens.PROBLEM["none"]])

    def test_the_thumbnail_is_what_blink_read(self):
        up = screens.load_upload(png_file())
        shot, _ = screens.upload_shot(up, "Save it.", "")
        self.assertTrue(screens.thumb(shot).startswith("data:image/jpeg;base64,"))


class _Ids:
    def __init__(self, n):
        self.shape = (1, n)


class _Live:
    """A stand-in for blink-mimo-9b's torch engine: renders like it, never touches a device. The GPU
    call itself is blink._forward_images, which the tests replace with `gpu` below."""

    name, temperature, accepts_images, model_id, key = "torch", 1.0, True, MIMO, "fake-live"
    image_limits = blink.ImageLimits()

    def __init__(self):
        self.gpu_calls = []  # one entry per GPU call: the questions it carried

    def render_images(self, request, questions):
        if not self.accepts_images:
            raise blink.ImageError("this model reads text only", request.loc)
        pixels = request.images[0].width * request.images[0].height
        return [{"qkey": q, "keys": [k for k, _ in blink.question_options(spec)], "inputs": {"input_ids": _Ids(433)},
                 "cand": [], "image_pixels": pixels, "visual_tokens": pixels // 1024} for q, spec in questions.items()]

    def gpu(self, key, work):
        self.gpu_calls.append([item["qkey"] for item in work])
        return [[float(i == 1) for i in range(len(item["keys"]))] for item in work], 12.5, 4.0, 8.5


class _Replay:
    name, temperature, hardware = "replay", 1.0, ""

    def __init__(self, model, cache):
        self.model_id, self.cache = model, cache


def recording(shot: screens.Shot, **extra) -> dict:
    qs = screens.questions(len(shot.boxes), shot.grid)
    logits = {q: [0.0] * len(blink.question_options(s)) for q, s in qs.items()}
    logits["element"][1] = 3.0
    logits["done"] = [-2.0, 2.0]
    return {screens.request_key(shot): {"names": [f"screen/{shot.key}"], "logits": logits, "input_tokens": 1301,
                                        "latency_ms": 431.5, **extra}}


class TestDecide(unittest.TestCase):
    """blink-mimo-9b answers every screenshot, from a recording, the mock or the live model."""

    def setUp(self):
        self.shot = screens.preset("directory")

    def test_only_blink_mimo_9b_reads_screenshots(self):
        with serving(FOUR, MIMO):
            self.assertEqual(screens.vision_model(), MIMO)
            self.assertTrue(screens.available())
        with serving(MIMO):
            self.assertEqual(screens.vision_model(), MIMO)
        with serving(FOUR):
            self.assertIsNone(screens.vision_model())
            self.assertFalse(screens.available())
            with self.assertRaises(screens.ScreenError):
                screens.decide(self.shot)
        with serving(FOUR, "Qwen/Qwen3.5-4B"):
            self.assertIsNone(screens.vision_model())  # the drafter's base is never the screen model

    def test_mock_mode_runs_the_real_request_end_to_end(self):
        with serving(FOUR, MIMO):
            out = screens.decide(self.shot)
            self.assertEqual(out["meta"]["model"], MIMO)  # not the page's default blink-4b
            self.assertEqual(out["meta"]["engine"], "mock")
            self.assertEqual(out["meta"]["image_pixels"], 1440 * 864)
            self.assertEqual(out["meta"]["visual_tokens"], 1440 * 864 // 1024)
            answers = out["answers"]
            self.assertEqual(set(answers), {"element", "done", "risky"})
            self.assertAlmostEqual(sum(answers["element"]["probabilities"].values()), 1.0, places=6)
            self.assertEqual(answers["element"]["choice"], "2")  # shaped by the preset's expectation
            self.assertLess(answers["done"]["noul"], 0.5)
            done = screens.decide(screens.preset("done"))
            self.assertGreater(done["answers"]["done"]["noul"], 0.5)
            with mock.patch.object(blink, "_decode_image", side_effect=blink.ImageError("bad", ["x"])):
                with self.assertRaises(blink.ImageError):
                    screens.decide(self.shot)  # the image is decoded and checked even in mock mode

    def test_a_text_model_would_refuse_the_screenshot(self):
        with serving(FOUR, MIMO):
            with self.assertRaisesRegex(blink.ImageError, "reads text only"):
                blink.decide(*screens.request(self.shot), model=FOUR)

    def test_the_space_never_grafts_a_tower_onto_blink_4b(self):
        """BLINK_VISION=1 turns on MiMo's own tower and nothing else: blink-4b is built text-only."""
        built = {}

        class Recorder:
            def __init__(self, model_id, revision=None, temperature=1.0, **options):
                built[model_id] = options

        with serving(FOUR, MIMO), mock.patch.object(blink, "VISION", True), \
                mock.patch.object(blink, "VISION_TOWER", None), \
                mock.patch.object(blink, "engine_kind", return_value="torch"), \
                mock.patch.object(blink, "TorchEngine", Recorder):
            blink._build(FOUR)
            blink._build(MIMO)
        self.assertEqual(built[FOUR], {})
        self.assertEqual(built[MIMO], {"vision": True, "vision_tower": None})
        stage_space = _stage_space()

        for name in stage_space.FILES:
            if name == "blink.py":
                continue  # it reads the setting; nothing the Space ships sets it
            with open(os.path.join(HERE, name), encoding="utf-8") as fh:
                self.assertNotIn("BLINK_VISION_TOWER", fh.read(), name)

    def test_a_saved_run_answers_the_first_render_and_says_so(self):
        with serving(FOUR, MIMO), mock.patch.object(blink, "_forward_images", (live := _Live()).gpu):
            blink._ENGINES[MIMO] = blink.HybridEngine(_Replay(MIMO, recording(self.shot, image_pixels=1244160,
                                                                              visual_tokens=1215)), live)
            out = screens.decide(self.shot, prefer="saved")
            self.assertEqual((out["meta"]["engine"], out["meta"]["latency_ms"]), ("replay", 431.5))
            self.assertEqual((out["meta"]["image_pixels"], out["meta"]["visual_tokens"]), (1244160, 1215))
            self.assertEqual(out["answers"]["element"]["choice"], "2")
            self.assertLess(out["answers"]["done"]["noul"], 0.1)  # the recording says no
            self.assertEqual(live.gpu_calls, [])
            # every click after that runs the model
            again = screens.decide(self.shot)
            self.assertEqual((again["meta"]["engine"], len(live.gpu_calls)), ("torch", 1))
            self.assertEqual(again["meta"]["model_ms"], 12.5)
            # an edited task has no saved run, and the first render never runs live
            with self.assertRaises(screens.ScreenMiss):
                screens.decide(screens.with_task(self.shot, "Open Green Workshop."), prefer="saved")
            self.assertEqual(len(live.gpu_calls), 1)

    def test_one_gpu_call_carries_every_question(self):
        """On ZeroGPU each call attaches a device: all three questions go in one, each still its own forward."""
        with serving(FOUR, MIMO), mock.patch.object(blink, "_forward_images", (live := _Live()).gpu):
            blink._ENGINES[MIMO] = blink.HybridEngine(None, live)
            out = screens.decide(self.shot)
        self.assertEqual(live.gpu_calls, [["element", "done", "risky"]])
        qs = screens.questions(len(self.shot.boxes))
        for qkey, spec in qs.items():
            keys = [k for k, _ in blink.question_options(spec)]
            alone = blink.answer_for(spec, keys, blink.softmax([float(i == 1) for i in range(len(keys))], 1.0))
            self.assertEqual(out["answers"][qkey], alone)
        meta = out["meta"]
        self.assertEqual((meta["model"], meta["engine"], meta["generated_tokens"]), (MIMO, "torch", 0))
        self.assertEqual((meta["input_tokens"], meta["model_ms"]), (3 * 433, 12.5))
        self.assertEqual((meta["image_pixels"], meta["visual_tokens"]), (1440 * 864, 1440 * 864 // 1024))

    def test_replay_only_hosting_answers_from_recordings_alone(self):
        with serving(FOUR, MIMO):
            blink._ENGINES[MIMO] = _Replay(MIMO, recording(self.shot))
            self.assertEqual(screens.decide(self.shot)["meta"]["engine"], "replay")
            with self.assertRaises(screens.ScreenMiss):
                screens.decide(screens.preset("lookup"))

    def test_the_verdict_follows_the_rule(self):
        def a(element, done, risky, choice="2"):
            probs = {"1": 1 - element, "2": element}
            return {"element": {"choice": choice, "probabilities": probs},
                    "done": {"noul": done}, "risky": {"noul": risky}}

        self.assertEqual(screens.verdict(a(0.9, 0.1, 0.1))[:3], ("go", "click", "Click 2"))
        self.assertEqual(screens.verdict(a(0.9, 0.7, 0.1))[:3], ("go", "done", "Task looks done"))
        self.assertEqual(screens.verdict(a(0.9, 0.1, 0.6))[:3], ("hold", "ask", "Ask before clicking 2"))
        self.assertEqual(screens.verdict(a(0.4, 0.1, 0.1, "1"))[1], "click")
        self.assertEqual(screens.verdict({"element": {"choice": "1", "probabilities": {"1": .4, "2": .3, "3": .3}},
                                          "done": {"noul": .1}, "risky": {"noul": .1}})[:3],
                         ("hold", "unsure", "Let a person take over"))


class TestScreenHtml(unittest.TestCase):
    """The card: marks exactly on the boxes, the choice lit, the saved run labelled."""

    def setUp(self):
        import ui

        self.ui = ui
        self.shot = screens.preset("directory")

    def answered(self, shot=None, **meta):
        shot = shot or self.shot
        with serving(FOUR, MIMO):
            out = screens.decide(shot)
        out["meta"].update(meta)
        return out

    def test_every_mark_sits_on_its_box(self):
        html = self.ui.screen_stage(self.shot)
        W, H = self.shot.size
        s = screens.style(W)
        pad = s["stroke"] + s["halo"]
        marks = re.findall(r'class="blk-som[^"]*" data-n="(\d+)" style="--sx:([\d.]+)%;--sy:([\d.]+)%;'
                           r'--sw:([\d.]+)%;--sh:([\d.]+)%', html)
        self.assertEqual([int(m[0]) for m in marks], [1, 2, 3, 4])
        for (_n, x, y, w, h), b in zip(marks, self.shot.boxes):
            self.assertAlmostEqual(float(x), (b.box[0] - pad) / W * 100, places=2)
            self.assertAlmostEqual(float(y), (b.box[1] - pad) / H * 100, places=2)
            self.assertAlmostEqual(float(w), (b.box[2] - b.box[0] + 2 * pad) / W * 100, places=2)
            self.assertAlmostEqual(float(h), (b.box[3] - b.box[1] + 2 * pad) / H * 100, places=2)
        tags = re.findall(r'class="blk-somtag (\w+)[^"]*" data-n="(\d+)"', html)
        self.assertEqual(tags, [(b.at, str(b.n)) for b in self.shot.boxes])
        self.assertEqual(html.count(';--tc:1"'), 4)  # one character each, so the page can keep it inside
        self.assertIn(f'aspect-ratio:{W} / {H}', html)
        self.assertIn('data-draw="0"', html)
        self.assertIn(self.shot.uri, html)
        self.assertNotIn("blk-shot-pill", html)

    def test_the_answer_glows_each_mark_and_lights_the_choice(self):
        out = self.answered()
        html = self.ui.screen_panel(self.shot, out, rev=3)
        choice = out["answers"]["element"]["choice"]
        self.assertIn(f'class="blk-som win" data-n="{choice}"', html)
        self.assertEqual(html.count("blk-som win"), 1)
        for n, p in out["answers"]["element"]["probabilities"].items():
            self.assertIn(f";--sp:{p:.4f}", html)
            self.assertIn(f"<i>{p:.0%}</i>", html)
            chars = len(n) + len(f"{p:.0%}")  # the number, then its share
            self.assertRegex(html, rf'class="blk-somtag[^"]*" data-n="{n}" style="[^"]*;--tc:{chars};--tg:4px"')
        self.assertIn('data-act="click"', html)
        self.assertIn(f"Click {choice}", html)
        self.assertIn("checkbox \u00b7 Morning only", html)  # the element's own name, from the page
        self.assertEqual(html.count('class="blk-ring"'), 2)
        self.assertIn(htmllib.escape(self.ui.SCREEN["done"]), html)
        self.assertIn(self.ui.SCREEN["risky"], html)
        self.assertIn("Mock mode", html)
        self.assertIn('data-rev="3"', html)
        self.assertIn("image tokens", html)

    def test_a_finished_task_lights_nothing(self):
        done = screens.preset("done")
        html = self.ui.screen_panel(done, self.answered(done))
        self.assertIn('data-act="done"', html)
        self.assertNotIn("blk-som win", html)
        self.assertIn("Task looks done", html)

    def test_a_saved_run_is_labelled_as_one(self):
        out = self.answered(engine="replay", latency_ms=431.5)
        html = self.ui.screen_panel(self.shot, out)
        self.assertIn("431.5</b> ms recorded call \u00b7 saved run", html)
        self.assertIn(f'<em class="blk-tag blk-saved">{self.ui.SCREEN["saved"]}</em>', html)
        live = self.ui.screen_panel(self.shot, self.answered())
        self.assertNotIn("blk-saved", live)

    def test_the_raw_response_leaves_the_pixels_out(self):
        html = self.ui.screen_panel(self.shot, self.answered())
        raw = htmllib.unescape(html.split("<summary>Raw response</summary>")[1].split("</details>")[0])
        self.assertIn("data:image/png;base64,\u2026 (", raw)
        self.assertLess(len(raw), 12_000)
        self.assertIn('"screenshot_note": "Use the attached screenshot with boxes 1-4."', raw)
        self.assertNotIn(self.ui.SCREEN["read"], html)  # a preset's image is already on screen

    def test_waiting_keeps_the_screenshot_and_says_so(self):
        html = self.ui.screen_panel(self.shot, busy=True)
        self.assertIn('class="blk-scr busy"', html)
        self.assertIn('class="blk-shot busy"', html)
        self.assertIn('aria-busy="true"', html)
        self.assertEqual(html.count('class="blk-meter empty"'), 2)  # the rings wait, empty, where they fill in
        self.assertNotIn("%</b>", html)
        self.assertIn(self.shot.uri, html)
        self.assertIn('data-shot="directory"', html)

    def test_an_upload_draws_and_shows_what_blink_read(self):
        up = screens.load_upload(png_file())
        draft, _ = screens.draft_shot(up)
        html = self.ui.screen_panel(draft, draw=True)
        self.assertIn('data-draw="1"', html)
        self.assertEqual(html.count('class="blk-som cell"'), 9)
        self.assertIn(self.ui.SCREEN["grid_pill"].format(k=3), html)
        self.assertIn('data-w="1280" data-h="800"', html)
        self.assertIn(up.view, html)
        shot, _ = screens.upload_shot(up, "Save it.", "60, 300, 380, 380\n420, 300, 740, 380")
        answered = self.ui.screen_panel(shot, self.answered(shot), draw=True)
        self.assertIn(self.ui.SCREEN["read"], answered)
        self.assertIn('class="blk-read"', answered)
        self.assertNotIn(shot.uri, answered)  # the page shows the light copy, not the full PNG

    def test_the_task_is_escaped(self):
        shot = screens.with_task(self.shot, '<img src=x onerror="alert(1)">')
        html = self.ui.screen_panel(shot, self.answered(shot))
        self.assertNotIn('<img src=x', html)
        self.assertIn("&lt;img src=x", html)

    def test_what_goes_wrong_is_said_in_the_card(self):
        with serving(FOUR, MIMO), mock.patch.object(blink, "_forward_images", _Live().gpu):
            blink._ENGINES[MIMO] = _Replay(MIMO, {})
            self.assertIn(self.ui.SCREEN["miss"], self.ui.run_screen(self.shot))
            blink._ENGINES[MIMO] = blink.HybridEngine(None, _Live())
            self.assertIn(self.ui.SCREEN["not_run"], self.ui.run_screen(self.shot, prefer="saved"))
            text_only = _Live()
            text_only.accepts_images = False
            blink._ENGINES[MIMO] = blink.HybridEngine(None, text_only)
            self.assertIn(self.ui.SCREEN["off"], self.ui.run_screen(self.shot))
            self.assertFalse(screens.accepts_images())
            blink._ENGINES[MIMO] = blink.HybridEngine(None, _Live())
            self.assertIn("blk-shot answered", self.ui.run_screen(self.shot))
            self.assertTrue(screens.accepts_images())

    def test_the_model_chip_names_blink_mimo_9b(self):
        with serving(FOUR, MIMO):
            chip = self.ui.screen_model_chip()
        self.assertIn("Screenshots use blink-mimo-9b", chip)
        self.assertIn("<svg", chip)

    def test_the_copy_is_short_and_in_the_spaces_voice(self):
        first_person = re.compile(r"\b(?:we|our|ours|us|ourselves)\b", re.IGNORECASE)
        words = list(self.ui.SCREEN.values()) + list(screens.PROBLEM.values()) + [s.label for s in screens.presets()]
        words += [screens.RISKY_ASK, *screens.RISKY_CRITERIA.values()]
        for text in words:
            with self.subTest(text=text):
                self.assertIsNone(first_person.search(text))
                for claim in ("TypeSafe", "Jev", "accuracy", "JevBench"):
                    self.assertNotIn(claim, text)
        for key in ("blurb", "model", "upload", "task", "done", "risky", "saved", "not_run", "off"):
            self.assertLessEqual(len(self.ui.SCREEN[key]), 60, key)


class TestScreenTab(unittest.TestCase):
    """The tab exists where blink-mimo-9b is served, and the page's masthead follows it there."""

    def setUp(self):
        need_gradio(self)
        self.addCleanup(reload_app)

    def test_the_tab_is_there_only_with_a_model_that_reads_screens(self):
        import gradio as gr

        with serving(FOUR, MIMO):
            app = reload_app()
            self.assertEqual(app.ui.CASE_IDS[-1], "screen")
            self.assertEqual(app.ui.CASE_IDS[:-1], tuple(c.key for c in app.examples.USE_CASES))
            self.assertEqual(app.ui.parse_deep_link("?tab=use-cases&case=screen"), ("use-cases", "screen"))
            self.assertEqual(app.picked_case(SimpleNamespace(value="Screen click", index=None)), "screen")
            demo = app.build()
            try:
                ids = [b.id for b in demo.blocks.values() if isinstance(b, gr.Tab)]
                names = {getattr(fn.fn, "__name__", "") for fn in demo.fns.values()}
                events = {t[1] for fn in demo.fns.values() for t in (fn.targets or [])}
            finally:
                demo.close()
            self.assertIn("screen", ids)
            self.assertEqual(ids.index("screen"), ids.index("nextclick") + 1)
            self.assertLessEqual({"screen_run", "load_upload", "drawn", "dropped", "redraw"}, names)
            self.assertLessEqual({"draw", "drop", "upload"}, events)
        with serving(FOUR):
            app = reload_app()
            self.assertNotIn("screen", app.ui.CASE_IDS)
            self.assertEqual(app.ui.parse_deep_link("?tab=use-cases&case=screen"), ("use-cases", None))
            demo = app.build()
            try:
                ids = [b.id for b in demo.blocks.values() if isinstance(b, gr.Tab)]
            finally:
                demo.close()
            self.assertNotIn("screen", ids)

    def test_how_it_works_says_where_the_vision_tower_is_used(self):
        with serving(FOUR, MIMO):
            app = reload_app()
            how = "".join(app.ui.how_blocks())
            self.assertIn("Screen click uses that tower for screenshots. Every other tab uses the text side. "
                          "Either way, blink reads option-letter scores and generates no text.", how)
            self.assertNotIn("text side only", how)
        with serving(FOUR):
            app = reload_app()
            self.assertIn("The app uses its text side only", "".join(app.ui.how_blocks()))
        with open(os.path.join(HERE, "README.md"), encoding="utf-8") as fh:
            card = fh.read()
        self.assertIn("Screen click uses that tower for screenshots. Everything else uses the text side.", card)
        self.assertNotIn("The app uses its text side only", card)

    def test_the_masthead_names_the_model_answering_the_panel_in_view(self):
        with serving(FOUR, MIMO):
            app = reload_app()
            on_screen = app.masthead_for("use-cases", "screen", FOUR)
            self.assertIn(MIMO, on_screen)
            self.assertIn("MiMo-V2.6-Distill-Qwen-9B", on_screen)
            self.assertIn(FOUR, app.masthead_for("use-cases", "support", FOUR))
            self.assertIn(FOUR, app.masthead_for("playground", "screen", FOUR))
            self.assertIn(FOUR, app.masthead_for("use-cases", "", FOUR))
            self.assertEqual(app.masthead_for("ask", "", FOUR), app.ui.masthead(FOUR, drafting=True))

    def test_the_request_on_screen_is_what_runs(self):
        with serving(FOUR, MIMO):
            app = reload_app()
            shot, problems = app.screen_request("lookup", None, "Open Parks.", "", "3")
            self.assertEqual((shot.key, shot.task, problems), ("lookup", "Open Parks.", []))
            _, problems = app.screen_request("lookup", None, "  ", "", "3")
            self.assertEqual(problems, [screens.PROBLEM["task"]])
            up = screens.load_upload(png_file())
            shot, problems = app.screen_request("", up, "Save it.", "", "4")
            self.assertEqual((shot.grid, len(shot.boxes), problems), (4, 16, []))
            shot, problems = app.screen_request("", up, "", "0, 0, 9", "3")
            self.assertEqual(problems, [screens.PROBLEM["task"], screens.PROBLEM["line"].format(n=1)])
            self.assertEqual(app._grid("4"), 4)
            self.assertEqual(app._grid("nope"), 3)
            self.assertEqual(app._grid(None), 3)

    def test_the_drawing_script_only_speaks_through_its_own_events(self):
        with serving(FOUR, MIMO):
            app = reload_app()
        js = app.SCREEN_DRAW
        self.assertIn("trigger('draw'", js)
        self.assertIn("trigger('drop'", js)
        self.assertIn("element.addEventListener('pointerdown'", js)
        for missing in ("document.querySelector", "fetch(", "innerHTML", "textarea"):
            self.assertNotIn(missing, js)
        self.assertIn("blk-scr", app.SCREEN_BUSY)


class TestScreenRecording(unittest.TestCase):
    """record_replay.py records the presets as the page sends them, into blink-mimo-9b's file only."""

    def test_the_recorder_sends_what_the_page_sends(self):
        import record_replay

        reqs = record_replay.screen_requests()
        self.assertEqual([n for n, _, _ in reqs], [f"screen/{s.key}" for s in screens.presets()])
        for (_, st, qs), shot in zip(reqs, screens.presets()):
            self.assertEqual(blink.request_key(st, qs), screens.request_key(shot))
        bundled = {blink.request_key(s, q) for _, s, q in record_replay.bundled_requests()}
        self.assertFalse(bundled & {screens.request_key(s) for s in screens.presets()})

    def test_record_reads_screenshots_through_the_image_path(self):
        import record_replay

        live = _Live()
        live.logits = mock.Mock(side_effect=AssertionError("text path used for a screenshot"))
        with serving(MIMO), mock.patch.object(blink, "_forward_images", live.gpu), \
                mock.patch.object(blink, "decide", side_effect=AssertionError("not the Space's own path")):
            blink._ENGINE = live
            with contextlib.redirect_stdout(io.StringIO()):
                cache = record_replay.record(live, record_replay.screen_requests()[:2], repeats=2)
        # one read and two timed Decides per screenshot, each a single GPU call carrying all three questions
        self.assertEqual(live.gpu_calls, [["element", "done", "risky"]] * (2 * (1 + 2)))
        entry = cache[screens.request_key(screens.preset("catalog"))]
        self.assertEqual(entry["names"], ["screen/catalog"])
        self.assertEqual((entry["image_pixels"], entry["visual_tokens"]), (1440 * 864, 1440 * 864 // 1024))
        self.assertEqual((entry["model_ms"], entry["gpu_calls"], entry["input_tokens"]), (12.5, 1, 3 * 433))
        self.assertEqual(set(entry["logits"]), {"element", "done", "risky"})
        self.assertEqual(len(entry["logits"]["element"]), 3)

    def test_a_merge_keeps_text_runs_and_refuses_other_weights(self):
        import record_replay

        old = {"model": MIMO, "weights_sha256": "abc", "requests": {"t": {"names": ["text"]}}}
        merged = record_replay.merge(old, {"s": {"names": ["screen/x"]}}, MIMO, "abc", code="f00d")
        self.assertEqual(set(merged["requests"]), {"t", "s"})
        self.assertIn("screens_recorded_utc", merged)
        self.assertEqual(merged["screens_blink_py_sha256"], "f00d")  # which blink.py made the saved runs
        with open(blink.__file__, "rb") as fh:
            self.assertEqual(record_replay.file_digest(blink.__file__), hashlib.sha256(fh.read()).hexdigest())
        self.assertEqual(set(old["requests"]), {"t"})
        with self.assertRaises(SystemExit):
            record_replay.merge(old, {}, FOUR, "abc")
        with self.assertRaises(SystemExit):
            record_replay.merge(old, {}, MIMO, "other")

    def test_every_screen_preset_has_its_saved_run(self):
        with open(os.path.join(HERE, "replay-blink-mimo-9b.json"), encoding="utf-8") as fh:
            rec = json.load(fh)
        saved = rec["requests"]
        covered = [s.key for s in screens.presets() if screens.request_key(s) in saved]
        if SCREEN_REPLAY_PENDING:
            self.assertEqual(covered, [], "the screen presets are recorded now; set SCREEN_REPLAY_PENDING = False")
        else:
            self.assertEqual(covered, [s.key for s in screens.presets()])
            # made by the blink.py the Space ships, which staging checks again
            with open(os.path.join(HERE, "blink.py"), "rb") as fh:
                self.assertEqual(rec["screens_blink_py_sha256"], hashlib.sha256(fh.read()).hexdigest())
        with open(os.path.join(HERE, "replay.json"), encoding="utf-8") as fh:
            self.assertFalse({screens.request_key(s) for s in screens.presets()} & set(json.load(fh)["requests"]))

    def test_staging_ships_the_screenshots_as_bytes(self):
        stage_space = _stage_space()

        self.assertIn("screens.py", stage_space.FILES)
        self.assertIn("screens/screens.json", stage_space.FILES)
        self.assertEqual(stage_space.assets(), sorted(f"screens/{s.key}.png" for s in screens.presets()))
        self.assertTrue(all(not f.endswith(".png") for f in stage_space.FILES))
        pending = stage_space.unrecorded_screens([FOUR, MIMO], FOUR)
        self.assertEqual(pending, [s.key for s in screens.presets()] if SCREEN_REPLAY_PENDING else [])
        self.assertEqual(stage_space.unrecorded_screens([FOUR], FOUR), [])


class TestTextOnlyElsewhere(unittest.TestCase):
    """Only Screen click sends a screenshot. The Space's API, and every other tab, stay text-only
    even where blink-mimo-9b's vision tower is on."""

    QS = {"q": {"type": "noul", "instructions": "Is there a button?"}}

    def setUp(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        self.png = screens.preset("catalog").uri

    def vision_on(self):
        """blink-mimo-9b as the Space runs it with BLINK_VISION=1; its GPU call must never happen."""
        live = _Live()
        live.gpu = mock.Mock(side_effect=AssertionError("an image reached the model"))
        live.logits = lambda state, questions: (
            {q: [0.0] * len(blink.question_options(spec)) for q, spec in questions.items()}, 7)
        blink._ENGINES[MIMO] = blink.HybridEngine(None, live)
        return mock.patch.object(blink, "_forward_images", live.gpu)

    def refusal(self, caught) -> dict:
        return json.loads(str(caught.exception.message if hasattr(caught.exception, "message") else caught.exception))

    def test_the_api_refuses_an_image_with_serve_py_s_422_body(self):
        import gradio as gr

        with serving(FOUR, MIMO):
            app = reload_app()
            with self.vision_on():
                for state, loc in (
                    ({"task": "t", "screenshot": self.png}, ["body", "state", "screenshot"]),
                    (json.dumps({"page": {"shot": self.png}}), ["body", "state", "page", "shot"]),
                    (f"See {self.png}", ["body", "state"]),
                    ("data:image/png;foo=bar;base64,AAAA", ["body", "state"]),  # malformed is still an image
                ):
                    for model in (MIMO, FOUR, None):
                        with self.subTest(state=str(state)[:30], model=model), self.assertRaises(gr.Error) as caught:
                            app.systemone(state, self.QS, model=model)
                        body = self.refusal(caught)
                        self.assertEqual(body, {"error": "this Space's API reads text only",
                                                "detail": [{"loc": loc, "msg": "this Space's API reads text only",
                                                            "type": "value_error"}]})
                # text still answers, on either model
                for model in (FOUR, MIMO):
                    out = app.systemone("A plain sentence.", self.QS, model=model)
                    self.assertEqual(set(out), {"model", "answers", "usage", "meta"})

    def test_ask_refuses_an_image_before_anything_is_drafted(self):
        import gradio as gr

        with serving(FOUR, MIMO):
            app = reload_app()
            with self.vision_on(), mock.patch.object(app.author, "draft",
                                                     side_effect=AssertionError("drafted")) as draft:
                with self.assertRaises(gr.Error) as caught:
                    app.ask_api(f"Which button? {self.png}", model=MIMO)
                self.assertEqual(self.refusal(caught)["detail"][0]["loc"], ["body", "ask"])
                draft.assert_not_called()
            # a drafted state that picked up an image is refused as well
            drafted = {"state": {"screen": self.png}, "questions": self.QS, "author": {}}
            with self.vision_on(), mock.patch.object(app.author, "draft", return_value=drafted):
                with self.assertRaises(gr.Error) as caught:
                    app.ask_api("Which button?", model=MIMO)
                self.assertEqual(self.refusal(caught)["error"], "this Space's API reads text only")

    def test_the_other_tabs_send_no_screenshot(self):
        with serving(FOUR, MIMO):
            app = reload_app()
            ui = app.ui
            case = app.examples.CASES_BY_KEY["support"]
            with self.vision_on():
                for html in (ui.run_questions({"screen": self.png}, self.QS, MIMO),
                             ui.run_playground(json.dumps({"screen": self.png}), json.dumps(self.QS), MIMO),
                             ui.run_use_case(case, f"Ticket with a screenshot: {self.png}", MIMO)):
                    self.assertIn(ui.TEXT_ONLY_UI, htmllib.unescape(html))
                    self.assertNotIn("blk-answers", html)
                # the Screen click card is the one path that reads it
                self.assertIn("blk-shot", ui.run_screen(screens.preset("catalog"), prefer="saved"))

    def test_only_the_api_endpoints_are_public_and_screen_click_is_not_among_them(self):
        with serving(FOUR, MIMO):
            app = reload_app()
            demo = app.build()
            try:
                named = set(demo.get_api_info()["named_endpoints"])
                ids = {b._id for b in demo.blocks.values()
                       if getattr(b, "elem_id", None) in ("screen-out", "screen-upload", "screen-task", "screen-boxes",
                                                          "screen-grid", "screen-run")}
                touching = [fn for fn in demo.fns.values()
                            if ids & ({getattr(x, "_id", x) for x in fn.inputs + fn.outputs}
                                      | {t[0] for t in (fn.targets or [])})]
            finally:
                demo.close()
        self.assertIn("/v1_systemone", named)
        self.assertIn("/v1_ask", named)
        self.assertEqual(len(ids), 6)
        self.assertGreaterEqual(len(touching), 12)
        for fn in touching:
            with self.subTest(fn=fn.api_name):
                self.assertEqual(fn.api_visibility, "private")
                self.assertNotIn("/" + str(fn.api_name), named)
        for name in ("/screen_run", "/load_upload", "/drawn", "/dropped", "/redraw"):
            self.assertNotIn(name, named)


class TestHonestCard(unittest.TestCase):
    """What a saved run says is what the card shows, with the rule that read it."""

    def setUp(self):
        import ui

        self.ui = ui

    def saved(self, shot, done_logits):
        with serving(FOUR, MIMO):
            entry = recording(shot)
            logits = entry[screens.request_key(shot)]["logits"]
            logits["done"], logits["risky"] = done_logits, [-2.0, 2.0]
            blink._ENGINES[MIMO] = blink.HybridEngine(_Replay(MIMO, entry), _Live())
            return htmllib.unescape(self.ui.run_screen(shot, prefer="saved"))

    @staticmethod
    def said(html: str) -> tuple:
        """(what the verdict does, its headline) as the card shows them."""
        found = re.search(r'<div class="blk-sv [^"]*" data-act="([^"]+)"[^>]*>.*?<h3>(.*?)<em', html)
        return found.group(1), found.group(2)

    def test_already_done_that_blink_does_not_call_done_shows_what_it_said(self):
        text = self.saved(screens.preset("done"), [-1.0, 1.0])  # p(yes) about 12%: under the rule's 60%
        self.assertEqual(self.said(text), ("click", "Click 2"))  # what blink said, as it said it
        self.assertIn('<div class="blk-meter" data-kind="done"><span class="blk-ring" style="--sp:0.1192"', text)
        self.assertIn('aria-label="Done? 12%"', text)
        self.assertIn('<em class="blk-tag blk-saved">saved run</em>', text)
        # the rule that turned those numbers into that verdict is in the card
        self.assertIn(f"<summary>{self.ui.SCREEN['rule']}</summary>", text)
        self.assertIn("def verdict(a: dict) -> tuple:", text)
        self.assertIn("DONE_AT = 0.6\nRISKY_AT = 0.5\nSURE_AT = 0.45\n", text)  # and the cut-offs it names

    def test_and_one_that_does_says_so(self):
        text = self.saved(screens.preset("done"), [2.0, -2.0])
        self.assertEqual(self.said(text), ("done", "Task looks done"))
        self.assertIn('<div class="blk-meter on" data-kind="done">', text)
        self.assertIn("def verdict(a: dict) -> tuple:", text)

    def test_no_accuracy_is_claimed_on_the_tab(self):
        shot = screens.preset("directory")
        with serving(FOUR, MIMO):
            parts = [self.ui.screen_note(), self.ui.screen_model_chip(), self.ui.screen_panel(shot),
                     self.ui.screen_panel(shot, screens.decide(shot)), self.saved(shot, [-2.0, 2.0])]
        text = htmllib.unescape(re.sub(r"<[^>]+>", " ", " ".join(parts))).lower()
        text += " " + " ".join(self.ui.SCREEN.values()).lower()
        for claim in ("accura", "benchmark", "screenspot", "mind2web", "guiodyssey", "held-out", "state of the art",
                      "sota", "jevbench", "typesafe", "success rate", "correct"):
            with self.subTest(claim=claim):
                self.assertNotIn(claim, text)


class TestStagedRecording(unittest.TestCase):
    def test_saved_screen_runs_must_come_from_the_staged_blink_py(self):
        import shutil

        stage_space = _stage_space()

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        shutil.copy(os.path.join(HERE, "blink.py"), os.path.join(tmp, "blink.py"))
        with open(os.path.join(HERE, "blink.py"), "rb") as fh:
            here = hashlib.sha256(fh.read()).hexdigest()
        cases = ((None, ""), (here, ""), ("0" * 64, "recorded with blink.py 000000000000"))
        for recorded, want in cases:
            with self.subTest(recorded=recorded):
                rec = {"model": MIMO, "requests": {}}
                if recorded:
                    rec["screens_blink_py_sha256"] = recorded
                with open(os.path.join(tmp, "replay-blink-mimo-9b.json"), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh)
                with mock.patch.object(stage_space, "HERE", tmp):
                    got = stage_space.screens_code_mismatch([FOUR, MIMO], FOUR)
                self.assertIn(want, got) if want else self.assertEqual(got, "")
        self.assertEqual(stage_space.screens_code_mismatch([FOUR], FOUR), "")


class TestScreenLinks(unittest.TestCase):
    """?tab=use-cases&case=screen&shot=<key> opens Screen click on that preset; picking one keeps the
    address in step, and a visitor's own screenshot has no link."""

    KEYS = ["catalog", "directory", "lookup", "done"]

    def setUp(self):
        need_gradio(self)
        self.addCleanup(reload_app)

    def test_every_preset_has_a_link(self):
        with serving(FOUR, MIMO):
            ui = reload_app().ui
            self.assertEqual([s.key for s in screens.presets()], self.KEYS)
            for key in self.KEYS:
                for raw in (f"?tab=use-cases&case=screen&shot={key}", f"?case=screen&shot={key}",
                            f"?tab=use-cases&case=SCREEN&shot={key.upper()}",
                            f"?tab=use-cases&case=screen&shot={key}#x",
                            f"?__theme=dark&tab=use-cases&case=screen&shot={key}"):
                    with self.subTest(raw=raw):
                        self.assertEqual(ui.parse_deep_link(raw), ("use-cases", "screen"))
                        self.assertEqual(ui.parse_shot(raw), key)
            # an unknown shot opens the first preset
            self.assertEqual(ui.parse_shot("?tab=use-cases&case=screen&shot=nonsense"), "catalog")
            # no shot, or a link to anything but Screen click, carries none
            for raw in ("?tab=use-cases&case=screen", "?tab=use-cases&case=screen&shot=",
                        "?tab=use-cases&case=nextclick&shot=done", "?tab=results&shot=done", "?shot=done",
                        "?case=screen&shot=done#results", "", None, 7):
                with self.subTest(raw=raw):
                    self.assertIsNone(ui.parse_shot(raw))
            # the links there were keep working, a shot beside them or not
            self.assertEqual(ui.parse_deep_link("?tab=use-cases&case=screen"), ("use-cases", "screen"))
            self.assertEqual(ui.parse_deep_link("?tab=use-cases&case=nextclick"), ("use-cases", "nextclick"))
            self.assertEqual(ui.parse_deep_link("?tab=use-cases&case=nextclick&shot=done"), ("use-cases", "nextclick"))
            self.assertEqual(ui.parse_deep_link("?tab=results&shot=done"), ("results", None))
        with serving(FOUR):  # no model reads screens here: the link opens the use cases, no preset
            ui = reload_app().ui
            self.assertEqual(ui.parse_deep_link("?tab=use-cases&case=screen&shot=done"), ("use-cases", None))
            self.assertIsNone(ui.parse_shot("?tab=use-cases&case=screen&shot=done"))

    def test_a_link_opens_the_preset_s_saved_run_and_never_a_live_one(self):
        with serving(FOUR, MIMO):
            app = reload_app()
            demo = app.build()
            try:
                fns = list(demo.fns.values())
            finally:
                demo.close()
            opened = [fn for fn in fns if getattr(fn.fn, "__name__", "") == "opened"]
            self.assertEqual(len(opened), 1)
            self.assertEqual(opened[0].api_visibility, "private")
            self.assertEqual(opened[0].js, app.READ_URL)
            live = _Live()
            live.gpu = mock.Mock(side_effect=AssertionError("a link ran the model"))
            blink._ENGINES[MIMO] = blink.HybridEngine(blink._matching_replay(MIMO), live)
            with mock.patch.object(blink, "_forward_images", live.gpu):
                for key in self.KEYS:
                    picked, upload, task, group, boxes, trouble, card, shot = opened[0].fn(
                        f"?tab=use-cases&case=screen&shot={key}")
                    with self.subTest(key=key):
                        self.assertEqual((picked, upload, task, boxes, trouble, shot),
                                         (key, None, screens.preset(key).task, "", "", key))
                        self.assertEqual(group, {"__type__": "update", "visible": False})
                        self.assertIn(f'data-shot="{key}"', card)
                        self.assertIn('<em class="blk-tag blk-saved">saved run</em>', card)
                self.assertIn("Task looks done", htmllib.unescape(opened[0].fn("?case=screen&shot=done")[6]))
                self.assertEqual(opened[0].fn("?tab=use-cases&case=screen&shot=nonsense")[-1], "catalog")
                for raw in ("?tab=use-cases&case=screen", "?tab=use-cases&case=nextclick&shot=done", ""):
                    self.assertTrue(all(x == {"__type__": "update"} for x in opened[0].fn(raw)), raw)
            live.gpu.assert_not_called()

    def test_the_address_follows_the_preset_in_view(self):
        with serving(FOUR, MIMO):
            app = reload_app()
            demo = app.build()
            try:
                fns = list(demo.fns.values())
                synced = [fn for fn in fns if fn.js == app.SYNC_URL]
                # every holder reports all three, so each change writes the whole address
                self.assertEqual(len(synced), 3)
                holders = {tuple(b._id for b in fn.inputs) for fn in synced}
                self.assertEqual(len(holders), 1)
                picks = [fn for fn in fns if fn.fn is None and fn.js in {f'() => "{k}"' for k in self.KEYS}]
                self.assertEqual(sorted(fn.js for fn in picks), sorted(f'() => "{k}"' for k in self.KEYS))
                self.assertTrue(all(fn.api_visibility == "private" for fn in picks))
                shot_holder = picks[0].outputs[0]
                self.assertTrue(all(fn.outputs == [shot_holder] for fn in picks))
                self.assertEqual(list(holders)[0][2], shot_holder._id)
                # an upload clears it: the upload's own step writes it
                self.assertTrue(any(fn.fn is not None and getattr(fn.fn, "__name__", "") == "load_upload"
                                    and fn.outputs[-1] is shot_holder for fn in fns))
            finally:
                demo.close()
        self.assertIn("searchParams.set('shot', shot)", app.SYNC_URL)
        self.assertIn('sub === "screen"', app.SYNC_URL)

    def test_the_card_links_to_running_it_yourself(self):
        import ui

        shot = screens.preset("lookup")
        link = ('<div class="blk-links blk-runit"><a href="https://thegovind.github.io/blink/computer-use/"'
                ' target="_blank" rel="noopener">Run it yourself</a></div>')
        with serving(FOUR, MIMO):
            for html in (ui.screen_panel(shot), ui.screen_panel(shot, busy=True), ui.screen_panel(shot, draw=True),
                         ui.screen_panel(shot, screens.decide(shot))):
                self.assertEqual(html.count(link), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
