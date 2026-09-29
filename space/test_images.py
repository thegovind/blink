"""Image lifting, vision rendering, and fixed text-only goldens from the pre-image runtime.

Optional model tests use tiny random weights and run on CPU with torch and transformers installed.
"""

import base64
import builtins
import concurrent.futures
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import queue
import random
import shutil
import string
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    from PIL import Image
except ImportError:  # pragma: no cover - the public CI installs pillow
    raise unittest.SkipTest("pillow is not installed") from None

os.environ.setdefault("BLINK_MOCK", "1")
import blink  # noqa: E402

try:
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5VisionConfig
except ImportError:
    torch = None


def uri(width=8, height=8, kind="PNG") -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (205, 15, 10)).save(buf, format=kind)
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif"}[kind]
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class TestImageInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.png = uri()

    def test_string_inline_and_repeated_images(self):
        result = blink.extract_images(f"Before {self.png} between {self.png} after")
        self.assertEqual(result.state, "Before [image 1] between [image 2] after")
        self.assertEqual(len(result.images), 2)
        self.assertEqual(result.loc, ["body", "state"])
        self.assertNotIn(self.png, str(result.state))

    def test_inline_data_uris_can_be_followed_by_punctuation(self):
        result = blink.extract_images(f"Look at {self.png}, then {self.png}.")
        self.assertEqual(result.state, "Look at [image 1], then [image 2].")
        for left, right in (("`", "`"), ('"', '"'), ("(", ")")):
            with self.subTest(wrapper=(left, right)):
                result = blink.extract_images(f"Screenshot: {left}{self.png}{right}")
                self.assertEqual(result.state, f"Screenshot: {left}[image 1]{right}")
                self.assertEqual(len(result.images), 1)

    def test_structured_state_and_messages(self):
        value = {"messages": [{"content": ["Look ", {"image": self.png}, f"then {self.png}"]}],
                 "text": "[image 1] is literal"}
        result = blink.extract_images(value)
        self.assertEqual(result.state, {"messages": [{"content": [
            "Look ", {"image": "[image 1]"}, "then [image 2]"]}], "text": "[image 1] is literal"})
        self.assertEqual(result.loc, ["body", "state", "messages", 0, "content", 1, "image"])
        self.assertEqual(value["messages"][0]["content"][1]["image"], self.png)

    def test_top_level_images_for_each_state_shape(self):
        for state, expected in (
            ("context", "context\n[image 1]"),
            (["context"], ["context", "[image 1]"]),
            ({"context": "yes"}, {"state": {"context": "yes"}, "images": ["[image 1]"]}),
            (None, {"state": None, "images": ["[image 1]"]}),
        ):
            with self.subTest(state=state):
                self.assertEqual(blink.extract_images(state, [self.png]).state, expected)

    def test_inline_images_precede_top_level_images(self):
        result = blink.extract_images({"screen": self.png}, [self.png])
        self.assertEqual(result.state, {"state": {"screen": "[image 1]"}, "images": ["[image 2]"]})
        self.assertEqual(len(set(result.markers)), 2)

    def test_supported_mime_types(self):
        for kind in ("PNG", "JPEG", "WEBP"):
            with self.subTest(kind=kind):
                self.assertEqual(len(blink.extract_images(None, [uri(kind=kind)]).images), 1)

    def test_text_without_images_is_not_transformed(self):
        state = {"text": "[image 1] and data:text/plain;base64,YQ=="}
        self.assertIsNone(blink.extract_images(state))
        self.assertIsNone(blink.extract_images(state, []))
        with mock.patch.dict(os.environ, {"BLINK_MAX_IMAGES": "bad"}):
            self.assertIsNone(blink.extract_images(state))

    def test_incomplete_image_headers_are_ordinary_text(self):
        for state in ("The manual explains the data:image URI scheme.",
                      {"mime": "data:image/png"}, {"mime": "data:image/gif"},
                      ["data:image/jpeg;base64"], "data:image/gif;base64", "data:image/png,YQ==",
                      "data:image/png;charset=utf-8", "data:image/png; charset=utf-8"):
            with self.subTest(state=state):
                self.assertFalse(blink.contains_image_uri(state))
                self.assertIsNone(blink.extract_images(state))
        split = {"messages": [{"content": ["data:image/png;", "base64,YQ=="]}]}
        self.assertFalse(blink.contains_image_uri(split))
        self.assertIsNone(blink.extract_images(split))

    def test_incomplete_image_headers_scan_in_linear_time(self):
        def median_time(state):
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                self.assertFalse(blink.contains_image_uri(state))
                self.assertIsNone(blink.extract_images(state))
                samples.append(time.perf_counter() - start)
            return sorted(samples)[2]

        short = median_time("data:image/png;" * 20000)
        long = median_time("data:image/png;" * 40000)
        self.assertLess(short, 0.2)
        self.assertLess(long, 0.2)
        self.assertLessEqual(long, 3 * short)
        large = median_time("x" * (4 * 1024 * 1024))
        self.assertLess(large, 0.2)
        print(f"image header scaling: 20k={short * 1000:.2f}ms, 40k={long * 1000:.2f}ms, "
              f"4MiB={large * 1000:.2f}ms", file=sys.stderr)

    def test_seeded_header_fuzz_has_no_silent_fallback(self):
        def reference_headers(value):
            lowered = value.lower()
            found = []
            offset = 0
            while True:
                start = lowered.find("data:image/", offset)
                if start == -1:
                    return found
                comma = lowered.find(",", start, start + 256)
                if comma != -1:
                    original = lowered[start:comma]
                    compact = "".join(char for char in original if not char.isspace())
                    if compact.endswith(";base64"):
                        canonical = compact in {"data:image/png;base64", "data:image/jpeg;base64",
                                                "data:image/webp;base64"} and compact == original
                        found.append(canonical)
                offset = start + len("data:image/")

        rng = random.Random(20260927)

        def mixed_case(text):
            return "".join(char.upper() if rng.randrange(2) else char.lower() for char in text)

        prose = ("", "Screenshot: `", 'The page says "', "data:image/png, then ", "Look (")
        mime_types = ("png", "jpeg", "webp", "gif", "svg+xml", "x-icon")
        params = ("", ";charset=utf-8", "; charset=utf-8 ", ";foo=bar;charset=utf-8", " ; ")
        endings = (";base64,", ";BASE64,", " ;BASE64,", " ;BASE64 ,", ",", ";base64", ";other,")
        payloads = ("", "YQ==", "AAAA", "%%%%", "a", "QUJD", "word", "abc==")
        valid_payloads = {"png": self.png.partition(",")[2],
                          "jpeg": uri(kind="JPEG").partition(",")[2],
                          "webp": uri(kind="WEBP").partition(",")[2]}
        tails = ("", "`", '"', ")", " caption", " and more")

        def fragment():
            mime = rng.choice(mime_types)
            return (mixed_case("data:image/") + mixed_case(mime) + rng.choice(params)
                    + rng.choice(endings) + rng.choice((*payloads, valid_payloads.get(mime, ""))))

        max_scan = max_extract = 0.0
        counts = {"no_header": 0, "canonical": 0, "noncanonical": 0, "extracted": 0, "error": 0}
        for case in range(2400):
            value = rng.choice(prose) + fragment() + rng.choice(tails)
            if rng.randrange(4) == 0:
                value += " " + fragment()
            headers = reference_headers(value)
            counts["no_header" if not headers else "canonical" if all(headers) else "noncanonical"] += 1
            with self.subTest(case=case, value=value[:100]):
                started = time.perf_counter()
                self.assertEqual(blink.contains_image_uri(value), bool(headers))
                max_scan = max(max_scan, time.perf_counter() - started)
                started = time.perf_counter()
                try:
                    extracted = blink.extract_images(value)
                except blink.ImageError:
                    self.assertTrue(headers, "image-free text must not be rejected")
                    counts["error"] += 1
                else:
                    if not headers:
                        self.assertIsNone(extracted)
                    else:
                        self.assertTrue(all(headers), "unsupported image headers must not become text")
                        self.assertIsNotNone(extracted)
                        self.assertEqual(len(extracted.images), len(headers))
                        counts["extracted"] += 1
                max_extract = max(max_extract, time.perf_counter() - started)
        self.assertLess(max_scan, 0.2)
        self.assertTrue(all(counts.values()))
        print(f"seeded header fuzz: seed=20260927, cases=2400, {counts}, "
              f"max_scan_ms={max_scan * 1000:.3f}, max_extract_ms={max_extract * 1000:.3f}",
              file=sys.stderr)

    def test_seeded_canonical_extraction_matches_b397(self):
        if shutil.which("git") is None:
            self.skipTest("requires the frozen source Git object")
        root = Path(__file__).resolve().parents[1]
        try:
            source = subprocess.check_output(
                ["git", "-C", str(root), "show", "b397863:space/blink.py"], stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            self.skipTest("requires the frozen source Git object")
        frozen = types.ModuleType("blink_b397_fuzz")
        frozen.__file__ = "b397863:space/blink.py"
        sys.modules[frozen.__name__] = frozen
        try:
            exec(compile(source, frozen.__file__, "exec"), frozen.__dict__)
            rng = random.Random(20260928)
            sources = {kind: uri(kind=kind) for kind in ("PNG", "JPEG", "WEBP")}
            wrappers = (("", ""), ("Screenshot: `", "`"), ('Look "', '"'),
                        ("see (", ")"), ("\u0130 preface: ", " and done"), ("before ", " word"))
            max_extract = 0.0
            def normalized(request):
                serialized = json.dumps(request.marked_state, ensure_ascii=False)
                for i, marker in enumerate(request.markers, 1):
                    serialized = serialized.replace(marker, f"<image {i}>")
                return serialized

            def compare(state):
                nonlocal max_extract
                started = time.perf_counter()
                try:
                    old = frozen.extract_images(state)
                except frozen.ImageError as expected:
                    with self.assertRaises(blink.ImageError) as caught:
                        blink.extract_images(state)
                    self.assertEqual((str(caught.exception), caught.exception.loc),
                                     (str(expected), expected.loc))
                else:
                    got = blink.extract_images(state)
                    self.assertEqual(got.state, old.state)
                    self.assertEqual(got.loc, old.loc)
                    self.assertEqual(len(got.images), len(old.images))
                    self.assertEqual([image.size for image in got.images],
                                     [image.size for image in old.images])
                    self.assertEqual([image.tobytes() for image in got.images],
                                     [image.tobytes() for image in old.images])
                    self.assertEqual(normalized(got), normalized(old))
                max_extract = max(max_extract, time.perf_counter() - started)

            for case in range(384):
                kind = rng.choice(tuple(sources))
                header, comma, payload = sources[kind].partition(",")
                varied = "".join(char.upper() if rng.randrange(2) else char.lower() for char in header)
                before, after = rng.choice(wrappers)
                value = before + varied + comma + payload + after
                state = rng.choice((value, {"screen": value}, [value], {"messages": [{"content": value}]}))
                with self.subTest(case=case, kind=kind):
                    compare(state)
            padded, unpadded = sources["PNG"], sources["JPEG"]
            self.assertTrue(padded.endswith("="))
            self.assertFalse(unpadded.endswith("="))
            boundaries = (
                ("padded-adjacent", padded + padded),
                ("unpadded-adjacent", unpadded + padded),
                ("padded-suffix", padded + "abc"),
                ("unpadded-word", unpadded + "word"),
                ("unpadded-caption", unpadded + "caption"),
            )
            for name, state in boundaries:
                with self.subTest(boundary=name):
                    compare(state)
            print(f"frozen extraction fuzz: seed=20260928, cases={384 + len(boundaries)}, "
                  f"max_extract_ms={max_extract * 1000:.3f}", file=sys.stderr)
        finally:
            del sys.modules[frozen.__name__]

    def test_deep_image_free_and_image_states_have_no_scanner_depth_limit(self):
        plain, pictured = "leaf", self.png
        for _ in range(500):
            plain, pictured = [plain], [pictured]
        self.assertFalse(blink.contains_image_uri(plain))
        self.assertIsNone(blink.extract_images(plain))
        lifted = blink.extract_images(pictured)
        self.assertEqual(len(lifted.loc), 502)
        visible = lifted.state
        for _ in range(500):
            visible = visible[0]
        self.assertEqual(visible, "[image 1]")

    def test_rejects_bad_data_uris_and_image_bytes(self):
        invalid = (
            ("data:image/png;base64,", "data:image"),
            ("data:image/png;base64,%%%%", "data:image"),
            ("data:image/png;base64,a", "base64"),
            ("data:image/png;base64,YQ==", "decoded"),
            (self.png.replace("image/png", "image/jpeg"), "MIME"),
        )
        for value, reason in invalid:
            with self.subTest(value=value[:38]):
                with self.assertRaises(blink.ImageError) as caught:
                    blink.extract_images({"screen": value})
                self.assertIn(reason, str(caught.exception))
                self.assertEqual(caught.exception.loc, ["body", "state", "screen"])
        for value in (uri(kind="GIF"), "data:image/png,YQ=="):
            with self.subTest(explicit=value), self.assertRaises(blink.ImageError) as caught:
                blink.extract_images("context", [value])
            self.assertIn("data:image", str(caught.exception))
            self.assertEqual(caught.exception.loc, ["body", "images", 0])

    def test_unsupported_inline_mime_is_rejected_before_decoding_or_admission(self):
        svg = "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode("ascii")
        for source in (uri(kind="GIF"), svg):
            with self.subTest(mime=source.split(";", 1)[0]):
                self.assertTrue(blink.contains_image_uri({"screen": source}))
                with mock.patch.object(blink, "_decode_image", side_effect=AssertionError("decoded")) as decoder:
                    with self.assertRaises(blink.ImageError) as caught:
                        blink.inspect_images({"screen": source})
                    self.assertIn("data:image/png", str(caught.exception))
                    self.assertEqual(caught.exception.loc, ["body", "state", "screen"])
                    decoder.assert_not_called()

    def test_parameterized_images_are_located_422s_before_decode(self):
        svg = "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode("ascii")
        for source in (self.png, uri(kind="GIF"), svg):
            for separator in (";charset=utf-8;base64,", "; charset=utf-8 ;BASE64,"):
                parameterized = source.replace(";base64,", separator, 1)
                cases = (
                    (f"Screenshot: `{parameterized}`", blink._NO_IMAGES, ["body", "state"]),
                    ({"screen": parameterized}, blink._NO_IMAGES, ["body", "state", "screen"]),
                    (["text", parameterized], blink._NO_IMAGES, ["body", "state", 1]),
                    ({"messages": [{"content": f"see {parameterized} now"}]}, blink._NO_IMAGES,
                     ["body", "state", "messages", 0, "content"]),
                    ("context", [parameterized], ["body", "images", 0]),
                )
                for state, images, loc in cases:
                    with self.subTest(mime=source.split(";", 1)[0], separator=separator, loc=loc):
                        with mock.patch.object(blink, "_decode_image", side_effect=AssertionError("decoded")) as decoder:
                            if images is blink._NO_IMAGES:
                                self.assertTrue(blink.contains_image_uri(state))
                            with self.assertRaises(blink.ImageError) as caught:
                                blink.inspect_images(state, images)
                            self.assertIn("parameters", str(caught.exception))
                            self.assertEqual(caught.exception.loc, loc)
                            decoder.assert_not_called()

    def test_rejects_top_level_shape_and_count(self):
        for images in (None, "not a list", [1]):
            with self.subTest(images=images), self.assertRaises(blink.ImageError):
                blink.extract_images("look", images)
        with self.assertRaises(blink.ImageError) as caught:
            blink.extract_images(self.png, [self.png, self.png])
        self.assertIn("at most 2", str(caught.exception))
        self.assertEqual(caught.exception.loc, ["body", "images", 1])

    def test_rejects_byte_and_source_pixel_limits(self):
        limits = blink.ImageLimits(max_bytes=len(base64.b64decode(self.png.split(",")[1])) - 1)
        with self.assertRaisesRegex(blink.ImageError, "bytes"):
            blink.extract_images(None, [self.png], limits)
        with self.assertRaisesRegex(blink.ImageError, "source pixels"):
            blink.extract_images(None, [uri(512, 512)], blink.ImageLimits(max_source_pixels=200_000))
        with self.assertRaisesRegex(blink.ImageError, "aspect ratio"):
            blink.extract_images(None, [uri(1, 201)])

    def test_explicit_byte_limit_accepts_large_frozen_screenshots(self):
        raw = io.BytesIO()
        Image.frombytes("RGB", (2048, 2048), random.Random(13).randbytes(2048 * 2048 * 3)).save(
            raw, format="PNG")
        self.assertGreater(len(raw.getvalue()), 8 * 1024 * 1024)
        source = "data:image/png;base64," + base64.b64encode(raw.getvalue()).decode("ascii")
        with self.assertRaisesRegex(blink.ImageError, "decoded bytes"):
            blink.extract_images(None, [source])
        accepted = blink.extract_images(None, [source], limits=blink.ImageLimits(max_bytes=20 * 1024 * 1024))
        self.assertLessEqual(accepted.images[0].width * accepted.images[0].height, 1920 * 1088)

    def test_resizes_to_the_processor_policy(self):
        self.assertEqual(blink.extract_images(self.png).images[0].size, (256, 256))
        image = blink.extract_images(uri(1600, 900)).images[0]
        self.assertGreater(image.width * image.height, 400_000)
        self.assertLessEqual(image.width * image.height, 1920 * 1088)
        self.assertGreaterEqual(image.width * image.height, 65_536)
        self.assertEqual((image.width % 32, image.height % 32), (0, 0))
        full = blink.extract_images(uri(1920, 1080)).images[0]
        self.assertEqual(full.size, (1920, 1088))
        self.assertEqual(full.width * full.height // 32**2, 2040)
        larger = blink.extract_images(uri(2560, 1440),
                                      limits=blink.ImageLimits(max_pixels=16_777_216)).images[0]
        self.assertEqual(larger.size, (2560, 1440))
        with self.assertRaisesRegex(blink.ImageError, "cannot fit"):
            blink.extract_images(self.png, limits=blink.ImageLimits(max_pixels=65_535))

    def test_oriented_jpeg_keeps_its_display_aspect_ratio(self):
        buffer = io.BytesIO()
        orientation = Image.Exif()
        orientation[274] = 6
        Image.new("RGB", (480, 240), "blue").save(buffer, format="JPEG", exif=orientation)
        src = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        image = blink.extract_images(None, [src]).images[0]
        self.assertLess(image.width, image.height)
        self.assertLessEqual(image.width * image.height, 1920 * 1088)

    def test_environment_limits_apply_only_to_images(self):
        with mock.patch.dict(os.environ, {"BLINK_MAX_IMAGES": "1", "BLINK_MAX_IMAGE_PIXELS": "65536"}):
            self.assertEqual(blink.extract_images(self.png).limits.max_pixels, 65_536)
            with self.assertRaisesRegex(blink.ImageError, "at most 1"):
                blink.extract_images(None, [self.png, self.png])
        with mock.patch.dict(os.environ, {"BLINK_MAX_IMAGE_BYTES": str(20 * 1024 * 1024),
                                          "BLINK_MAX_IMAGE_SOURCE_PIXELS": "64000000"}):
            self.assertEqual(blink.image_limits().max_bytes, 20 * 1024 * 1024)
            self.assertEqual(blink.image_limits().max_source_pixels, 64_000_000)
        with mock.patch.dict(os.environ, {"BLINK_MAX_IMAGE_BYTES": "bad"}):
            with self.assertRaisesRegex(blink.BlinkError, "BLINK_MAX_IMAGE_BYTES"):
                blink.extract_images(self.png)

    def test_text_model_refuses_images(self):
        question = {"q": {"type": "noul", "instructions": "Is this visible?"}}
        prior = blink._ENGINE
        blink._ENGINE = blink.MockEngine()
        try:
            for kwargs in ({"images": [self.png]}, {}):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(blink.ImageError, "reads text only"):
                    blink.decide(self.png if not kwargs else "state", question, **kwargs)
            batcher = blink.Batcher(0.01)
            with self.assertRaisesRegex(blink.ImageError, "reads text only"):
                batcher.submit("state", question, images=[self.png])
        finally:
            blink._ENGINE = prior

    def test_text_only_refusal_never_decodes_or_imports_pillow(self):
        questions = {"q": {"type": "noul", "instructions": "Visible?"}}
        prior = blink._ENGINE
        blink._ENGINE = blink.MockEngine()
        original_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise AssertionError("Pillow imported on a text-only image refusal")
            return original_import(name, *args, **kwargs)

        try:
            with mock.patch.object(blink, "_decode_image", side_effect=AssertionError("decoder called")) as decoder:
                with self.assertRaisesRegex(blink.ImageError, "reads text only"):
                    blink.decide("state", questions, images=[self.png])
                decoder.assert_not_called()
            with mock.patch("builtins.__import__", side_effect=checked_import):
                with self.assertRaisesRegex(blink.ImageError, "reads text only"):
                    blink.decide({"image": self.png}, questions)
        finally:
            blink._ENGINE = prior

    def test_full_batch_queue_does_not_decode_before_admission(self):
        batcher = object.__new__(blink.Batcher)
        batcher.max_queued, batcher.model = 1, None
        batcher.worker = SimpleNamespace(is_alive=lambda: True)
        batcher.q = queue.Queue(maxsize=1)
        batcher.q.put(("occupied", {}, None))
        vision = SimpleNamespace(accepts_images=True, image_limits=blink.ImageLimits())
        questions = {"q": {"type": "noul", "instructions": "Visible?"}}
        with mock.patch.object(blink, "engine", return_value=vision), \
                mock.patch.object(blink, "_decode_image",
                                  side_effect=AssertionError("decoder called")) as decoder:
            with self.assertRaises(blink.BlinkBusy):
                batcher.submit({"screenshot": self.png}, questions)
            with self.assertRaises(blink.ImageError) as caught:
                batcher.submit({"screenshot": uri(kind="GIF")}, questions)
            self.assertEqual(caught.exception.loc, ["body", "state", "screenshot"])
            svg = "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode("ascii")
            for source in (self.png, uri(kind="GIF"), svg):
                for separator in (";charset=utf-8;base64,", "; charset=utf-8 ;BASE64,"):
                    parameterized = source.replace(";base64,", separator, 1)
                    with self.subTest(mime=source.split(";", 1)[0], separator=separator):
                        with self.assertRaisesRegex(blink.ImageError, "parameters") as caught:
                            batcher.submit({"screenshot": parameterized}, questions)
                        self.assertEqual(caught.exception.loc, ["body", "state", "screenshot"])
                        with self.assertRaisesRegex(blink.ImageError, "parameters") as caught:
                            batcher.submit("context", questions, images=[parameterized])
                        self.assertEqual(caught.exception.loc, ["body", "images", 0])
            decoder.assert_not_called()

    def test_batch_worker_serializes_image_decoders(self):
        class Vision:
            name = "torch"
            model_id = blink.MODEL_ID
            temperature = 1.0
            accepts_images = True
            image_limits = blink.ImageLimits()

            def logits_images(self, request, questions):
                self.last_image_pixels = request.images[0].width * request.images[0].height
                self.last_visual_tokens = self.last_image_pixels // 1024
                return {name: [0.1, 0.2] for name in questions}, 10

        lock, second = threading.Lock(), threading.Event()
        active = peak = 0

        def decode(_uri, _index, _loc, _limits):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    second.set()
            if active == 1:
                second.wait(timeout=0.5)
            try:
                return Image.new("RGB", (256, 256), "navy")
            finally:
                with lock:
                    active -= 1

        questions = {"q": {"type": "noul", "instructions": "Visible?"}}
        with mock.patch.object(blink, "_ENGINE", Vision()), mock.patch.object(
            blink, "_decode_image", side_effect=decode
        ):
            batcher = blink.Batcher(0.01, max_requests=2, max_queued=4)
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                responses = [pool.submit(batcher.submit, {"screenshot": self.png}, questions) for _ in range(2)]
                self.assertTrue(all(future.result(timeout=10)["answers"]["q"]["type"] == "noul"
                                    for future in responses))
        self.assertEqual(active, 0)
        self.assertEqual(peak, 1)


class StubProcessor:
    image_processor = SimpleNamespace(patch_size=16, merge_size=2)

    def __init__(self):
        self.contents = []
        self.images = []

    def apply_chat_template(self, messages, **kwargs):
        self.contents.append(messages[-1]["content"])
        return "prepared"

    def __call__(self, *, text, images, return_tensors):
        self.images.append(images)
        return {"input_ids": SimpleNamespace(shape=(1, 10)),
                "image_grid_thw": [(1, 2, 2)] * len(images)}


class TestImageRendering(unittest.TestCase):
    def test_images_take_the_exact_placeholder_positions(self):
        png = uri()
        request = blink.extract_images({"before": png, "literal": "[image 1]", "after": png})
        engine = object.__new__(blink.TorchEngine)
        engine.processor = StubProcessor()
        engine.accepts_images = True
        engine.image_layout = "inline"
        engine.image_limits = blink.ImageLimits()
        engine.labels, engine.label_ids = ["A", "B"], [5, 6]
        questions = {"first": {"type": "noul", "instructions": "First?"},
                     "second": {"type": "noul", "instructions": "Second?"}}
        work = engine.render_images(request, questions)
        self.assertEqual(len(work), 2)
        for content in engine.processor.contents:
            parts = [x["type"] for x in content]
            self.assertEqual(parts.count("image"), 2)
            combined = "".join(x.get("text", "#") for x in content)
            self.assertIn('"literal": "[image 1]"', combined)
            self.assertLess(combined.index('"before"'), combined.index('"after"'))
            self.assertNotIn(request.markers[0], combined)
            self.assertIs(next(x["image"] for x in content if x["type"] == "image"), request.images[0])
        self.assertEqual([w["qkey"] for w in work], ["first", "second"])

    def test_first_layout_preserves_markers_and_image_reading_order(self):
        request = blink.extract_images({"before": uri(), "literal": "[image 1]",
                                        "after": ["see " + uri(kind="JPEG")]})
        engine = object.__new__(blink.TorchEngine)
        engine.processor = StubProcessor()
        engine.accepts_images = True
        engine.image_layout = "first"
        engine.image_limits = blink.ImageLimits()
        engine.labels, engine.label_ids = ["A", "B"], [5, 6]
        engine.render_images(request, {"q": {"type": "noul", "instructions": "Which?"}})
        content = engine.processor.contents[0]
        self.assertEqual([part["type"] for part in content], ["image", "image", "text"])
        self.assertIs(content[0]["image"], request.images[0])
        self.assertIs(content[1]["image"], request.images[1])
        self.assertEqual(engine.processor.images[0], list(request.images))
        text = content[2]["text"]
        self.assertIn('"before": "[image 1]"', text)
        self.assertIn('"literal": "[image 1]"', text)
        self.assertIn('"after": ["see [image 2]"]', text)
        self.assertFalse(any(marker in text for marker in request.markers))

    def test_uninitialized_training_renderer_uses_frozen_default(self):
        request = blink.extract_images(uri())
        engine = object.__new__(blink.TorchEngine)
        engine.processor = StubProcessor()
        engine.accepts_images = True
        engine.image_limits = blink.ImageLimits()
        engine.labels, engine.label_ids = ["A", "B"], [5, 6]
        self.assertEqual(blink.IMAGE_LAYOUT_DEFAULT, "first")
        engine.render_images(request, {"q": {"type": "noul", "instructions": "Which?"}})
        self.assertEqual([part["type"] for part in engine.processor.contents[0]], ["image", "text"])

    def test_invalid_layout_is_explicit(self):
        request = blink.extract_images(uri())
        engine = object.__new__(blink.TorchEngine)
        engine.accepts_images = True
        engine.image_layout = "missing"
        with self.assertRaisesRegex(blink.BlinkError, "image_layout must be inline or first"):
            engine.render_images(request, {"q": {"type": "noul", "instructions": "Which?"}})

    def test_image_context_limit_is_a_render_error(self):
        request = blink.extract_images(uri())
        engine = object.__new__(blink.TorchEngine)
        engine.processor = StubProcessor()
        engine.accepts_images = True
        engine.image_limits = blink.ImageLimits()
        engine.labels, engine.label_ids = ["A", "B"], [5, 6]
        with mock.patch.object(blink, "MAX_INPUT_TOKENS", 5):
            with self.assertRaisesRegex(blink.BlinkError, "question 'q' renders to 10 tokens"):
                engine.render_images(request, {"q": {"type": "noul", "instructions": "?"}})

    def test_live_pixel_storage_is_bounded_across_many_questions(self):
        class TrackedPixels:
            live = 0
            peak = 0

            def __init__(self):
                cls = type(self)
                cls.live += 1
                cls.peak = max(cls.peak, cls.live)

            def __del__(self):
                type(self).live -= 1

        class TrackingProcessor(StubProcessor):
            def __call__(self, **kwargs):
                return {**super().__call__(**kwargs), "pixel_values": TrackedPixels()}

        engine = object.__new__(blink.TorchEngine)
        engine.processor = TrackingProcessor()
        engine.accepts_images = True
        engine.image_limits = blink.ImageLimits()
        engine.labels, engine.label_ids = ["A", "B"], [5, 6]
        engine.key = id(engine)
        request = blink.extract_images(uri())

        def forward(_key, works):
            return [[0.1, 0.2] for _ in works], 1.0, 0.5, 0.5

        for count in (1, 32, blink.MAX_QUESTIONS):
            questions = {f"q{i}": {"type": "noul", "instructions": "Visible?"} for i in range(count)}
            TrackedPixels.peak = 0
            with mock.patch.object(blink, "_forward_images", forward):
                result, _ = engine.logits_images(request, questions)
            gc.collect()
            self.assertEqual(len(result), count)
            self.assertEqual(TrackedPixels.live, 0)
            self.assertLessEqual(TrackedPixels.peak, 1)


class TestVisionSources(unittest.TestCase):
    def test_shared_graft_keys_follow_the_parent_layout(self):
        from graft_keys import text_to_parent

        self.assertEqual(text_to_parent("model.layers.0.mlp.up_proj.weight"),
                         "model.language_model.layers.0.mlp.up_proj.weight")
        self.assertEqual(text_to_parent("model.language_model.embed_tokens.weight"),
                         "model.language_model.embed_tokens.weight")
        self.assertEqual(text_to_parent("lm_head.weight"), "lm_head.weight")
        with self.assertRaisesRegex(ValueError, "unexpected key"):
            text_to_parent("visual.patch_embed.weight")

    def test_each_text_model_uses_only_its_own_base(self):
        self.assertEqual(blink.vision_source("blink-4b", "Qwen/Qwen3.5-4B@rev"),
                         ("Qwen/Qwen3.5-4B", "rev"))
        self.assertEqual(blink.vision_source("blink-27b", "Qwen/Qwen3.8-27B@rev"),
                         ("Qwen/Qwen3.8-27B", "rev"))
        for model, tower in (("blink-4b", "Qwen/Qwen3.8-27B"),
                             ("blink-27b", "Qwen/Qwen3.5-4B"),
                             ("blink-4b", "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"),
                             ("blink-27b", "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B")):
            with self.subTest(model=model, tower=tower):
                with self.assertRaisesRegex(blink.BlinkError, "must use"):
                    blink.vision_source(model, tower + "@rev")
        with self.assertRaisesRegex(blink.BlinkError, "only supported"):
            blink.vision_source("blink-mimo-9b", "Qwen/Qwen3.8-27B@rev")
        with self.assertRaisesRegex(blink.BlinkError, "repo@revision"):
            blink.vision_source("blink-27b", "Qwen/Qwen3.8-27B")


@unittest.skipIf(torch is None, "torch/transformers not installed")
class TestTextGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

        class TinyTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return blink.SYSTEM + "\n" + messages[-1]["content"]

            def __call__(self, prompt, add_special_tokens=False):
                return {"input_ids": [ord(c) % 100 + 27 for c in prompt]}

        torch.manual_seed(17)
        cls.text_config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, layer_types=["full_attention"],
            tie_word_embeddings=False, max_position_embeddings=4096,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                             "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        )
        cls.engine = engine = object.__new__(blink.TorchEngine)
        engine.model = modeling.Qwen3_5ForCausalLM(cls.text_config).eval()
        engine.model_id, engine.temperature, engine.token_budget = blink.MODEL_ID, 1.0, 4096
        engine.prefix_cache, engine.pad_id = False, 0
        engine.tok = TinyTokenizer()
        engine.labels, engine.label_ids = list(string.ascii_uppercase), list(range(1, 27))
        engine.key = id(engine)
        blink._LIVE[engine.key] = engine
        cls.state = {"ticket": "Confirm next button", "messages": [{"role": "user", "content": "select Save"}]}
        cls.questions = {
            "decision": {"type": "choice", "instructions": "Which button?",
                         "criteria": {"save": "Save button", "cancel": "Cancel button"}},
            "complete": {"type": "noul", "instructions": "Already done?"},
        }

    def test_pre_image_prompt_tensor_logits_and_probabilities(self):
        eng = self.engine
        work = eng.render(self.state, self.questions)
        prompts = [eng._wrap(blink.user_message(self.state, q, eng.labels[:len(blink.question_options(q))],
                                                blink.question_options(q))) for q in self.questions.values()]
        self.assertEqual([hashlib.sha256(s.encode()).hexdigest() for s in prompts], [
            "1075a4a8713e31dddd8b486a5377f87b4c011af182ce35d254fd026828485d6b",
            "17f67da7be11df6984ca89e95b5a98c02c198a9a278bc06dc893c4f1a7610d4a",
        ])
        self.assertEqual([hashlib.sha256(bytes(w["ids"])).hexdigest() for w in work], [
            "4a60da60ea4c3c50c365df6c7284594fd62bd6b1973172e8dbe1a0f8254cd277",
            "91386ee12aa4ba34e63747f48f706767c79b8e28d00642666cebceb578c6713a",
        ])
        self.assertEqual([hashlib.sha256(blink._padded([w["ids"]], [0], 0).numpy().tobytes()).hexdigest()
                          for w in work], [
            "834683a4b3b216a77af1ec53bd12b77dea928c60aa9edad24a58c24208b44469",
            "73cf136c1ba2f08fd94228af8a661722f5f1ff5de85f6396ef61ca3b536f49f9",
        ])
        raw, n_tokens = eng.logits(self.state, self.questions)
        self.assertEqual(n_tokens, 809)
        self.assertEqual(raw, {"decision": [0.22355754673480988, -0.13881993293762207],
                               "complete": [0.22887927293777466, -0.1397402137517929]})
        prior = blink._ENGINE
        try:
            blink._ENGINE = eng
            out = blink.decide(self.state, self.questions)
        finally:
            blink._ENGINE = prior
        self.assertEqual(out["answers"]["decision"]["probabilities"],
                         {"save": 0.5896158327668223, "cancel": 0.41038416723317767})
        self.assertEqual(out["answers"]["complete"]["probabilities"],
                         {"yes": 0.591125355676282, "no": 0.408874644323718})

    def test_tiny_graft_has_exact_text_logits(self):
        config = Qwen3_5Config(
            text_config=self.text_config,
            vision_config=Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                                               out_hidden_size=64, num_position_embeddings=16),
            image_token_id=3, vision_start_token_id=2, vision_end_token_id=4, video_token_id=7,
        )
        graft = Qwen3_5ForConditionalGeneration(config).eval()
        target = graft.state_dict()
        copied = {}
        for key, value in self.engine.model.state_dict().items():
            dest = key if key in target else key.replace("model.", "model.language_model.", 1)
            self.assertEqual(target[dest].shape, value.shape)
            copied[dest] = value
        graft.load_state_dict(copied, strict=False)
        vision = object.__new__(blink.TorchEngine)
        vision.model, vision.accepts_images = graft, True
        vision.model_id, vision.temperature, vision.token_budget = blink.MODEL_ID, 1.0, 4096
        vision.prefix_cache, vision.pad_id = False, 0
        vision.tok, vision.labels, vision.label_ids = self.engine.tok, self.engine.labels, self.engine.label_ids
        vision.key = id(vision)
        blink._LIVE[vision.key] = vision
        expected = self.engine.logits(self.state, self.questions)
        self.assertEqual(vision.render(self.state, self.questions), self.engine.render(self.state, self.questions))
        self.assertEqual(vision.logits(self.state, self.questions), expected)
        prior = blink._ENGINE
        try:
            blink._ENGINE = self.engine
            plain_answers = blink.decide(self.state, self.questions)["answers"]
            blink._ENGINE = vision
            self.assertEqual(blink.decide(self.state, self.questions)["answers"], plain_answers)
        finally:
            blink._ENGINE = prior


@unittest.skipIf(torch is None or importlib.util.find_spec("torchvision") is None,
                 "torch/transformers/torchvision not installed")
class TestImageForward(unittest.TestCase):
    _B397_PIXELS = {
        "single": ("ea93a35cef3de73ddcfcafb47ac41c11ca7fd4f4f48b3fddb3aec3ae4fbc3db4",
                   "8e593fdee7021d9c6f6f5c9766fcc2be8aa2b14b7196012be47a03197031dc3e", (4, 1536), 1024, 1),
        "two": ("d39cc769d26d4b076e1eda02763fc6621c728c9bf5ccca08a6571f9f2dbedb4f",
                "f7f5d2d5a2184ed0f3624e9cb58f8b388c06a185523fda3c45e6bacceleratorce5703c", (8, 1536), 2048, 2),
        "large": ("eb22eb8e22869e4a1a7fd7377dc3a91e9ad268d11295e0f4d58384a2ed26cc70",
                  "c85bbdd80e487ca0ca7da4de0b54f672263379d10e9d694a5ab11c927fbaf529",
                  (8160, 1536), 1920 * 1088, 2040),
    }
    _B397_IDS = {
        "first": {
            "single": ("f53bbde301842dba5c963ca8b3d6c1b0a2acfbaaf46b790e77cee89b5d152abe", 92),
            "two": ("1c2b6f7f60ab7d52d1f03de9a2820bb774c361056da45ea33a0bb023809f0a2b", 93),
            "large": ("d750a8de727301b9eadc4bb2d81116aad4347deac20ff6d43beb0903879c4dd6", 2119),
            "wrapped": ("3a968c0b6a7c94eab1da5b5d370b189c3707422fe13a487cfe3e3283b096d2f7", 95),
        },
        "inline": {
            "single": ("1ac4ade34d0e77202da211d26d1975c3c0471bc3646671ba95929c739e35c5a4", 89),
            "two": ("f79665ab50422113dcfe7b0fbde5412db4861d207338e35ffb1530c9146d096b", 88),
            "large": ("c5c52171cfc2347bd8e30b83bd30e99ab904990a4c568b2d3107b4550b6eb40a", 2116),
            "wrapped": ("d350aab28d74f975864f8c9f565493525bc64f1b95b93a674ea9d8cceb39e8a1", 92),
        },
    }

    def assert_b397_image_golden(self, work, layout, kind):
        pixel_hash, grid_hash, shape, pixels, visual_tokens = self._B397_PIXELS[
            "single" if kind == "wrapped" else kind]
        ids_hash, token_count = self._B397_IDS[layout][kind]
        for item in work:
            with self.subTest(layout=layout, kind=kind, question=item["qkey"]):
                inputs = item["inputs"]
                self.assertEqual((item["image_pixels"], item["visual_tokens"]), (pixels, visual_tokens))
                self.assertEqual(tuple(inputs["pixel_values"].shape), shape)
                self.assertEqual(tuple(inputs["input_ids"].shape), (1, token_count))
                for name, expected in (("input_ids", ids_hash), ("pixel_values", pixel_hash),
                                       ("image_grid_thw", grid_hash)):
                    actual = hashlib.sha256(inputs[name].contiguous().numpy().tobytes()).hexdigest()
                    self.assertEqual(actual, expected, f"b397863 {layout}/{kind}/{name} changed")

    def test_real_tiny_processor_and_vision_forward(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import (PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor,
                                  Qwen3VLVideoProcessor)

        vocab = {"<unk>": 0, "<pad>": 1, "<|vision_start|>": 2, "<|image_pad|>": 3,
                 "<|vision_end|>": 4, "<|video_pad|>": 7}
        vocab.update({letter: 10 + i for i, letter in enumerate(string.ascii_uppercase)})
        backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
            additional_special_tokens=["<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>"],
        )
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['role'] }}:"
            "{% if m['content'] is string %}{{ m['content'] }}"
            "{% else %}{% for part in m['content'] %}"
            "{% if part['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
            "{% else %}{{ part['text'] }}{% endif %}{% endfor %}{% endif %}"
            "{% endfor %}assistant:"
        )
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(size={"shortest_edge": 1024, "longest_edge": 4096},
                                                   patch_size=16, merge_size=2),
            tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(), chat_template=tokenizer.chat_template,
        )
        text_config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, layer_types=["full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                             "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        )
        vision_config = Qwen3_5VisionConfig(
            depth=1, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=64,
            num_position_embeddings=16, patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
        )
        torch.manual_seed(9)
        engine = object.__new__(blink.TorchEngine)
        engine.model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
            text_config=text_config, vision_config=vision_config, image_token_id=3,
            vision_start_token_id=2, vision_end_token_id=4, video_token_id=7,
        )).eval()
        engine.model_id, engine.temperature, engine.token_budget = blink.MODEL_ID, 1.0, 4096
        engine.accepts_images, engine.prefix_cache, engine.pad_id = True, False, 1
        engine.tok, engine.processor = tokenizer, processor
        engine.labels = list(string.ascii_uppercase)
        engine.label_ids = [tokenizer.convert_tokens_to_ids(label) for label in engine.labels]
        engine.image_limits = blink.ImageLimits(max_pixels=4096, min_pixels=1024)
        engine.key = id(engine)
        blink._LIVE[engine.key] = engine
        image = uri(32, 32)
        prepared = blink.extract_images({"before": "See ", "screenshot": image, "after": " choose."},
                                        limits=engine.image_limits)
        questions = {"action": {"type": "choice", "instructions": "Next?",
                                "criteria": {"A": "first", "B": "second"}},
                     "done": {"type": "noul", "instructions": "Finished?"}}
        for layout in ("first", "inline"):
            engine.image_layout = layout
            try:
                self.assert_b397_image_golden(engine.render_images(prepared, questions), layout, "single")
            finally:
                del engine.image_layout
        for left, right in (("`", "`"), ('"', '"'), ("(", ")")):
            enclosed = blink.extract_images(
                {"before": "See ", "screenshot": f"Screenshot: {left}{image}{right}",
                 "after": " choose."}, limits=engine.image_limits)
            self.assertEqual(enclosed.state["screenshot"], f"Screenshot: {left}[image 1]{right}")
            for layout in ("first", "inline"):
                engine.image_layout = layout
                try:
                    self.assert_b397_image_golden(engine.render_images(enclosed, questions), layout, "wrapped")
                finally:
                    del engine.image_layout
        work = engine.render_images(prepared, questions)
        self.assertEqual([tuple(w["inputs"]["image_grid_thw"][0].tolist()) for w in work], [(1, 2, 2)] * 2)
        self.assertEqual([int(w["inputs"]["input_ids"].shape[1]) for w in work],
                         [int(w["inputs"]["attention_mask"][0].sum()) for w in work])
        prior = blink._ENGINE
        try:
            blink._ENGINE = engine
            both = blink.decide(prepared, questions)
            alone = {key: blink.decide(prepared, {key: q})["answers"][key] for key, q in questions.items()}
        finally:
            blink._ENGINE = prior
        self.assertEqual(both["answers"], alone)
        self.assertEqual(both["meta"]["input_tokens"], sum(w["inputs"]["input_ids"].shape[1] for w in work))
        self.assertEqual((both["meta"]["image_pixels"], both["meta"]["visual_tokens"]), (1024, 1))
        for answer in both["answers"].values():
            self.assertTrue(all(math.isfinite(p) for p in answer["probabilities"].values()))
            self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)

        prior = blink._ENGINE
        try:
            engine.image_timing = True
            blink._ENGINE = engine
            timed = blink.decide(prepared, questions)
        finally:
            engine.image_timing = False
            blink._ENGINE = prior
        self.assertEqual(timed["answers"], both["answers"])
        for name in ("image_preprocess_ms", "vision_encoder_ms", "lm_prefill_ms"):
            self.assertGreater(timed["meta"][name], 0)

        prior = blink._ENGINE
        try:
            engine.image_layout = "first"
            blink._ENGINE = engine
            first_layout = blink.decide(prepared, questions)
        finally:
            del engine.image_layout
            blink._ENGINE = prior
        self.assertEqual(set(first_layout["answers"]), set(both["answers"]))
        self.assertEqual((first_layout["meta"]["image_pixels"], first_layout["meta"]["visual_tokens"]),
                         (1024, 1))
        for answer in first_layout["answers"].values():
            self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)

        pair = blink.extract_images({"reference": image}, [image], limits=engine.image_limits)
        for layout in ("first", "inline"):
            engine.image_layout = layout
            try:
                self.assert_b397_image_golden(engine.render_images(pair, questions), layout, "two")
            finally:
                del engine.image_layout
        pair_work = engine.render_images(pair, questions)
        self.assertEqual([len(w["inputs"]["image_grid_thw"]) for w in pair_work], [2, 2])
        self.assertEqual(set(engine.logits_images(pair, questions)[0]), set(questions))
        self.assertEqual((engine.last_image_pixels, engine.last_visual_tokens), (2048, 2))

        full_limits = blink.ImageLimits(max_pixels=1920 * 1088, min_pixels=1024)
        large = blink.extract_images({"screenshot": uri(1920, 1080)}, limits=full_limits)
        original_size, original_limits = processor.image_processor.size, engine.image_limits
        try:
            processor.image_processor.size = {"shortest_edge": 1024, "longest_edge": full_limits.max_pixels}
            engine.image_limits = full_limits
            for layout in ("first", "inline"):
                engine.image_layout = layout
                try:
                    self.assert_b397_image_golden(
                        engine.render_images(large, {"action": questions["action"]}), layout, "large")
                finally:
                    del engine.image_layout
            full_work = engine.render_images(large, {"action": questions["action"]})
            self.assertEqual((full_work[0]["image_pixels"], full_work[0]["visual_tokens"]), (1920 * 1088, 2040))
        finally:
            processor.image_processor.size, engine.image_limits = original_size, original_limits

        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

        with tempfile.TemporaryDirectory() as root:
            native = Path(root) / "blink-mimo-9b"
            native.mkdir()
            engine.model.save_pretrained(native)
            processor.save_pretrained(native)
            loaded = blink.TorchEngine(str(native), vision=True, limits=engine.image_limits)
            self.assertEqual(set(loaded.logits_images(prepared, questions)[0]), set(questions))
            base_4b = Path(root) / "Qwen3.5-4B"
            base_27b = Path(root) / "Qwen3.8-27B"
            shutil.copytree(native, base_4b)
            shutil.copytree(native, base_27b)
            base_native = blink.TorchEngine(str(base_4b), vision=True, limits=engine.image_limits)
            self.assertEqual(set(base_native.logits_images(prepared, questions)[0]), set(questions))

            text_model = modeling.Qwen3_5ForCausalLM(text_config)
            native_weights = engine.model.state_dict()
            text_model.load_state_dict({
                name: native_weights[name if name in native_weights else
                                     name.replace("model.", "model.language_model.", 1)]
                for name in text_model.state_dict()
            })
            text_dir = Path(root) / "blink-4b"
            text_dir.mkdir()
            text_model.save_pretrained(text_dir)
            tokenizer.save_pretrained(text_dir)
            plain = blink.TorchEngine(str(text_dir))
            graft = blink.TorchEngine(str(text_dir), vision_tower=f"{base_4b}@local", limits=engine.image_limits)
            self.assertEqual(plain.render("text", questions), graft.render("text", questions))
            self.assertEqual(plain.logits("text", questions), graft.logits("text", questions))
            self.assertEqual(set(graft.logits_images(prepared, questions)[0]), set(questions))

            text_27b = Path(root) / "blink-27b"
            shutil.copytree(text_dir, text_27b)
            plain_27b = blink.TorchEngine(str(text_27b))
            graft_27b = blink.TorchEngine(str(text_27b), vision_tower=f"{base_27b}@local",
                                          limits=engine.image_limits)
            self.assertEqual(plain_27b.render("text", questions), graft_27b.render("text", questions))
            self.assertEqual(plain_27b.logits("text", questions), graft_27b.logits("text", questions))
            self.assertEqual(set(graft_27b.logits_images(prepared, questions)[0]), set(questions))

            config_path = base_27b / "config.json"
            original_config = config_path.read_text(encoding="utf-8")
            for field, value, reason in (("text_config", {"hidden_size": 96}, "hidden_size"),
                                         ("vision_config", {"out_hidden_size": 96}, "visual output"),
                                         ("image_token_id", 42, "image_token_id")):
                data = json.loads(original_config)
                if isinstance(value, dict):
                    data[field].update(value)
                else:
                    data[field] = value
                config_path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(field=field), mock.patch(
                    "transformers.AutoModelForImageTextToText.from_pretrained",
                    side_effect=AssertionError("weights must not load for a mismatched tower"),
                ):
                    with self.assertRaisesRegex(blink.BlinkError, reason):
                        blink.TorchEngine(str(text_27b), vision_tower=f"{base_27b}@local",
                                          limits=engine.image_limits)
            config_path.write_text(original_config, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
