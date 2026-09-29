"""Unit tests for the pure-python parts of blink.py (rendering, labels, assembly).

    BLINK_MOCK=1 python -m unittest test_blink -v
"""

import copy
import json
import os
import contextlib
import inspect
import re
import string
import tempfile
import unittest
import unittest.mock

os.environ.setdefault("BLINK_MOCK", "1")

import blink  # noqa: E402
import examples  # noqa: E402


def results_escape(s) -> str:
    import results

    return results._e(s)


def need_gradio(test):
    """Gradio is the Space SDK, not a test dependency; skip when it is absent."""
    try:
        import gradio as gr
    except ImportError:  # pragma: no cover - depends on the environment
        test.skipTest("gradio is not installed")
    return gr


LABEL_RE = re.compile(
    r'<text class="blk-pt-lab( on)?" data-for="([^"]+)" x="(-?[\d.]+)" y="(-?[\d.]+)">'
    r'([^<]*)<tspan class="blk-ax" dx="8">([^<]*)</tspan></text>'
)
DOT_RE = re.compile(
    r'<circle class="blk-pt" data-id="([^"]+)"[^>]*cx="(-?[\d.]+)" cy="(-?[\d.]+)" r="([\d.]+)"'
)
LEAD_RE = re.compile(
    r'<path class="blk-lead" data-for="([^"]+)" d="M(-?[\d.]+),(-?[\d.]+) L(-?[\d.]+),(-?[\d.]+)"/>'
)


def assert_chart_geometry(test, html: str, points: list, frame=(0.0, 0.0, 1060.0, 500.0)):
    """No label may touch another label or anyone else's dot, and each one points home."""
    import html as htmllib

    import results

    dots = {
        m.group(1): (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        for m in DOT_RE.finditer(html)
    }
    test.assertEqual(len(dots), len(points), "every point needs a dot")
    leads = {
        m.group(1): (float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
        for m in LEAD_RE.finditer(html)
    }
    boxes = {}
    for m in LABEL_RE.finditer(html):
        pid, x, y = m.group(2), float(m.group(3)), float(m.group(4))
        name, tail = htmllib.unescape(m.group(5)), htmllib.unescape(m.group(6))
        width = results.label_width(name, tail)
        boxes[pid] = (x, y - results.LAB_TOP, x + width, y + results.LAB_BOT)

    test.assertTrue(boxes, "the chart labelled nothing")
    test.assertLess(len(boxes), len(points), "not every point should be labelled")

    for pid, box in boxes.items():
        test.assertIn(pid, dots, f"{pid} labelled but not plotted")
        test.assertGreaterEqual(box[0], frame[0], pid)
        test.assertLessEqual(box[2], frame[2], pid)
        test.assertGreaterEqual(box[1], frame[1], pid)
        test.assertLessEqual(box[3], frame[3], pid)

        test.assertIn(pid, leads, f"{pid} has no leader line")
        _, _, ex, ey = leads[pid]
        cx, cy, _ = dots[pid]
        test.assertAlmostEqual(ex, cx, places=1, msg=f"{pid} leader misses its dot")
        test.assertAlmostEqual(ey, cy, places=1, msg=f"{pid} leader misses its dot")

    ids = list(boxes)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            test.assertFalse(results._overlap(boxes[a], boxes[b]), f"{a} overlaps {b}")

    pad = results.DOT_PAD
    for pid, box in boxes.items():
        for other, (cx, cy, rad) in dots.items():
            if other == pid:
                continue
            rect = (cx - rad - pad, cy - rad - pad, cx + rad + pad, cy + rad + pad)
            test.assertFalse(results._overlap(box, rect), f"{pid}'s label covers {other}'s dot")


# Cases whose saved runs have not been recorded yet. The coverage check stays strict for
# every other case, and test_pending_replay_list_is_still_needed clears entries that land.
PENDING_REPLAY: set = set()


@contextlib.contextmanager
def serving(*ids):
    """Pretend this deployment serves these models, then put everything back."""
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


def _clip_name(name: str, n: int = 20) -> str:
    import results

    return results._clip(name, n)


def speed_pareto_fixture(data=None) -> dict:
    """The block build_results.py will emit, built from the latencies already in the file."""
    import results

    d = data or results.load()
    pts = []
    for sysrow in d["decision_index"]["systems"]:
        lat = sysrow.get("latency_ms")
        if not isinstance(lat, dict) or not lat.get("median"):
            continue
        pts.append(
            {
                "id": sysrow["id"],
                "name": sysrow["name"],
                "kind": sysrow["kind"],
                "latency_ms": lat["median"],
                "p95_ms": lat.get("p95"),
                "score": sysrow["index"],
            }
        )
    pts.sort(key=lambda q: q["latency_ms"])
    best, frontier = -1.0, []
    for q in pts:
        if q["score"] > best:
            frontier.append(q["id"])
            best = q["score"]
    return {
        "label": "Speed against quality",
        "caption": "Decision Index 0.1 against median time per request.",
        "x_key": "latency_ms",
        "x_label": "Median time per request",
        "y_label": "Decision Index",
        "points": pts,
        "frontier": frontier,
        "note": "Measured in different runtimes; treat as indicative.",
    }


def reload_app():
    import importlib

    import app
    import ui

    importlib.reload(ui)
    return importlib.reload(app)


def handlers(demo) -> dict:
    """The server-side functions gradio actually wired, by name."""
    found = {}
    for block_fn in demo.fns.values():
        fn = getattr(block_fn, "fn", None)
        name = getattr(fn, "__name__", None)
        if name and name != "<lambda>":
            found.setdefault(name, fn)
    return found


def gr_skip() -> dict:
    """What an output that was left alone looks like on its way back."""
    return {"__type__": "update"}


def write_synthetic_replay(path: str, drop: int = 0) -> dict:
    """A stand-in recording of every bundled request, shaped exactly like the real one."""
    import record_replay
    import ui

    eng = blink.MockEngine()
    cache = {}
    for name, state, qs in record_replay.bundled_requests():
        key = blink.request_key(state, qs)
        if key in cache:
            cache[key]["names"].append(name)
            continue
        lookup = state.strip() if isinstance(state, str) else None
        raw, n = eng.logits(state, qs, ui.BIAS_BY_STATE.get(lookup) if lookup else None)
        cache[key] = {
            "names": [name],
            "logits": {k: [round(float(x), 5) for x in v] for k, v in raw.items()},
            "input_tokens": int(n),
            "latency_ms": round(38.0 + (int(key[:4], 16) % 90) * 0.9, 1),
        }
    for key in list(cache)[:drop]:
        del cache[key]
    data = {
        "model": "thegovind/blink-4b",
        "engine": "synthetic: MockEngine logits, tests only",
        "requests": cache,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


class TestOnePalette(unittest.TestCase):
    def test_dark_system_draws_like_light(self):
        need_gradio(self)
        app = reload_app()
        values = vars(app.THEME)
        dark = [n for n in values if n.endswith("_dark") and n[: -len("_dark")] in values]
        self.assertGreater(len(dark), 50)
        for name in dark:
            self.assertEqual(values[name], values[name[: -len("_dark")]], name)
        self.assertIn("body_text_color_dark", dark)


class TestOptionText(unittest.TestCase):
    def test_empty_description_uses_key(self):
        self.assertEqual(blink.option_text("refund", ""), "refund")
        self.assertEqual(blink.option_text("refund", None), "refund")

    def test_generic_key_uses_description_only(self):
        for k in ("A", "AB", "option_3", "option 3", "opt7", "12"):
            self.assertEqual(blink.option_text(k, "Escalate now"), "Escalate now")

    def test_key_equal_to_description_is_not_doubled(self):
        self.assertEqual(blink.option_text("Billing", " billing "), " billing ")

    def test_named_key_is_prefixed(self):
        self.assertEqual(blink.option_text("billing", "Money questions"), "billing: Money questions")

    def test_non_string_description_is_json(self):
        self.assertEqual(blink.option_text("payload", {"a": 1}), 'payload: {\n  "a": 1\n}')

    def test_single_and_double_letter_keys_are_generic(self):
        self.assertEqual(blink.option_text("b", "Bee"), "Bee")
        self.assertEqual(blink.option_text("zz", "Zed"), "Zed")


class TestQuestionOptions(unittest.TestCase):
    def test_choice_dict_preserves_order(self):
        q = {"type": "choice", "criteria": {"beta": "Bee", "alpha": "Ay", "gamma": ""}}
        self.assertEqual(
            blink.question_options(q),
            [("beta", "beta: Bee"), ("alpha", "alpha: Ay"), ("gamma", "gamma")],
        )

    def test_choice_list_of_keys(self):
        q = {"type": "choice", "criteria": ["x", "y"]}
        self.assertEqual(blink.question_options(q), [("x", "x"), ("y", "y")])

    def test_noul_with_and_without_criteria(self):
        self.assertEqual(
            blink.question_options({"type": "noul"}), [("yes", "Yes"), ("no", "No")]
        )
        q = {"type": "noul", "criteria": {"true": "Is spam", "false": "Is not spam"}}
        self.assertEqual(
            blink.question_options(q),
            [("yes", "Yes — Is spam"), ("no", "No — Is not spam")],
        )

    def test_noul_accepts_yes_no_aliases(self):
        q = {"type": "noul", "criteria": {"yes": "Eligible", "no": "Not eligible"}}
        self.assertEqual(blink.question_options(q)[0], ("yes", "Yes — Eligible"))

    def test_score_levels_are_numbered_from_zero(self):
        q = {"type": "score", "criteria": ["none", "some", "lots"]}
        self.assertEqual(
            blink.question_options(q),
            [("0", "Level 0: none"), ("1", "Level 1: some"), ("2", "Level 2: lots")],
        )

    def test_score_level_bounds(self):
        with self.assertRaises(blink.BlinkError):
            blink.question_options({"type": "score", "criteria": ["only one"]})
        with self.assertRaises(blink.BlinkError):
            blink.question_options({"type": "score", "criteria": [str(i) for i in range(11)]})

    def test_rejects_unknown_type_and_duplicate_keys(self):
        with self.assertRaises(blink.BlinkError):
            blink.question_options({"type": "rank", "criteria": ["a"]})
        with self.assertRaises(blink.BlinkError):
            blink.question_options({"type": "choice", "criteria": {1: "a", "1": "b"}})

    def test_option_ceiling(self):
        q = {"type": "choice", "criteria": {f"k{i}": "" for i in range(blink.MAX_OPTIONS + 1)}}
        with self.assertRaises(blink.BlinkError):
            blink.question_options(q)


class TestLabels(unittest.TestCase):
    def test_pool_is_a_to_z_then_two_letter(self):
        self.assertEqual(blink.LABEL_POOL[:26], list(string.ascii_uppercase))
        self.assertEqual(blink.LABEL_POOL[26:29], ["AA", "AB", "AC"])

    def test_pool_covers_the_option_ceiling_and_is_unique(self):
        self.assertGreaterEqual(len(blink.LABEL_POOL), blink.MAX_OPTIONS)
        head = blink.LABEL_POOL[: blink.MAX_OPTIONS]
        self.assertEqual(len(set(head)), blink.MAX_OPTIONS)


class TestUserMessage(unittest.TestCase):
    def test_shape_is_evidence_criterion_options(self):
        q = {
            "type": "choice",
            "instructions": "  Pick one.  ",
            "criteria": {"alpha": "Ay", "beta": "Bee"},
        }
        items = blink.question_options(q)
        msg = json.loads(blink.user_message({"ticket": 7}, q, ["A", "B"], items))
        self.assertEqual(list(msg), ["evidence", "criterion", "options"])
        self.assertEqual(msg["evidence"], {"ticket": 7})
        self.assertEqual(msg["criterion"], "Pick one.")
        self.assertEqual(
            msg["options"],
            [
                {"letter": "A", "description": "alpha: Ay"},
                {"letter": "B", "description": "beta: Bee"},
            ],
        )

    def test_unicode_is_not_escaped(self):
        q = {"type": "choice", "instructions": "café", "criteria": {"x": ""}}
        self.assertIn("café", blink.user_message("naïve", q, ["A"], blink.question_options(q)))


class TestAssembly(unittest.TestCase):
    def test_choice_argmax_and_confidence(self):
        q = {"type": "choice", "criteria": {"a": "", "b": "", "c": "", "d": ""}}
        a = blink.answer_for(q, ["a", "b", "c", "d"], [0.1, 0.7, 0.1, 0.1])
        self.assertEqual(a["choice"], "b")
        self.assertAlmostEqual(a["confidence"], (0.7 - 0.25) / 0.75)
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0)

    def test_uniform_choice_has_zero_confidence(self):
        q = {"type": "choice", "criteria": {"a": "", "b": ""}}
        self.assertAlmostEqual(blink.answer_for(q, ["a", "b"], [0.5, 0.5])["confidence"], 0.0)

    def test_choice_ties_go_to_the_first_listed_option(self):
        q = {"type": "choice", "criteria": {"a": "", "b": ""}}
        self.assertEqual(blink.answer_for(q, ["a", "b"], [0.5, 0.5])["choice"], "a")

    def test_noul_reports_probability_of_yes(self):
        a = blink.answer_for({"type": "noul"}, ["yes", "no"], [0.82, 0.18])
        self.assertAlmostEqual(a["noul"], 0.82)

    def test_score_is_the_expected_level(self):
        q = {"type": "score", "criteria": ["none", "some", "lots"]}
        a = blink.answer_for(q, ["0", "1", "2"], [0.2, 0.3, 0.5])
        self.assertAlmostEqual(a["score"], 0 * 0.2 + 1 * 0.3 + 2 * 0.5)
        self.assertEqual(a["choice"], "2")
        self.assertEqual(a["legend"]["1"], "some")

    def test_score_confidence_is_the_choice_formula_over_levels(self):
        """TypeSafe's score answers carry confidence; blink's is the formula its choice answers use, (K*p_max-1)/(K-1),
        which reproduces TypeSafe's documented examples. It comes after the fields blink always had."""
        q = {"type": "score", "criteria": ["none", "some", "lots"]}
        a = blink.answer_for(q, ["0", "1", "2"], [0.2, 0.3, 0.5])
        self.assertAlmostEqual(a["confidence"], (0.5 - 1 / 3) / (1 - 1 / 3))
        self.assertEqual(list(a), ["type", "score", "probabilities", "legend", "choice", "confidence"])
        for probs, documented in (([0.0, 0.76, 0.24], 0.64), ([0.0, 0.72, 0.28], 0.58), ([0.0, 0.57, 0.43], 0.35)):
            got = blink.answer_for(q, ["0", "1", "2"], probs)["confidence"]
            self.assertAlmostEqual(got, documented, delta=0.0051)  # docs.typesafe.ai/primitives/score, 2 decimals

    def test_score_confidence_stays_in_range(self):
        for n in range(2, 11):
            q = {"type": "score", "criteria": [str(i) for i in range(n)]}
            keys = [str(i) for i in range(n)]
            flat = blink.answer_for(q, keys, [1.0] * n)["confidence"]
            sure = blink.answer_for(q, keys, [0.0] * (n - 1) + [1.0])["confidence"]
            self.assertTrue(0.0 <= flat <= 1e-12, (n, flat))
            self.assertEqual(sure, 1.0)

    def test_probabilities_are_renormalised(self):
        q = {"type": "choice", "criteria": {"a": "", "b": ""}}
        a = blink.answer_for(q, ["a", "b"], [2.0, 2.0])
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0)


class TestSoftmax(unittest.TestCase):
    def test_sums_to_one_and_is_monotone(self):
        p = blink.softmax([1.0, 2.0, 3.0], 1.0)
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertEqual(p, sorted(p))

    def test_large_logits_do_not_overflow(self):
        p = blink.softmax([1000.0, 999.0], 1.0)
        self.assertAlmostEqual(sum(p), 1.0)

    def test_high_temperature_flattens(self):
        sharp = blink.softmax([0.0, 4.0], 1.0)
        flat = blink.softmax([0.0, 4.0], 8.0)
        self.assertLess(max(flat), max(sharp))


class TestValidate(unittest.TestCase):
    def test_rejects_empty_and_non_object(self):
        for bad in ({}, [], None, "q"):
            with self.assertRaises(blink.BlinkError):
                blink.validate(bad)

    def test_rejects_too_many_questions(self):
        with self.assertRaises(blink.BlinkError):
            blink.validate({f"q{i}": {"type": "noul"} for i in range(blink.MAX_QUESTIONS + 1)})

    def test_accepts_a_mixed_request(self):
        blink.validate(
            {
                "route": {"type": "choice", "criteria": {"a": "Ay", "b": "Bee"}},
                "refund": {"type": "noul"},
                "urgency": {"type": "score", "criteria": ["low", "high"]},
            }
        )


class TestDecideMock(unittest.TestCase):
    QUESTIONS = {
        "route": {"type": "choice", "instructions": "Route it.", "criteria": {"billing": "Money", "tech": "Broken"}},
        "refund": {"type": "noul", "instructions": "Refund eligible?"},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["none", "low", "high"]},
    }

    def test_engine_is_mock(self):
        self.assertEqual(blink.engine().name, "mock")

    def test_answers_cover_every_question_with_the_right_type(self):
        out = blink.decide("Card was charged twice.", self.QUESTIONS)
        self.assertEqual(set(out["answers"]), set(self.QUESTIONS))
        self.assertEqual(out["answers"]["route"]["type"], "choice")
        self.assertEqual(out["answers"]["refund"]["type"], "noul")
        self.assertEqual(out["answers"]["urgency"]["type"], "score")

    def test_probabilities_are_normalised_everywhere(self):
        out = blink.decide("Card was charged twice.", self.QUESTIONS)
        for a in out["answers"].values():
            self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0, places=9)

    def test_score_stays_inside_the_level_range(self):
        s = blink.decide("Card was charged twice.", self.QUESTIONS)["answers"]["urgency"]["score"]
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 2.0)

    def test_meta_reports_no_generated_tokens(self):
        meta = blink.decide("x", self.QUESTIONS)["meta"]
        self.assertEqual(meta["generated_tokens"], 0)
        self.assertGreater(meta["input_tokens"], 0)
        self.assertEqual(meta["model"], blink.MODEL_ID)

    def test_mock_is_deterministic(self):
        a = blink.decide("Card was charged twice.", self.QUESTIONS)["answers"]
        b = blink.decide("Card was charged twice.", self.QUESTIONS)["answers"]
        self.assertEqual(a, b)

    def test_different_states_give_different_answers(self):
        a = blink.decide("My card was charged twice, please refund.", self.QUESTIONS)["answers"]
        b = blink.decide("The mobile app crashes when I open settings.", self.QUESTIONS)["answers"]
        self.assertNotEqual(a["route"]["probabilities"], b["route"]["probabilities"])

    def test_json_state_is_accepted(self):
        out = blink.decide({"subject": "double charge", "plan": "pro"}, self.QUESTIONS)
        self.assertEqual(set(out["answers"]), set(self.QUESTIONS))

    def test_temperature_override_flattens_the_distribution(self):
        sharp = blink.decide("x", self.QUESTIONS, temperature=0.5)["answers"]["route"]
        flat = blink.decide("x", self.QUESTIONS, temperature=5.0)["answers"]["route"]
        self.assertLess(max(flat["probabilities"].values()), max(sharp["probabilities"].values()))

    def test_non_positive_temperature_is_rejected(self):
        with self.assertRaises(blink.BlinkError):
            blink.decide("x", self.QUESTIONS, temperature=0.0)

    def test_many_options_exercise_two_letter_labels(self):
        q = {"q": {"type": "choice", "criteria": {f"k{i}": f"Option {i}" for i in range(60)}}}
        out = blink.decide("pick one", q)
        self.assertEqual(len(out["answers"]["q"]["probabilities"]), 60)


class TestExamplesAndResults(unittest.TestCase):
    def test_every_bundled_example_runs(self):
        import examples

        for case in examples.USE_CASES:
            for ex in case.examples:
                out = blink.decide(ex.state, case.questions)
                self.assertEqual(set(out["answers"]), set(case.questions))
                self.assertIsInstance(case.verdict(out["answers"]), tuple)

    def test_playground_default_parses_and_runs(self):
        import examples

        qs = json.loads(examples.PLAYGROUND_QUESTIONS)
        types = {q["type"] for q in qs.values()}
        self.assertEqual(types, {"choice", "noul", "score"})
        blink.decide(examples.PLAYGROUND_STATE, qs)

    def test_results_json_is_real_and_complete(self):
        import results

        data = results.load()
        self.assertFalse(data["placeholder"])
        self.assertTrue(data["decision_index"]["systems"])
        self.assertTrue(data["jevbench"]["systems"])
        self.assertTrue(data["pareto"]["points"])

    def test_every_chart_renders_svg(self):
        import results

        data = results.load()
        for html in (
            results.decision_index_chart(data),
            results.jevbench_chart(data),
            results.pareto_chart(data),
        ):
            self.assertIn("<svg", html)
            self.assertNotIn("http://", html)
            self.assertNotIn("https://", html)

    def test_no_placeholder_wording_is_rendered(self):
        import results

        data = results.load()
        page = "".join(
            (
                results.decision_index_chart(data),
                results.jevbench_chart(data),
                results.pareto_chart(data),
            )
        ).lower()
        for word in ("placeholder", "fabricated", "made up", "tbd"):
            self.assertNotIn(word, page)


class TestDecisionIndexFigure(unittest.TestCase):
    """The board is a real leaderboard snapshot: 33 systems, some without panel data."""

    def setUp(self):
        import results

        self.results = results
        self.data = results.load()
        self.di = self.data["decision_index"]

    def test_focus_keeps_ours_and_the_reference_and_trims_the_tail(self):
        shown = self.results.focus(self.di["systems"])
        kinds = {s["kind"] for s in shown}
        self.assertIn("ours", kinds)
        self.assertIn("reference", kinds)
        self.assertLess(len(shown), len(self.di["systems"]))
        self.assertEqual([s["index"] for s in shown], sorted((s["index"] for s in shown), reverse=True))

    def test_focus_works_with_and_without_a_second_ours_row(self):
        systems = [dict(s) for s in self.di["systems"]]
        without = self.results.focus(systems)
        four_b = dict(without[0], id="blink-4b", name="blink-4b", params="4B",
                      served_params=4659865088, index=54.9, skill=41.0, breadth=39.0)
        with_4b = self.results.focus(systems + [four_b])
        ids = [s["id"] for s in with_4b]
        self.assertIn("blink-4b", ids)
        self.assertIn("blink-27b", ids)
        self.assertEqual(len(with_4b), len(without) + 1)

    def test_focus_keeps_a_small_class_entry_outside_the_top(self):
        base = [
            {"id": "ours", "name": "ours", "kind": "ours", "params": "27B", "served_params": 27e9,
             "index": 62.5, "skill": 50.0, "breadth": 48.0, "areas": None},
            {"id": "ref", "name": "ref", "kind": "reference", "params": "closed", "served_params": None,
             "index": 59.5, "skill": 46.0, "breadth": 44.0, "areas": None},
        ]
        big = [
            {"id": f"big{i}", "name": f"big{i}", "kind": "open", "params": "70B",
             "served_params": 70e9, "index": 58.0 - i, "skill": 40.0, "breadth": 38.0, "areas": None}
            for i in range(self.results.TOP_OPEN + 4)
        ]
        small = {"id": "small4b", "name": "small4b", "kind": "open", "params": "4B",
                 "served_params": 4.6e9, "index": 20.0, "skill": 9.0, "breadth": 8.0, "areas": None}
        ids = [s["id"] for s in self.results.focus(base + big + [small])]
        self.assertIn("small4b", ids)
        self.assertEqual(len(ids), 2 + self.results.TOP_OPEN + 1)

    def test_areas_none_renders_a_dash_instead_of_bars(self):
        shown = self.results.focus(self.di["systems"])
        self.assertTrue(any(s["areas"] is None for s in shown), "fixture lost its null-area rows")
        svg = self.results._area_bars(shown, self.di["areas"])
        n_missing = sum(1 for s in shown if s["areas"] is None)
        self.assertEqual(svg.count(">\u2014</text>"), n_missing * len(self.di["areas"]))

    def test_every_area_none_still_renders(self):
        systems = [dict(s, areas=None) for s in self.di["systems"]]
        data = dict(self.data, decision_index=dict(self.di, systems=systems))
        self.assertIn("<svg", self.results.decision_index_chart(data))

    def test_skill_and_breadth_appear_for_every_shown_row(self):
        chart = self.results.decision_index_chart(self.data)
        for s in self.results.focus(self.di["systems"]):
            self.assertIn(f"{s['skill']:.1f} \u00b7 {s['breadth']:.1f}", chart)

    def test_the_full_board_is_behind_a_disclosure(self):
        chart = self.results.decision_index_chart(self.data)
        self.assertIn("<details", chart)
        for s in self.di["systems"]:
            self.assertIn(results_escape(s["name"]), chart)

    def test_training_overlap_is_disclosed_and_starts_collapsed(self):
        chart = self.results.decision_index_chart(self.data)
        self.assertIn("Training overlap with the suite", chart)
        for name in ("ContractNLI", "iSarcasmEval", "VAST", "Amazon ESCI", "Humicroedit"):
            self.assertIn(name, chart)
        self.assertIn("Every final training row was audited", chart)
        self.assertIn("30 or more characters in any field", chart)
        self.assertIn("20 or more suite requests were treated as templates", chart)
        self.assertIn("16 rows in each of blink-27b and blink-mimo-9b shared a passage with 31 suite requests", chart)
        self.assertIn("did not search long-document bodies or option text", chart)
        self.assertIn("Language area", chart)
        self.assertIn("not leaderboard submissions", chart)
        self.assertNotIn("<details open", chart)


class TestEditionAndQualification(unittest.TestCase):
    """Everything measured is on 0.1, and the board has moved on. Both must be visible."""

    def setUp(self):
        import results
        import ui

        self.results = results
        self.ui = ui
        self.data = results.load()
        self.di = self.data["decision_index"]
        self.blocks = "".join(ui.results_blocks(self.data))

    def test_chart_is_titled_with_the_edition_it_measures(self):
        self.assertEqual(self.di["label"], "Decision Index 0.1")
        self.assertIn(">Decision Index 0.1</h2>", self.blocks)
        self.assertIn(results_escape(self.di["edition"]), self.blocks)

    def test_the_move_to_02_is_stated_in_the_open(self):
        self.assertIn(results_escape(self.ui.EDITION_MOVED), self.blocks)
        self.assertIn("2026-09-24", self.ui.EDITION_MOVED)
        self.assertNotIn(f"<details{self.ui.EDITION_MOVED}", self.blocks)

    def test_in_domain_overlap_is_visible_not_collapsed(self):
        line = self.ui.in_domain_line(self.di)
        self.assertTrue(line.startswith('<p class="blk-note">'))
        for name in ("ContractNLI", "iSarcasmEval", "VAST", "Amazon ESCI", "Humicroedit"):
            self.assertIn(name, line)
        ours = next(s for s in self.di["systems"] if s["kind"] == "ours")
        ref = next(s for s in self.di["systems"] if s["id"] == self.di["reference"])
        self.assertIn(f"{ours['index_language_equalized']:.2f}", line)
        self.assertIn(f"{ref['index']:.2f}", line)
        self.assertIn(line, self.blocks)

    def test_in_domain_line_is_dropped_when_the_field_is_absent(self):
        systems = [{k: v for k, v in s.items() if k != "index_language_equalized"}
                   for s in self.di["systems"]]
        self.assertEqual(self.ui.in_domain_line(dict(self.di, systems=systems)), "")

    def test_the_lead_claim_names_the_model_it_belongs_to(self):
        chart = self.results.decision_index_chart(self.data)
        self.assertIn("Most of blink-27b's lead", chart)
        self.assertNotIn("Most of the lead over Jev", chart)

    def test_skill_and_breadth_are_defined_by_what_they_measure(self):
        page = self.blocks + self.results.decision_index_chart(self.data)
        self.assertIn("above each metric's chance baseline", page)
        self.assertIn("shifted geometric mean", page)
        self.assertNotIn("rewards the requests most", page)


V02_FIXTURE = {
    "label": "Decision Index 0.2 · shared benchmarks",
    "caption": "Area means over the 33 of 40 panel benchmarks that 0.2 keeps from 0.1. Not a 0.2 score.",
    "snapshot": "2026-09-24T00:00:00+00:00", "shared": 33, "panel": 40,
    "missing": ["MMLU-Pro", "BBH", "RAGTruth"], "jev_official_02": 63.87,
    "systems": [
        {"id": "open-a", "name": "Open A", "kind": "open", "partial": 60.9, "areas": {}, "official_02": 63.37},
        {"id": "blink-27b", "name": "blink-27b", "kind": "ours", "partial": 60.8, "areas": {}},
        {"id": "jev-1.13.0", "name": "Jev 1.13.0", "kind": "reference", "partial": 60.5, "areas": {}, "official_02": 63.87},
        {"id": "open-b", "name": "Open B", "kind": "open", "partial": 56.5, "areas": {}, "official_02": 59.17},
    ],
}


class TestV02Partial(unittest.TestCase):
    """A same-benchmark subset, never presented as a 0.2 score."""

    def setUp(self):
        import results
        import ui

        self.results = results
        self.ui = ui
        self.data = results.load()
        if not self.data.get("decision_index_v02_partial"):
            # The published results withhold this section; exercise the renderer with a fixture.
            self.data = dict(self.data, decision_index_v02_partial=V02_FIXTURE)
        self.v02 = self.data["decision_index_v02_partial"]
        self.html = results.v02_partial_chart(self.data)

    def test_hidden_when_the_section_is_null(self):
        self.assertEqual(self.results.v02_partial_chart({"decision_index_v02_partial": None}), "")
        blocks = "".join(self.ui.results_blocks(dict(self.data, decision_index_v02_partial=None)))
        self.assertNotIn(results_escape(self.v02["label"]), blocks)

    def test_rows_are_the_top_of_the_board_with_ours_and_the_reference(self):
        shown = self.results._top_rows(self.v02["systems"], "partial", self.results.V02_TOP)
        self.assertIn("ours", {s["kind"] for s in shown})
        self.assertIn("reference", {s["kind"] for s in shown})
        self.assertLessEqual(len(shown), self.results.V02_TOP + 2)
        self.assertEqual([s["partial"] for s in shown], sorted((s["partial"] for s in shown), reverse=True))
        for s in shown:
            self.assertIn(f"{s['partial']:.2f}", self.html)

    def test_official_02_is_a_separate_muted_column(self):
        self.assertIn(">0.2 official</text>", self.html)
        ours = next(s for s in self.v02["systems"] if s["kind"] == "ours")
        self.assertIsNone(ours.get("official_02"))
        self.assertIn("not on the board", self.html)
        ref = next(s for s in self.v02["systems"] if s["kind"] == "reference")
        self.assertIn(f"{ref['official_02']:.2f}", self.html)

    def test_the_page_says_it_is_not_a_02_score(self):
        blocks = "".join(self.ui.results_blocks(self.data))
        self.assertIn("not a 0.2 score", self.html)
        self.assertIn("Not a 0.2 score", blocks)
        self.assertIn(f"{self.v02['shared']} of {self.v02['panel']}", self.html)

    def test_the_missing_benchmarks_are_listed(self):
        for name in self.v02["missing"]:
            self.assertIn(results_escape(name), self.html)


class TestClaimsAndLabels(unittest.TestCase):
    """The wording the review asked for, checked where a reader would see it."""

    def setUp(self):
        import results
        import ui

        self.results = results
        self.ui = ui
        self.data = results.load()

    def test_jevbench_section_disclaims_any_official_reading(self):
        blocks = "".join(self.ui.results_blocks(self.data))
        self.assertIn("development proxies computed here", blocks)
        self.assertIn("No official score, rank, or parity", blocks)
        self.assertIn(self.results.JB_OFFICIAL_TITLE, blocks)

    def test_public_notes_come_from_the_data(self):
        notes = self.data["jevbench"]["notes"]
        html = self.results.jevbench_chart(self.data)
        hc = notes["hard_correct"]
        self.assertIn(f"{hc['blink-4b']}/{hc['n']}", html)
        self.assertIn(f"{hc['jevk5-0.2.0']}/{hc['n']}", html)
        self.assertIn(f"{hc['jevk5_native_published']}/{hc['n']}", html)
        self.assertIn("No hard-accuracy advantage is claimed", html)
        for key in ("selection_note", "speed_note", "cost_note"):
            self.assertIn(results_escape(notes[key]), html)

    def test_public_notes_survive_a_board_without_them(self):
        data = dict(self.data, jevbench=dict(self.data["jevbench"], notes=None))
        self.assertIn("<svg", self.results.jevbench_chart(data))

    def test_pareto_caption_comes_from_the_data(self):
        blocks = "".join(self.ui.results_blocks(self.data))
        self.assertIn(results_escape(self.data["pareto"]["caption"]), blocks)
        self.assertIn("not a cost or latency result", self.data["pareto"]["caption"])

    def test_probabilities_are_described_as_model_probabilities(self):
        how = "".join(self.ui.how_blocks())
        self.assertIn("softmax(logits / T) with T = 1.0", how)
        self.assertIn("not certified probabilities of being correct", how)
        self.assertIn("concentration score", how)
        self.assertNotIn("same kind of 0.7", how)

    def test_context_cap_is_stated(self):
        how = "".join(self.ui.how_blocks())
        self.assertIn("131,072 tokens per question", how)
        self.assertIn("37,906", how)
        self.assertIn("refused, not truncated", how)

    def test_weights_are_not_given_a_licence_in_the_ui(self):
        page = "".join(self.ui.how_blocks()) + "".join(self.ui.results_blocks(self.data))
        page += self.ui.masthead() + self.ui.footer()
        self.assertIn("Weights are for non-commercial research and evaluation; app code is Apache-2.0.", page)
        self.assertEqual(page.count("Apache-2.0"), 1)
        self.assertIn("non-commercial research use", page)

    def test_footer_is_the_licence_line_only(self):
        foot = self.ui.footer()
        self.assertIn("Weights are for non-commercial research and evaluation", foot)
        self.assertNotIn("Personal research release", foot)
        self.assertNotIn("affiliated", foot)

    def test_verdicts_say_whose_answers_they_read(self):
        saved = self.ui.render_verdict("go", "Send to billing", "Because.",
                                       {"engine": "replay", "model": "thegovind/blink-4b", "latency_ms": 41.5})
        self.assertIn(">saved run \u00b7 blink-4b</em>", saved)
        self.assertIn("recorded earlier", saved)  # a saved run is the model's own output, said on hover
        live = self.ui.render_verdict("go", "Send to billing", "Because.",
                                      {"engine": "torch", "model": "thegovind/blink-4b", "latency_ms": 812.4})
        self.assertIn(">live \u00b7 blink-4b \u00b7 812 ms</em>", live)
        mock = self.ui.render_verdict("go", "Send to billing", "Because.", {"engine": "mock", "model": "m/x"})
        self.assertIn(">mock \u00b7 x</em>", mock)
        for html in (saved, live, mock):
            self.assertNotIn("simulated", html)

    def test_demo_label_only_appears_when_answers_are_recorded(self):
        self.assertEqual(self.ui.engine_name(), "mock")
        self.assertEqual(self.ui.demo_label(), "")
        self.assertIn("Saved examples", self.ui.DEMO_LABEL)
        self.assertIn("live", self.ui.LIVE_LABEL)
        for label in (self.ui.DEMO_LABEL, self.ui.LIVE_LABEL):
            self.assertNotIn("GPU", label)
            self.assertNotIn("accelerator", label)


class TestJevBenchFigure(unittest.TestCase):
    """Official sealed scores and our public-item estimate must never share a scale."""

    def setUp(self):
        import results

        self.results = results
        self.data = results.load()
        self.jb = self.data["jevbench"]
        self.html = results.jevbench_chart(self.data)

    def test_two_named_groups(self):
        self.assertIn(self.results.JB_OFFICIAL_TITLE, self.html)
        self.assertIn(self.results.JB_PUBLIC_TITLE, self.html)
        self.assertIn("sealed", self.results.JB_OFFICIAL_TITLE)
        self.assertIn("proxy", self.results.JB_PUBLIC_TITLE)
        self.assertIn('class="blk-duo"', self.html)

    def test_official_none_is_shown_as_no_official_score_without_a_number(self):
        ours = next(s for s in self.jb["systems"] if s["kind"] == "ours")
        self.assertIsNone(ours["official"])
        self.assertIn("no official score published", self.html)
        head, _, tail = self.html.partition(self.results.JB_PUBLIC_TITLE)
        self.assertIn("\u2014", head)
        self.assertNotIn(f"{ours['public_estimate']:.2f}", head)
        self.assertIn(f"{ours['public_estimate']:.2f}", tail)

    def test_official_scores_only_appear_in_the_official_group(self):
        head, _, tail = self.html.partition(self.results.JB_PUBLIC_TITLE)
        for s in self.jb["systems"]:
            if s.get("official"):
                self.assertIn(f"{s['official']['score']:.2f}", head)
                self.assertNotIn(f"{s['official']['score']:.2f}", tail)

    def test_public_group_holds_only_systems_we_measured(self):
        _, _, tail = self.html.partition(self.results.JB_PUBLIC_TITLE)
        tail = tail.split('<div class="blk-tablewrap')[0]
        unmeasured = [s for s in self.jb["systems"] if s["public_estimate"] is None]
        self.assertTrue(unmeasured)
        for s in unmeasured:
            self.assertNotIn(f'>{results_escape(s["name"])}<', tail)

    def test_mini_table_reports_hard_accuracy_ece_and_tvd(self):
        self.assertIn("Hard ECE", self.html)
        self.assertIn("TVD", self.html)
        for s in self.jb["systems"]:
            if s.get("public"):
                self.assertIn(f"{s['public']['hard'] * 100:.1f}%", self.html)
                self.assertIn(f"{s['public']['hard_ece']:.3f}", self.html)
                self.assertIn(f"{s['public']['tvd']:.3f}", self.html)

    def test_survives_a_board_with_no_official_scores_at_all(self):
        systems = [dict(s, official=None) for s in self.jb["systems"]]
        data = dict(self.data, jevbench=dict(self.jb, systems=systems))
        html = self.results.jevbench_chart(data)
        self.assertIn("no official score published", html)
        self.assertIn("<svg", html)


class TestParetoFigure(unittest.TestCase):

    def setUp(self):
        import results

        self.results = results
        self.data = results.load()
        self.pa = self.data["pareto"]
        self.html = results.pareto_chart(self.data)

    def test_x_axis_follows_x_key_not_a_hardcoded_cost(self):
        self.assertEqual(self.pa["x_key"], "params_b")
        self.assertIn("Served parameters", self.html)
        self.assertNotIn("$", self.html)
        for tick in ("100M", "300M", "1B", "3B", "10B", "30B"):
            self.assertIn(f">{tick}</text>", self.html)

    def test_cost_x_key_still_formats_as_money(self):
        pts = [dict(p, cost_per_1000=p["params_b"] / 400.0) for p in self.pa["points"]]
        data = dict(
            self.data,
            pareto=dict(self.pa, x_key="cost_per_1000", x_label="Cost per 1,000", points=pts),
        )
        html = self.results.pareto_chart(data)
        self.assertIn("$", html)
        self.assertIn("Cost per 1,000", html)

    def test_reference_is_a_dashed_line_with_a_label_and_no_x_position(self):
        ref = self.pa["reference"]
        self.assertIn("blk-ref", self.html)
        self.assertIn(results_escape(ref["name"]), self.html)
        self.assertNotIn("params_b", self.html)

    def test_only_notable_points_are_labelled_and_the_rest_carry_a_tooltip(self):
        labelled = {m.group(5) for m in LABEL_RE.finditer(self.html)}
        self.assertLess(len(labelled), len(self.pa["points"]))
        self.assertIn("blink-27b", labelled)
        self.assertEqual(self.html.count("<title>"), len(self.pa["points"]))

    def test_nothing_overlaps_and_every_label_points_at_its_own_dot(self):
        assert_chart_geometry(self, self.html, self.pa["points"])

    def test_our_rows_keep_their_labels(self):
        labelled = {m.group(2) for m in LABEL_RE.finditer(self.html)}
        for q in self.pa["points"]:
            if q["kind"] == "ours":
                self.assertIn(q["id"], labelled)

    def test_frontier_staircase_covers_every_frontier_id(self):
        self.assertIn("blk-front", self.html)
        by_id = {p["id"]: p for p in self.pa["points"]}
        for i in self.pa["frontier"]:
            self.assertIn(i, by_id)


class TestSpeedPareto(unittest.TestCase):
    """The same chart on a time axis, and nothing at all until the block exists."""

    def setUp(self):
        import results
        import ui

        self.results = results
        self.ui = ui
        self.block = speed_pareto_fixture()
        self.data = dict(results.load(), speed_pareto=self.block)
        self.html = results.speed_pareto_chart(self.data)

    def test_hidden_until_the_block_is_measured(self):
        base = {k: v for k, v in self.results.load().items() if k != "speed_pareto"}
        self.assertEqual(self.results.speed_pareto_chart(base), "")
        self.assertEqual(self.results.speed_pareto_chart(dict(base, speed_pareto=None)), "")
        blocks = "".join(self.ui.results_blocks(dict(base, speed_pareto=None)))
        self.assertNotIn("Speed against quality", blocks)

    def test_section_appears_once_the_block_is_there(self):
        blocks = "".join(self.ui.results_blocks(self.data))
        self.assertIn(results_escape(self.block["label"]), blocks)
        self.assertIn(results_escape(self.block["caption"]), blocks)
        self.assertIn(self.html, blocks)

    def test_axis_is_time_and_reads_left_to_right(self):
        self.assertIn("Faster is left", self.html)
        self.assertIn(results_escape(self.block["x_label"]), self.html)
        self.assertIn("log scale", self.html)
        for tick in ("10 ms", "30 ms", "100 ms"):
            self.assertIn(f">{tick}</text>", self.html)
        self.assertNotIn("B</text>", self.html)

    def test_millisecond_and_second_ticks(self):
        fmt = self.results._fmt_x
        self.assertEqual([fmt("latency_ms", v) for v in (10, 30, 100, 300)],
                         ["10 ms", "30 ms", "100 ms", "300 ms"])
        self.assertEqual([fmt("latency_ms", v) for v in (1000, 3000)], ["1 s", "3 s"])

    def test_the_frontier_is_drawn_and_reaches_our_row(self):
        self.assertIn("blk-front", self.html)
        self.assertIn('class="blk-halo"', self.html)
        ours = [q for q in self.block["points"] if q["kind"] == "ours"]
        self.assertTrue(ours)
        for q in ours:
            self.assertIn(results_escape(_clip_name(q["name"])), self.html)

    def test_the_reference_is_a_point_not_a_line(self):
        self.assertNotIn("blk-ref", self.html)
        ref = next(q for q in self.block["points"] if q["kind"] == "reference")
        self.assertIn(results_escape(ref["name"]), self.html)

    def test_every_point_has_a_tooltip_with_its_spread(self):
        self.assertEqual(self.html.count("<title>"), len(self.block["points"]))
        withp95 = [q for q in self.block["points"] if q.get("p95_ms")]
        self.assertTrue(withp95)
        self.assertEqual(self.html.count("p95 "), len(withp95))

    def test_nothing_overlaps_and_every_label_points_at_its_own_dot(self):
        assert_chart_geometry(self, self.html, self.block["points"])

    def test_the_frontier_and_our_rows_are_labelled(self):
        labelled = {m.group(2) for m in LABEL_RE.finditer(self.html)}
        for q in self.block["points"]:
            if q["kind"] == "ours":
                self.assertIn(q["id"], labelled)
        self.assertTrue(labelled & set(self.block["frontier"]))

    def test_no_hardware_words_anywhere_in_the_section(self):
        blocks = "".join(self.ui.results_blocks(self.data)).lower()
        for word in ("accelerator", "zerogpu", "gpu", "cuda", "accelerator", "recorded on"):
            self.assertNotIn(word, blocks)


class TestCuaProbe(unittest.TestCase):
    """One row per model, one dot per input, and a chance line to read them against."""

    def setUp(self):
        import results
        import ui

        self.results = results
        self.ui = ui
        self.data = results.load()
        self.block = self.data["cua_probe"]
        self.html = results.cua_probe_chart(self.data)

    def test_hidden_when_the_block_is_missing(self):
        self.assertEqual(self.results.cua_probe_chart({}), "")
        self.assertEqual(self.results.cua_probe_chart({"cua_probe": None}), "")
        blocks = "".join(self.ui.results_blocks(dict(self.data, cua_probe=None)))
        self.assertNotIn(results_escape(self.block["label"]), blocks)

    def test_section_appears_with_the_block(self):
        blocks = "".join(self.ui.results_blocks(self.data))
        self.assertIn(results_escape(self.block["label"]), blocks)
        self.assertIn(results_escape(self.block["caption"]), blocks)
        self.assertIn(self.html, blocks)
        self.assertIn(results_escape(self.block["note"]), blocks)

    def test_one_row_per_model_with_both_inputs(self):
        groups = self.results._cua_groups(self.block["rows"])
        self.assertEqual([g[0] for g in groups][0], "blink-4b")
        self.assertEqual(len(groups), len({r["name"].split(" + ")[0] for r in self.block["rows"]}))
        self.assertEqual(sum(len(items) for _, items in groups), len(self.block["rows"]))
        for name, _ in groups:
            self.assertIn(results_escape(name), self.html)

    def test_every_row_has_a_dot_an_interval_and_a_value(self):
        self.assertEqual(self.html.count('class="blk-pt"'), len(self.block["rows"]))
        self.assertEqual(self.html.count("blk-grow"), len(self.block["rows"]))
        for row in self.block["rows"]:
            self.assertIn(f'>{row["accuracy"]:.1f}</text>', self.html)

    def test_tooltips_carry_n_the_interval_and_ece(self):
        for row in self.block["rows"]:
            tip = (
                f'{row["name"]} \u00b7 {row["input"]} \u00b7 {row["accuracy"]:.1f}% '
                f'[{row["ci95"][0]:.1f}, {row["ci95"][1]:.1f}] \u00b7 n={row["n"]} '
                f'\u00b7 ECE {row["ece"]:.3f}'
            )
            self.assertIn(results_escape(tip), self.html)

    def test_the_chance_line_is_drawn_and_named(self):
        self.assertIn("blk-ref", self.html)
        self.assertIn(">chance</text>", self.html)
        self.assertGreater(self.block["chance"], 0)
        without = self.results.cua_probe_chart(
            {"cua_probe": dict(self.block, chance=None)}
        )
        self.assertNotIn(">chance</text>", without)

    def test_text_and_screenshot_are_told_apart(self):
        for mark in self.results.CUA_MARKS.values():
            self.assertIn(f">{mark}</text>", self.html)
        filled = self.html.count(f'fill="{self.results.OURS}"')
        self.assertGreater(filled, 0)
        self.assertIn('fill="#fff"', self.html)

    def test_our_rows_are_emphasised(self):
        ours = [r for r in self.block["rows"] if r["kind"] == "ours"]
        self.assertTrue(ours)
        self.assertIn('class="blk-name on"', self.html)
        self.assertIn('class="blk-val on"', self.html)

    def test_no_hardware_words(self):
        blocks = "".join(self.ui.results_blocks(self.data)).lower()
        for word in ("accelerator", "zerogpu", "gpu", "cuda", "accelerator", "recorded on"):
            self.assertNotIn(word, blocks)


class TestNextClickCase(unittest.TestCase):
    """An original page-action example that goes through decide like every other case."""

    def setUp(self):
        self.case = examples.CASES_BY_KEY["nextclick"]

    def test_it_is_a_use_case_with_the_two_questions(self):
        self.assertIn(self.case, examples.USE_CASES)
        self.assertEqual(list(self.case.questions), ["element", "done"])
        self.assertEqual(self.case.questions["element"]["type"], "choice")
        self.assertEqual(self.case.questions["done"]["type"], "noul")

    def test_the_options_read_as_page_elements(self):
        options = blink.question_options(self.case.questions["element"])
        self.assertEqual(len(options), 5)
        for key, shown in options:
            self.assertEqual(key, shown)
            self.assertRegex(shown, r"^(button|link|input|checkbox): ")

    def test_it_is_not_a_copy_of_a_public_benchmark(self):
        text = " ".join(e.state for e in self.case.examples).lower()
        for word in ("mind2web", "webarena", "miniwob"):
            self.assertNotIn(word, text)
        self.assertEqual(len({e.state for e in self.case.examples}), 3)

    def test_every_example_states_the_task_and_what_happened(self):
        for ex in self.case.examples:
            self.assertTrue(ex.state.startswith("Task: "))
            self.assertIn("Page: ", ex.state)
            self.assertIn("Done so far:", ex.state)

    def test_each_example_reaches_the_verdict_it_should(self):
        want = {
            "Nothing set yet": "checkbox: In stock only",
            "Ready to apply": "button: Apply filters",
        }
        for ex in self.case.examples:
            out = blink.decide(ex.state, self.case.questions,
                               bias=examples.bias_for("nextclick", ex.label))
            answers = out["answers"]
            tone, headline, detail = self.case.verdict(answers)
            self.assertIn(tone, ("go", "hold", "stop"))
            if ex.label in want:
                self.assertEqual(answers["element"]["choice"], want[ex.label])
                self.assertLess(answers["done"]["noul"], 0.5, ex.label)
            else:
                self.assertGreaterEqual(answers["done"]["noul"], 0.6, ex.label)
                self.assertIn("Nothing left", headline)

    def test_it_renders_through_the_shared_runner(self):
        import ui

        html = ui.run_use_case(self.case, self.case.examples[1].state)
        self.assertIn("blk-verdict", html)
        self.assertIn("Apply filters", html)
        self.assertIn("blk-answers", html)

    def test_it_joins_the_bundled_requests(self):
        import record_replay

        names = [n for n, _, _ in record_replay.bundled_requests()]
        for ex in self.case.examples:
            self.assertIn(f"nextclick/{ex.label}", names)


class TestLiveExperience(unittest.TestCase):
    """The first paint is a saved run; every click after it asks the model."""

    def test_skeleton_matches_the_question_count(self):
        import ui

        self.assertEqual(ui.pending_html(2).count("blk-sk-card"), 2)
        self.assertEqual(ui.pending_for('{"a": {}, "b": {}, "c": {}}').count("blk-sk-card"), 3)
        self.assertEqual(ui.pending_for("not json").count("blk-sk-card"), 3)
        self.assertEqual(ui.pending_for("[1, 2]").count("blk-sk-card"), 3)

    def test_skeleton_is_marked_busy_and_says_nothing_about_hardware(self):
        import ui

        html = ui.pending_html(3)
        self.assertIn('aria-busy="true"', html)
        self.assertIn("blk-run", html)
        for word in ("gpu", "accelerator", "zerogpu"):
            self.assertNotIn(word, html.lower())

    def test_stats_row_tags_saved_and_live_runs(self):
        import ui

        q = {"urgent": {"type": "noul", "instructions": "Now?"}}
        out = blink.decide("x", q)
        saved = ui.render_answers("x", q, dict(out, meta=dict(out["meta"], engine="replay", latency_ms=41.5)))
        self.assertIn("41.5</b> ms recorded call \u00b7 saved run", saved)
        live = ui.render_answers("x", q, dict(out, meta=dict(out["meta"], engine="torch", model_ms=12.3)))
        self.assertIn("12.3</b> ms model time \u00b7 live", live)
        self.assertNotIn("saved run", live)

    def test_first_paint_is_saved_and_clicks_are_live(self):
        need_gradio(self)
        app = reload_app()
        src = inspect.getsource(app)
        self.assertIn('prefer="saved"', src)
        # the playground, a use case, the screenshot card, and a link to one screenshot
        self.assertEqual(src.count('prefer="saved"'), 4)
        for chain in ("live(btn.click(", "live(run_btn.click("):
            self.assertIn(chain, src)
        self.assertNotIn('prefer="live"', src)

    def test_the_questions_editor_is_still_a_code_box(self):
        need_gradio(self)
        app = reload_app()
        self.assertIn('language="json"', inspect.getsource(app))


class TestMasthead(unittest.TestCase):
    def test_name_and_one_accent(self):
        import ui

        head = ui.masthead()
        self.assertIn('<h1><span class="blk-word">blink<em', head)
        self.assertIn('class="blk-mark"', head)
        self.assertIn('aria-hidden="true"', head)
        self.assertEqual(head.count("<em"), 1)
        self.assertNotIn("c" + "oreai", head.lower())

    def test_the_accent_is_styled_and_animated(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css"),
                  encoding="utf-8") as fh:
            css = fh.read()
        self.assertIn("em.blk-mark", css)
        self.assertIn(".blk-word", css)
        self.assertIn("@keyframes blk-lid", css)
        self.assertIn("@keyframes blk-shimmer", css)
        self.assertIn(".blk-sk {", css)


class TestAsk(unittest.TestCase):
    """A free-form ask becomes a request the visitor can see, and says who wrote it."""

    def setUp(self):
        import author
        import ui

        self.author = author
        self.ui = ui

    def test_the_mock_drafter_turns_an_ask_into_one_question(self):
        req = self.author.draft("How many r in strawberry")
        self.assertEqual(req["state"], "strawberry")
        self.assertEqual(len(req["questions"]), 1)
        q = next(iter(req["questions"].values()))
        self.assertEqual(q["type"], "choice")
        self.assertEqual(list(q["criteria"]), ["1", "2", "3", "4"])
        blink.validate(req["questions"])

    def test_something_it_cannot_decide_raises_a_showable_message(self):
        with self.assertRaises(self.author.AuthorError) as caught:
            self.author.draft("write me a poem")
        self.assertIn("few possible answers", str(caught.exception))

    def test_a_draft_survives_the_form_round_trip(self):
        req = self.author.draft("How many r in strawberry")
        rows = self.ui.questions_to_rows(req["questions"])
        back, problems = self.ui.rows_to_questions(rows)
        self.assertEqual(problems, [])
        self.assertEqual(back, req["questions"])

    def test_provenance_names_the_drafter_and_its_time(self):
        line = self.ui.drafted_line(
            {"model": "Qwen/Qwen3.5-4B", "generated_tokens": 77, "model_ms": 5000.0})
        self.assertIn("Qwen/Qwen3.5-4B", line)
        self.assertIn("5.0 s", line)
        # a token count read as jargon; the API's draft.author still carries it
        self.assertNotIn("generated", line)
        self.assertNotIn("77", line)
        self.assertEqual(self.ui.drafted_line(None), "")
        self.assertEqual(self.ui.drafted_line({}), "")

    def test_the_answer_carries_the_line_only_when_something_was_drafted(self):
        q = {"answer": {"type": "noul", "instructions": "Now?"}}
        out = blink.decide("x", q)
        plain = self.ui.render_answers("x", q, out)
        self.assertNotIn("blk-drafted", plain)
        drafted = self.ui.render_answers("x", q, out, {"model": "mock", "generated_tokens": 12,
                                                       "model_ms": 1234.0})
        self.assertIn("blk-drafted", drafted)
        # no "0 generated" chip on the stats row, which read as jargon; the raw response stays whole
        for html in (plain, drafted):
            stats = html.split('<div class="blk-stats">', 1)[1].split("</div>", 1)[0]
            self.assertNotIn("generated", stats)
            self.assertIn("&quot;generated_tokens&quot;: 0", html)

    def test_the_ask_link_pre_fills_and_nothing_more(self):
        self.assertEqual(self.ui.parse_ask("?ask=How%20many%20r"), "How many r")
        self.assertEqual(self.ui.parse_ask("?tab=playground&ask=hi%20there"), "hi there")
        for raw in ("", "?tab=results", "#results", None, 7, "?ask="):
            self.assertEqual(self.ui.parse_ask(raw), "")

    def test_the_status_and_the_error_share_one_slot(self):
        status = self.ui.ask_status_html(self.ui.ASK["drafting"])
        bad = self.ui.ask_error_html("nope")
        self.assertIn("blk-ask-note", status)
        self.assertIn("blk-ask-note", bad)
        self.assertIn("blk-ask-bad", bad)
        self.assertIn("blk-run", status)

    def test_the_waiting_state_can_be_labelled(self):
        self.assertIn(self.ui.ASK["deciding"], self.ui.pending_html(1, self.ui.ASK["deciding"]))
        self.assertIn(self.ui.PENDING_LABEL, self.ui.pending_html(1))

    def test_the_ask_box_is_there_when_the_drafter_is(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            boxes = [b for b in demo.blocks.values()
                     if "blk-ask" in (getattr(b, "elem_classes", None) or [])]
            buttons = [b for b in demo.blocks.values()
                       if isinstance(b, gr.Button) and b.value == self.ui.ASK["button"]]
        finally:
            demo.close()
        self.assertTrue(self.author.enabled())
        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(buttons), 1)

    def test_turning_the_drafter_off_hides_the_ask(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        import author

        with unittest.mock.patch.object(author, "enabled", lambda: False):
            app = reload_app()
            demo = app.build()
            try:
                boxes = [b for b in demo.blocks.values()
                         if "blk-ask" in (getattr(b, "elem_classes", None) or [])]
                buttons = [b for b in demo.blocks.values()
                           if isinstance(b, gr.Button) and b.value == self.ui.ASK["button"]]
                rows = [b for b in demo.blocks.values()
                        if "blk-qrow" in (getattr(b, "elem_classes", None) or [])]
            finally:
                demo.close()
        self.assertEqual(boxes, [])
        self.assertEqual(buttons, [])
        self.assertEqual(len(rows), self.ui.MAX_FORM_QUESTIONS)  # the form stays

    def test_the_ask_endpoint_returns_the_draft_and_the_decision(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        out = app.ask_api("How many r in strawberry")
        self.assertEqual(sorted(out), ["answers", "draft", "usage"])
        self.assertEqual(sorted(out["draft"]), ["author", "questions", "state"])
        self.assertEqual(out["draft"]["state"], "strawberry")
        self.assertEqual(list(out["answers"]), list(out["draft"]["questions"]))
        self.assertEqual(out["usage"]["generated_tokens"], 0)
        self.assertIn("generated_tokens", out["draft"]["author"])
        self.assertIn("model", out["draft"]["author"])

    def test_the_ask_endpoint_passes_the_model_through(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        with serving("thegovind/blink-4b", "thegovind/blink-mimo-9b"):
            app = reload_app()
            out = app.ask_api("How many r in strawberry", "thegovind/blink-mimo-9b")
        self.assertEqual(out["usage"]["model"], "thegovind/blink-mimo-9b")

    def test_a_refused_ask_comes_back_as_a_gradio_error(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        for bad in ("write me a poem", ""):
            with self.assertRaises(gr.Error) as caught:
                app.ask_api(bad)
            self.assertTrue(str(caught.exception))

    def test_the_ask_endpoint_is_registered_both_ways(self):
        need_gradio(self)
        app = reload_app()
        src = inspect.getsource(app)
        self.assertIn('gr.api(ask_api, api_name="v1_ask")', src)
        self.assertIn('api_name="v1_ask"', src.split("def _api_fallback")[1])
        self.assertIn('api_name="v1_systemone"', src.split("def _api_fallback")[1])

    def test_the_ask_controls_carry_stable_ids(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            ids = {b.elem_id for b in demo.blocks.values() if getattr(b, "elem_id", None)}
        finally:
            demo.close()
        self.assertIn("ask-input", ids)
        self.assertIn("ask-run", ids)
        self.assertIn('id="ask-provenance"', self.ui.drafted_line(
            {"model": "m", "generated_tokens": 1, "model_ms": 1.0}))

    def test_the_drafter_is_warmed_beside_blink(self):
        need_gradio(self)
        app = reload_app()
        src = inspect.getsource(app)
        self.assertIn("author.warm()", src)
        self.assertLess(src.index("blink.warm()"), src.index("def "))


class TestQuestionForm(unittest.TestCase):
    """The rows, the JSON and the wire format all have to say the same thing."""

    def setUp(self):
        import ui

        self.ui = ui

    def test_a_name_becomes_a_key(self):
        self.assertEqual(self.ui.slug("Pay invoice"), "pay_invoice")
        self.assertEqual(self.ui.slug("pay_invoice"), "pay_invoice")
        self.assertEqual(self.ui.slug("  Needs a REPLY!  "), "needs_a_reply")
        self.assertEqual(self.ui.slug(""), "")

    def test_every_preset_survives_the_round_trip(self):
        for label, _, text in examples.PLAYGROUND_PRESETS:
            wanted = json.loads(text)
            back, problems = self.ui.rows_to_questions(self.ui.questions_to_rows(wanted))
            self.assertEqual(problems, [], label)
            self.assertEqual(back, wanted, label)
            self.assertEqual(list(back), list(wanted), f"{label}: order")

    def test_the_first_render_matches_the_rows_it_shows(self):
        """prefer='saved' hashes the questions, so the form must produce them exactly."""
        wanted = json.loads(examples.PLAYGROUND_QUESTIONS)
        rows = self.ui.questions_to_rows(wanted)
        back, _ = self.ui.rows_to_questions(rows)
        self.assertEqual(blink.request_key(examples.PLAYGROUND_STATE, back),
                         blink.request_key(examples.PLAYGROUND_STATE, wanted))

    def test_choice_options_are_name_and_description(self):
        rows = [{"name": "q", "type": "choice", "ask": "Which?",
                 "options": "pay: Pay money\nBook a meeting"}]
        got, problems = self.ui.rows_to_questions(rows)
        self.assertEqual(problems, [])
        self.assertEqual(got["q"]["criteria"], {"pay": "Pay money", "book_a_meeting": "Book a meeting"})

    def test_colliding_names_keep_every_option(self):
        rows = [{"name": "q", "type": "choice", "ask": "Which?",
                 "options": "button: Apply\nbutton: Clear\nlink: Sort"}]
        got, problems = self.ui.rows_to_questions(rows)
        self.assertEqual(problems, [])
        self.assertEqual(len(got["q"]["criteria"]), 3)
        self.assertEqual(
            [d for _, d in blink.question_options(got["q"])],
            ["button: Apply", "button: Clear", "link: Sort"],
        )

    def test_yes_no_takes_optional_labels(self):
        bare, problems = self.ui.rows_to_questions(
            [{"name": "q", "type": "noul", "ask": "Now?", "options": ""}])
        self.assertEqual(problems, [])
        self.assertNotIn("criteria", bare["q"])
        both, _ = self.ui.rows_to_questions(
            [{"name": "q", "type": "noul", "ask": "Now?", "options": "true: Yes it is\nfalse: No"}])
        self.assertEqual(both["q"]["criteria"], {"true": "Yes it is", "false": "No"})

    def test_score_levels_are_a_list_in_order(self):
        rows = [{"name": "q", "type": "score", "ask": "How bad?",
                 "options": "None\nSome: a bit\nLots"}]
        got, problems = self.ui.rows_to_questions(rows)
        self.assertEqual(problems, [])
        self.assertEqual(got["q"]["criteria"], ["None", "Some: a bit", "Lots"])

    def test_blank_rows_are_ignored(self):
        rows = [dict(self.ui.BLANK_ROW),
                {"name": "q", "type": "noul", "ask": "Now?", "options": ""},
                dict(self.ui.BLANK_ROW)]
        got, problems = self.ui.rows_to_questions(rows)
        self.assertEqual(list(got), ["q"])
        self.assertEqual(problems, [])

    def test_a_choice_needs_two_options(self):
        got, problems = self.ui.rows_to_questions(
            [{"name": "q", "type": "choice", "ask": "Which?", "options": "only: one"}])
        self.assertEqual(got, {})
        self.assertIn("two options", problems[0])

    def test_too_many_options_are_refused(self):
        many = "\n".join(f"k{i}: option {i}" for i in range(blink.MAX_OPTIONS + 1))
        _, problems = self.ui.rows_to_questions(
            [{"name": "q", "type": "choice", "ask": "Which?", "options": many}])
        self.assertIn(str(blink.MAX_OPTIONS), problems[0])

    def test_score_levels_stay_between_two_and_ten(self):
        for options in ("only one", "\n".join(f"level {i}" for i in range(11))):
            _, problems = self.ui.rows_to_questions(
                [{"name": "q", "type": "score", "ask": "How bad?", "options": options}])
            self.assertIn("2 to 10", problems[0])

    def test_names_must_be_there_and_unique(self):
        _, problems = self.ui.rows_to_questions(
            [{"name": "", "type": "noul", "ask": "Now?", "options": ""}])
        self.assertEqual(problems, [self.ui.PROBLEM["name"]])
        _, problems = self.ui.rows_to_questions([
            {"name": "q", "type": "noul", "ask": "Now?", "options": ""},
            {"name": "q", "type": "noul", "ask": "Again?", "options": ""},
        ])
        self.assertIn("called q", problems[0])

    def test_an_empty_form_asks_for_a_question(self):
        got, problems = self.ui.rows_to_questions([dict(self.ui.BLANK_ROW)])
        self.assertEqual(got, {})
        self.assertEqual(problems, [self.ui.PROBLEM["empty"]])

    def test_bad_json_is_reported_rather_than_raised(self):
        for bad in ("{", "", "nope"):
            parsed, problems = self.ui.questions_from_json(bad)
            self.assertEqual(parsed, {})
            self.assertEqual(problems, [self.ui.PROBLEM["json"]])
        parsed, problems = self.ui.questions_from_json("[1, 2]")
        self.assertEqual(problems, [self.ui.PROBLEM["shape"]])

    def test_json_and_rows_agree_both_ways(self):
        wanted = json.loads(examples.PLAYGROUND_QUESTIONS)
        text = self.ui.questions_json(wanted)
        parsed, problems = self.ui.questions_from_json(text)
        self.assertEqual(problems, [])
        back, problems = self.ui.rows_to_questions(self.ui.questions_to_rows(parsed))
        self.assertEqual(problems, [])
        self.assertEqual(back, wanted)

    def test_the_form_holds_a_bounded_number_of_questions(self):
        self.assertGreaterEqual(self.ui.MAX_FORM_QUESTIONS, 4)
        self.assertLessEqual(self.ui.MAX_FORM_QUESTIONS, blink.MAX_QUESTIONS)
        crowd = {f"q{i}": {"type": "noul", "instructions": ""} for i in range(20)}
        self.assertEqual(len(self.ui.questions_to_rows(crowd)), self.ui.MAX_FORM_QUESTIONS)

    def test_problems_render_as_a_list(self):
        self.assertEqual(self.ui.problems_html([]), "")
        html = self.ui.problems_html(["one", "two"])
        self.assertIn("blk-problems", html)
        self.assertEqual(html.count("<li>"), 2)

    def test_the_form_is_on_screen_without_opening_anything(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            rows = [b for b in demo.blocks.values()
                    if "blk-qrow" in (getattr(b, "elem_classes", None) or [])]
            shown = [r for r in rows if r.visible]
            code = next(b for b in demo.blocks.values() if isinstance(b, gr.Code))
            acc = next(b for b in demo.blocks.values() if isinstance(b, gr.Accordion))
        finally:
            demo.close()
        self.assertEqual(len(rows), self.ui.MAX_FORM_QUESTIONS)
        self.assertEqual(len(shown), len(json.loads(examples.PLAYGROUND_QUESTIONS)))
        self.assertFalse(acc.open)  # only the JSON view is folded away
        self.assertNotEqual(code.interactive, False)


class TestDeepLinks(unittest.TestCase):
    """A link names a tab; an unknown one still opens the page."""

    def setUp(self):
        import ui

        self.ui = ui

    def test_the_slugs_cover_every_tab_and_case(self):
        # Computer use follows Home wherever a model that reads screens is served or there are runs to show
        cua = ("computer-use",) if self.ui.CUA_SERVED else ()
        self.assertEqual(
            self.ui.TAB_IDS,
            ("home", *cua, "use-cases", "ask", "playground", "results", "how-it-works", "api"),
        )
        self.assertEqual(self.ui.DEFAULT_TAB, "home")
        self.assertEqual(self.ui.HOME_TAB, "home")
        self.assertEqual(self.ui.ASK_TAB, "ask")
        self.assertEqual(self.ui.CASE_TAB, "use-cases")
        self.assertEqual(self.ui.API_TAB, "api")
        import screens

        # the text next click leads the use cases, then the rest in their own order; Screen click
        # lives on the Computer use tab
        rest = tuple(c.key for c in examples.USE_CASES if c.key != "nextclick")
        self.assertEqual(self.ui.CASE_IDS, ("nextclick",) + rest)
        self.assertNotIn(screens.KEY, self.ui.CASE_IDS)
        self.assertEqual(self.ui.DEFAULT_CASE, self.ui.CASE_IDS[0])

    def test_query_and_hash_both_name_a_tab(self):
        for raw, want in (
            ("?tab=results", "results"),
            ("#results", "results"),
            ("?tab=use-cases", "use-cases"),
            ("#how-it-works", "how-it-works"),
            ("?tab=playground", "playground"),
            ("?tab=home", "home"),
            ("?tab=ask", "ask"),
            ("#ask", "ask"),
        ):
            self.assertEqual(self.ui.parse_deep_link(raw), (want, None), raw)

    def test_the_tab_name_is_read_case_insensitively(self):
        for raw in ("?tab=RESULTS", "?tab=Results", "?tab=  results  ", "#Results"):
            self.assertEqual(self.ui.parse_deep_link(raw)[0], "results", raw)

    def test_a_query_beats_a_hash(self):
        self.assertEqual(self.ui.parse_deep_link("?tab=results#how-it-works")[0], "results")
        self.assertEqual(self.ui.parse_deep_link("?tab=nonsense#results")[0], "results")

    def test_anything_unknown_opens_the_first_tab(self):
        for raw in ("", "?", "#", "?tab=", "?tab=nonsense", "#nope", "?other=1", None, 7, []):
            self.assertEqual(self.ui.parse_deep_link(raw), ("home", None), repr(raw))

    def test_a_case_opens_its_use_case(self):
        for key in self.ui.CASE_IDS:
            self.assertEqual(
                self.ui.parse_deep_link(f"?tab=use-cases&case={key}"), ("use-cases", key)
            )
            self.assertEqual(self.ui.parse_deep_link(f"?case={key}"), ("use-cases", key))
        self.assertEqual(self.ui.parse_deep_link("?case=NEXTCLICK"), ("use-cases", "nextclick"))

    def test_a_case_that_is_not_ours_is_dropped(self):
        self.assertEqual(self.ui.parse_deep_link("?tab=use-cases&case=nope"), ("use-cases", None))
        self.assertEqual(self.ui.parse_deep_link("?case=nope"), ("home", None))

    def test_a_case_only_counts_on_the_use_cases_tab(self):
        self.assertEqual(self.ui.parse_deep_link("?tab=results&case=rag"), ("results", None))

    def test_other_parameters_are_none_of_its_business(self):
        self.assertEqual(
            self.ui.parse_deep_link("?__theme=light&tab=results&x=1"), ("results", None)
        )

    def test_every_tab_carries_its_slug_as_an_id(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            ids = [b.id for b in demo.blocks.values() if isinstance(b, gr.Tab)]
        finally:
            demo.close()
        for slug in self.ui.TAB_IDS:
            self.assertIn(slug, ids)
        for key in self.ui.CASE_IDS:
            self.assertIn(key, ids)
        self.assertEqual(len(ids), len(set(ids)), "tab ids must be unique")

    def test_the_load_handler_maps_a_link_to_tab_updates(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        tab, case, ask = app.open_from_url("?tab=use-cases&case=nextclick")
        self.assertEqual(tab["selected"], "use-cases")
        self.assertEqual(case["selected"], "nextclick")
        self.assertEqual(ask["value"], "")
        tab, case, _ = app.open_from_url("?tab=results")
        self.assertEqual(tab["selected"], "results")
        self.assertNotIn("selected", case)
        tab, _, ask = app.open_from_url("?ask=how%20many%20r")
        self.assertEqual(tab["selected"], "ask")
        self.assertEqual(ask["value"], "how many r")

    def test_the_address_is_replaced_not_pushed_and_told_to_the_parent(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        self.assertIn("history.replaceState", app.SYNC_URL)
        self.assertNotIn("pushState", app.SYNC_URL)
        self.assertIn("window.parent.postMessage", app.SYNC_URL)
        self.assertIn("'https://huggingface.co'", app.SYNC_URL)
        self.assertIn("new URL(window.location.href)", app.SYNC_URL)  # keeps other params
        self.assertIn("searchParams.delete('case')", app.SYNC_URL)
        self.assertIn("catch", app.SYNC_URL)  # a parent that ignores it must not break us
        # the ids arrive as arguments; nothing here reads the rendered tab strip
        self.assertIn("(tab, sub, shot) =>", app.SYNC_URL)
        self.assertIn("searchParams.delete('shot')", app.SYNC_URL)  # a Screen click preset only while in view
        for missing in ("role=\"tab\"", "aria-selected", "textContent"):
            self.assertNotIn(missing, app.SYNC_URL)

    def test_the_load_script_only_reads_the_address(self):
        """Opening a tab is python's job: a phone hides the strip behind a menu."""
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        self.assertIn("window.location.search + window.location.hash", app.READ_URL)
        for missing in ("role=\"tab\"", "aria-selected", "textContent", "click()"):
            self.assertNotIn(missing, app.READ_URL)

    def test_a_selection_is_read_from_the_event_not_the_page(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()

        class Pick:
            def __init__(self, value=None, index=None):
                self.value, self.index = value, index

        for slug, label in self.ui.TABS:
            self.assertEqual(app.picked_tab(Pick(value=label)), slug)
        # the overflow menu can hand back an index instead of a label
        for i, slug in enumerate(self.ui.TAB_IDS):
            self.assertEqual(app.picked_tab(Pick(index=i)), slug)
        self.assertEqual(app.picked_tab(Pick(value="nonsense")), self.ui.DEFAULT_TAB)
        self.assertEqual(app.picked_tab(Pick()), self.ui.DEFAULT_TAB)
        for case in examples.USE_CASES:
            self.assertEqual(app.picked_case(Pick(value=case.title)), case.key)
        self.assertEqual(app.picked_case(Pick(index=2)), self.ui.CASE_IDS[2])


class TestHomeAndAskTabs(unittest.TestCase):
    """Home is a static landing page; Ask has its own tab, and the draft can move next door."""

    def setUp(self):
        import ui

        self.ui = ui

    def test_home_leads_with_figures_then_links(self):
        joined = "".join(self.ui.home_blocks())
        self.assertIn(self.ui.HOME["lede"], joined)
        self.assertIn("<svg", joined)
        for slug, title, line in self.ui.HOME_LINKS:
            self.assertIn(slug, self.ui.TAB_IDS)
            face = self.ui.card_face(title, line)
            self.assertIn(title, face)
            self.assertIn(line, face)
        self.assertNotIn(self.ui.HOME_TAB, [s for s, _, _ in self.ui.HOME_LINKS])

    def test_every_outside_link_opens_in_a_new_tab(self):
        joined = self.ui.home_external()
        for href, title, _ in self.ui.HOME_EXTERNAL:
            self.assertIn(f'href="{href}"', joined)
            self.assertIn(title, joined)
            self.assertTrue(href.startswith("https://"), href)
        self.assertEqual(
            joined.count('target="_blank" rel="noopener noreferrer"'),
            len(self.ui.HOME_EXTERNAL),
        )
        for name in ("blink-4b", "blink-27b", "blink-mimo-9b"):
            self.assertIn(f"huggingface.co/thegovind/{name}", joined)
        self.assertIn("spaces/multimodalart/jev-decision-index", joined)
        self.assertIn("jevbench/issues/81", joined)

    def test_home_numbers_come_from_the_board(self):
        import results

        data = results.load()
        rows = results.headline_rows(data)
        self.assertEqual([r["name"] for r in rows][:2], ["blink-27b", "Jev 1.13.0"])
        for row in rows:
            match = next(s for s in data["decision_index"]["systems"]
                         if s["name"] == row["name"])
            self.assertEqual(row["index"], match["index"])
        self.assertTrue(any(r["kind"] == "ours" for r in rows))

    def test_home_never_calls_the_model(self):
        calls = []
        real = blink.decide
        blink.decide = lambda *a, **k: calls.append(a) or real(*a, **k)
        try:
            self.ui.home_blocks()
        finally:
            blink.decide = real
        self.assertEqual(calls, [])

    def test_every_in_app_card_is_a_button_not_a_link(self):
        """A link reloads the page; a phone hides the tab strip. A button does neither."""
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            hits = [b for b in demo.blocks.values()
                    if "blk-tile-hit" in (getattr(b, "elem_classes", None) or [])]
            wraps = [b for b in demo.blocks.values()
                     if "blk-tilewrap" in (getattr(b, "elem_classes", None) or [])]
            self.assertEqual(len(wraps), len(self.ui.HOME_LINKS))
            self.assertEqual(len(hits), len(self.ui.HOME_LINKS))
            self.assertTrue(all(isinstance(b, gr.Button) for b in hits))
            self.assertEqual([b.value for b in hits],
                             [t for _, t, _ in self.ui.HOME_LINKS])
            page = "".join(self.ui.home_blocks()) + self.ui.home_external()
            for slug, _, _ in self.ui.HOME_LINKS:
                self.assertNotIn(f'href="?tab={slug}"', page)
        finally:
            demo.close()

    def test_ask_lives_in_its_own_tab_with_the_ids_the_checks_use(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            ids = {b.elem_id for b in demo.blocks.values() if getattr(b, "elem_id", None)}
            self.assertLessEqual({"ask-input", "ask-run"}, ids)
            self.assertIn('id="ask-provenance"', self.ui.drafted_line(
                {"model": "m", "generated_tokens": 3, "model_ms": 5.0}))
            tabs = [b.id for b in demo.blocks.values() if isinstance(b, gr.Tab)]
            self.assertIn("ask", tabs)
            self.assertIn("home", tabs)
            ask = next(b for b in demo.blocks.values()
                       if getattr(b, "elem_id", None) == "ask-input")
            play = next(b for b in demo.blocks.values()
                        if "blk-state" in (getattr(b, "elem_classes", None) or []))
            at = {b.id: b._id for b in demo.blocks.values() if isinstance(b, gr.Tab)}
            # the use cases are the first working tab, then Ask, then the playground
            self.assertLess(at["use-cases"], at["ask"])
            self.assertLess(at["ask"], ask._id)
            self.assertLess(ask._id, at["playground"])
            self.assertLess(at["playground"], play._id)
        finally:
            demo.close()

    def test_no_ask_box_in_the_playground(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            ask = next(b for b in demo.blocks.values()
                       if getattr(b, "elem_id", None) == "ask-input")
            at = {b.id: b._id for b in demo.blocks.values() if isinstance(b, gr.Tab)}
            decide = next(b for b in demo.blocks.values()
                          if "blk-decide" in (getattr(b, "elem_classes", None) or []))
            self.assertEqual(decide.value, "Decide")
            # the playground runs from its tab to the next one; the ask box is not in it
            self.assertLess(at["playground"], decide._id)
            self.assertLess(decide._id, at["results"])
            self.assertFalse(at["playground"] < ask._id < at["results"])
        finally:
            demo.close()

    def test_the_ask_tab_disappears_when_drafting_is_off(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        with unittest.mock.patch.object(app.author, "enabled", lambda: False):
            demo = app.build()
            try:
                tabs = [b.id for b in demo.blocks.values() if isinstance(b, gr.Tab)]
                self.assertNotIn("ask", tabs)
                self.assertIn("home", tabs)
                self.assertIn("playground", tabs)
                ids = {b.elem_id for b in demo.blocks.values()
                       if getattr(b, "elem_id", None)}
                self.assertNotIn("ask-input", ids)
            finally:
                demo.close()

    def test_a_draft_moves_into_the_playground(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            held = {"state": "strawberry",
                    "answers": "<div class=\"blk-answers\">counted</div>",
                    "questions": {"count": {"type": "choice",
                                            "instructions": "How many?",
                                            "criteria": {"two": "2", "three": "3"}}}}
            req, shown, state, text, trouble, answers, *slots = app.into_playground(
                "1", {}, held)
        finally:
            demo.close()
        self.assertEqual(state, "strawberry")
        self.assertIn("count", text)
        self.assertEqual(shown, 1)
        self.assertEqual(req["questions"]["count"]["type"], "choice")
        self.assertEqual(req["problems"], [])
        # the answer that came with the draft, never the one the tab happened to show
        self.assertIn("counted", answers)
        self.assertEqual(slots[app.SLOTS]["value"], "count")
        self.assertTrue(slots[0]["visible"])
        self.assertFalse(slots[1]["visible"])

    def test_a_draft_with_no_answer_clears_the_column(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            out = app.into_playground(
                "1", {},
                {"state": "x", "questions": {"a": {"type": "noul", "instructions": "?"}}})
        finally:
            demo.close()
        self.assertIn(__import__('html').escape(self.ui.STALE_LABEL), out[5])
        self.assertNotIn("blk-answers", out[5])

    def test_the_hand_off_survives_an_empty_draft(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            out = app.into_playground("1", {}, None)
        finally:
            demo.close()
        self.assertTrue(all(isinstance(x, dict) for x in out))

    def test_the_draft_summary_shows_what_was_asked(self):
        q = {"count": {"type": "choice", "instructions": "How many r?",
                       "criteria": {"two": "2", "three": "3"}}}
        html = self.ui.draft_summary(q)
        for want in ("count", "choice", "How many r?", "2", "3"):
            self.assertIn(want, html)
        self.assertEqual(self.ui.draft_summary({}), "")

    def test_the_compact_row_still_shows_what_is_asked(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            head = next(b for b in demo.blocks.values()
                        if "blk-qhead" in (getattr(b, "elem_classes", None) or []))
            def leaves(node):
                kids = getattr(node, "children", None)
                if not kids:
                    return [node]
                return [leaf for kid in kids for leaf in leaves(kid)]

            labels = [b.label for b in leaves(head)]
            # the question reads first; its name and type sit under it
            self.assertEqual(
                labels, [self.ui.FORM["ask"], self.ui.FORM["name"], self.ui.FORM["type"]])
            fold = next(b for b in demo.blocks.values()
                        if "blk-qmore" in (getattr(b, "elem_classes", None) or []))
            self.assertFalse(fold.open)
            self.assertEqual(fold.label, self.ui.FORM["more"])
        finally:
            demo.close()


class TestHeldout(unittest.TestCase):
    """The numbers nothing was trained on or chosen with; the figure has to say so."""

    def setUp(self):
        import results
        import ui

        self.results = results
        self.ui = ui
        self.d = results.load()
        self.h = self.d["heldout"]

    def test_the_section_is_built_from_the_file(self):
        self.assertTrue(self.h["label"])
        self.assertTrue(self.h["caption"])
        self.assertEqual(len(self.h["tasks"]), 8)
        names = [s["name"] for s in self.results.heldout_systems(self.d)]
        self.assertEqual(names, ["blink-27b", "blink-mimo-9b", "blink-4b"])
        svg = self.results.heldout_chart(self.d)
        for s in self.h["systems"]:
            self.assertIn(s["name"], svg)
            self.assertIn(f'{s["accuracy"] * 100:.1f}', svg)
        self.assertIn("78.3", svg)
        self.assertIn("73.5", svg)
        self.assertIn("68.5", svg)

    def test_every_bar_carries_a_visible_interval(self):
        svg = self.results.heldout_chart(self.d)
        # drawn twice: an outline that reads on the page, a core that reads on the bar
        for cls in ("blk-whisk", "blk-whisk-edge"):
            self.assertEqual(svg.count(f'class="{cls}"'), 3 * len(self.h["systems"]))
        css = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css"),
                   encoding="utf-8").read()
        edge = css[css.index(".blk-whisk-edge {"):]
        edge = edge[: edge.index("}")]
        self.assertIn("var(--ink)", edge)
        core = css[css.index("\n.blk-whisk {"):]
        core = core[: core.index("}")]
        self.assertNotIn("var(--bg)", core)  # the page colour hides it on the page
        self.assertGreater(float(edge.split("stroke-width:")[1].split(";")[0].strip()),
                           float(core.split("stroke-width:")[1].split(";")[0].strip()))
        for s in self.h["systems"]:
            lo, hi = s["ci95"]
            self.assertIn(f'{lo * 100:.1f}, {hi * 100:.1f}', svg)
            self.assertIn(f'ECE {s["ece"]:.3f}', svg)

    def test_the_bars_start_at_zero(self):
        """A truncated axis would make a ten point gap look like a rout."""
        svg = self.results.heldout_chart(self.d)
        self.assertIn(">0<", svg)
        self.assertIn(">100<", svg)

    def test_every_task_is_shown_for_every_model(self):
        svg = self.results.heldout_tasks(self.d)
        for task in self.h["tasks"]:
            self.assertIn(task["label"], svg)
            self.assertIn(task["id"], svg)
        self.assertEqual(svg.count('class="blk-grow"'),
                         len(self.h["tasks"]) * len(self.h["systems"]))
        worst = self.h["systems"][2]["tasks"]["svamp_numeric"]["accuracy"]
        self.assertIn(f"{worst * 100:.0f}", svg)

    def test_a_task_the_file_does_not_score_is_skipped(self):
        data = copy.deepcopy(self.d)
        data["heldout"]["systems"][0]["tasks"].pop("toxic_chat")
        svg = self.results.heldout_tasks(data)
        self.assertEqual(svg.count('class="blk-grow"'),
                         len(self.h["tasks"]) * len(self.h["systems"]) - 1)

    def test_the_note_repeats_the_caption_and_when_it_was_frozen(self):
        note = self.results.heldout_note(self.d)
        self.assertIn(self.h["caption"], note)
        self.assertIn(self.h["frozen_utc"][:10], note)
        detail = self.ui.heldout_detail(self.h)
        self.assertIn(str(self.h["items_per_task"]), detail)
        self.assertIn(str(len(self.h["tasks"])), detail)

    def test_home_puts_it_beside_the_index(self):
        blocks = self.ui.home_blocks(self.d)
        duo = next(b for b in blocks if "blk-pair" in b)
        self.assertEqual(duo.count('class="blk-half"'), 2)
        self.assertIn(self.ui.HOME["index"], duo)
        self.assertIn(self.h["label"], duo)
        self.assertLess(duo.index(self.ui.HOME["index"]), duo.index(self.h["label"]))
        # figure first: each pane opens with its chart, and the note comes after
        self.assertLess(duo.index("<svg"), duo.index("blk-note"))

    def test_results_shows_the_overall_and_then_the_tasks(self):
        joined = "".join(self.ui.results_blocks(self.d))
        self.assertIn(self.h["label"], joined)
        self.assertIn(f'{self.h["label"]} by task', joined)
        self.assertLess(joined.index("blk-narrowfig"), joined.index("by task"))
        self.assertIn("About these tasks", joined)
        self.assertIn(self.h["caption"], joined)

    def test_it_all_disappears_when_the_file_has_none(self):
        data = copy.deepcopy(self.d)
        data.pop("heldout")
        self.assertEqual(self.results.heldout_chart(data), "")
        self.assertEqual(self.results.heldout_tasks(data), "")
        self.assertEqual(self.results.heldout_note(data), "")
        self.assertEqual(self.results.heldout_systems(data), [])
        self.assertNotIn("blk-pair", "".join(self.ui.home_blocks(data)))
        self.assertNotIn("Held-out", "".join(self.ui.results_blocks(data)))
        for empty in ({}, {"heldout": None}, {"heldout": {}}):
            self.assertEqual(self.results.heldout_chart(empty), "")
            self.assertEqual(self.results.heldout_tasks(empty), "")

    def test_no_numbers_beyond_the_file(self):
        """Every figure in the two held-out charts has to come from the json."""
        allowed = {"0", "100"}
        for s in self.h["systems"]:
            allowed |= {f'{s["accuracy"] * 100:.1f}', f'{s["ece"]:.3f}'}
            allowed |= {f'{v * 100:.1f}' for v in s["ci95"]}
            for cell in s["tasks"].values():
                allowed |= {f'{cell["accuracy"] * 100:.0f}',
                            f'{cell["accuracy"] * 100:.1f}', f'{cell["ece"]:.3f}'}
        text = re.sub(r"<[^>]+>", " ", self.results.heldout_chart(self.d)
                      + self.results.heldout_tasks(self.d))
        for s in self.h["systems"]:  # a name is not a measurement
            text = text.replace(s["name"], " ")
        shown = {t for t in re.findall(r"\d+(?:\.\d+)?", text)}
        axis = {str(t) for t in range(0, 101, 20)} | {"25", "50", "75"}
        self.assertEqual(shown - allowed - axis, set())

    def test_the_numbers_are_readable_without_a_pointer(self):
        table = self.results.heldout_table(self.d)
        self.assertIn("<caption>", table)
        self.assertIn("each task counts equally", table)
        self.assertIn("95% intervals", table)
        self.assertIn("95%", table)
        for s in self.h["systems"]:
            self.assertIn(s["name"], table)
            self.assertIn(f'{s["accuracy"] * 100:.1f}%', table)
            self.assertIn(f'{s["ci95"][0] * 100:.1f}', table)
            self.assertIn(f'{s["ci95"][1] * 100:.1f}', table)
            self.assertIn(f'{s["ece"]:.3f}', table)
        self.assertEqual(table.count('scope="row"'), len(self.h["systems"]))
        self.assertIn(table, "".join(self.ui.results_blocks(self.d)))

    def test_a_phone_gets_the_table_instead_of_the_wide_figure(self):
        joined = "".join(self.ui.home_blocks(self.d))
        self.assertEqual(joined.count('class="blk-wide"'), 2)
        self.assertEqual(joined.count('class="blk-narrow"'), 2)
        self.assertIn(self.results.heldout_table(self.d), joined)
        self.assertIn(self.results.headline_table(self.d), joined)
        css = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css"),
                   encoding="utf-8").read()
        phone = css[css.index("@media (max-width: 720px) {"):]
        self.assertIn(".blk-wide", phone)
        self.assertIn(".blk-narrow", phone)

    def test_the_section_names_no_hardware(self):
        text = (self.results.heldout_chart(self.d) + self.results.heldout_tasks(self.d)
                + self.results.heldout_note(self.d) + self.ui.heldout_detail(self.h))
        for word in ("gpu", "accelerator", "accelerator", "zerogpu", "cuda", "cpu"):
            self.assertNotIn(word, text.lower())


class TestRequestIntegrity(unittest.TestCase):
    """Whatever the answer column is about, the request on screen has to be it."""

    def setUp(self):
        import ui

        self.ui = ui

    def nine(self) -> str:
        return json.dumps({f"q{i}": {"type": "noul", "instructions": f"ask {i}?"}
                           for i in range(9)})

    def test_more_questions_than_rows_are_kept_not_dropped(self):
        held = self.ui.request_from_json(self.nine())
        self.assertEqual(len(held["questions"]), 9)
        self.assertEqual(held["problems"], [])
        self.assertIn("8", held["locked"])
        self.assertIn("9", held["locked"])
        # the rows are a view of the first eight, and they say so
        self.assertEqual(len(self.ui.questions_to_rows(held["questions"])), 8)
        self.assertIn(held["locked"], self.ui.locked_html(held["locked"]))

    def test_a_newline_in_a_description_stays_one_option(self):
        text = json.dumps({"a": {"type": "choice", "instructions": "?", "criteria": {
            "keep": "one line\nand another", "drop": "plain"}}})
        held = self.ui.request_from_json(text)
        self.assertEqual(held["problems"], [])
        self.assertEqual(len(held["questions"]["a"]["criteria"]), 2)
        self.assertEqual(held["questions"]["a"]["criteria"]["keep"], "one line\nand another")
        self.assertTrue(held["locked"])  # the rows would split it, so they cannot edit it

    def test_an_unsupported_type_is_refused_not_rewritten(self):
        text = json.dumps({"a": {"type": "ranking", "criteria": {"x": "1", "y": "2"}}})
        held = self.ui.request_from_json(text)
        self.assertEqual(held["questions"], {})
        self.assertIn("a", held["problems"][0])
        for wire in ("choice", "noul", "score"):
            self.assertNotIn(wire, json.dumps(held["questions"]))

    def test_malformed_json_blocks_the_run(self):
        for text in ("{", "[]", "null", '{"a": 3}', ""):
            held = self.ui.request_from_json(text)
            self.assertTrue(held["problems"], text)
            self.assertEqual(held["questions"], {}, text)

    def test_every_validation_error_empties_the_request(self):
        bad = {
            '{"a": {"type": "choice", "criteria": {"only": "one"}}}': "two options",
            '{"a": {"type": "score", "criteria": ["one"]}}': "2 to 10",
            '{"a": {"type": "choice", "criteria": "text"}}': "options",
            '{"a": 3}': "object",
            "{}": "Add a question",
        }
        for text, hint in bad.items():
            held = self.ui.request_from_json(text)
            self.assertEqual(held["questions"], {}, text)
            self.assertTrue(any(hint in p for p in held["problems"]),
                            f"{text} -> {held['problems']}")

    def test_a_request_the_rows_can_hold_is_not_locked(self):
        for text in (examples.PLAYGROUND_QUESTIONS,
                     '{"a": {"type": "noul"}}',
                     '{"a": {"type": "score", "instructions": "?", '
                     '"criteria": ["low", "high"]}}'):
            held = self.ui.request_from_json(text)
            self.assertEqual(held["problems"], [], text)
            self.assertEqual(held["locked"], "", text)

    def test_the_rows_and_the_json_agree_on_what_runs(self):
        held = self.ui.request_from_json(examples.PLAYGROUND_QUESTIONS)
        rows = self.ui.questions_to_rows(held["questions"])
        self.assertEqual(self.ui.request_from_rows(rows)["questions"], held["questions"])

    def test_decide_refuses_a_request_that_did_not_validate(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            src = inspect.getsource(app.playground_tab)
        finally:
            demo.close()
        # Decide validates the values it was handed, never a state a sync may rewrite
        self.assertIn("def current(touched, text, n, flat)", src)
        self.assertIn("held = current(touched, text, n, flat)", src)
        self.assertIn("run_in = [questions, shown, *fields]", src)
        self.assertNotIn("ui.rows_to_questions(_rows_from(flat))", src)
        decide = src.split("def decide(")[1].split("def ")[0]
        self.assertNotIn("req", decide)  # the stored request has no say here

    def test_two_names_that_normalise_alike_are_refused(self):
        """`q` and ` q ` are one name; the second must not quietly replace the first."""
        held = self.ui.request_from_json(
            json.dumps({"q": {"type": "noul", "instructions": "a?"},
                        " q ": {"type": "noul", "instructions": "b?"}}))
        self.assertEqual(held["questions"], {})
        self.assertEqual(held["problems"], [self.ui.PROBLEM["dupe"].format(name="q")])

    def test_a_padded_name_is_kept_as_written(self):
        held = self.ui.request_from_json(
            json.dumps({" q ": {"type": "noul", "instructions": "a?"}}))
        self.assertEqual(list(held["questions"]), [" q "])
        self.assertEqual(held["problems"], [])
        self.assertTrue(held["locked"])  # the rows would strip it, so they may not edit

    def test_a_request_from_anywhere_enters_through_one_gate(self):
        draft = {"a": {"type": "choice", "instructions": "?",
                       "criteria": {"keep": "one line\nand another", "drop": "plain"}}}
        held = self.ui.request_from_questions(draft)
        self.assertEqual(held["questions"], draft)
        self.assertEqual(len(held["questions"]["a"]["criteria"]), 2)
        self.assertTrue(held["locked"])
        self.assertEqual(held, self.ui.request_from_json(json.dumps(draft)))

    def test_the_hand_off_keeps_the_request_it_was_given(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            draft = {"state": "x", "questions": {"a": {
                "type": "choice", "instructions": "?",
                "criteria": {"keep": "one line\nand another", "drop": "plain"}}}}
            req, shown, state, text, trouble, _answers, *slots = app.into_playground(
                "1", {}, draft)
        finally:
            demo.close()
        self.assertEqual(req["questions"], draft["questions"])
        self.assertEqual(len(req["questions"]["a"]["criteria"]), 2)
        self.assertEqual(json.loads(text), draft["questions"])
        self.assertTrue(req["locked"])
        self.assertEqual(trouble, self.ui.locked_html(req["locked"]))
        self.assertFalse(slots[app.SLOTS]["interactive"])  # a view, not a rebuild

    def test_the_blank_starter_is_editable(self):
        held = self.ui.request_from_questions(self.ui.BLANK_QUESTIONS)
        self.assertEqual(held["problems"], [])
        self.assertEqual(held["locked"], "")
        rows = self.ui.questions_to_rows(held["questions"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.ui.request_from_rows(rows)["questions"], held["questions"])
        for value in self.ui.BLANK_QUESTIONS["label"]["criteria"].values():
            self.assertTrue(value.strip())  # an empty description cannot round-trip

    def test_decide_reads_the_editor_and_the_rows_it_was_handed(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            src = inspect.getsource(app.playground_tab)
            body = src.split("def current(")[1].split("\n    def ")[0]
        finally:
            demo.close()
        # the surface whose content changed last is the one that runs
        self.assertIn('if touched == "rows" and not held["locked"]', body)
        # otherwise bad json blocks, and so does a locked or freshly pasted request
        self.assertIn('held["problems"] or held["locked"] or touched == "json"', body)
        # and a row edit that has not synced yet still wins over a stale editor
        self.assertIn('edited["questions"] != held["questions"]', body)

    def test_a_slow_sync_cannot_be_outrun(self):
        """The bug: paste bad JSON, click Decide before validation lands, get answers."""
        import ui

        stale_rows = [dict(ui.BLANK_ROW, name="intent", ask="?",
                           options="a: one\nb: two")]
        # whatever the rows hold, the editor is what the visitor pasted
        held = ui.request_from_json("{oops")
        self.assertTrue(held["problems"])
        self.assertEqual(held["questions"], {})
        self.assertEqual(ui.request_from_rows(stale_rows)["problems"], [])

    def test_the_last_touched_surface_is_read_on_the_client(self):
        """A disagreement needs a tiebreak that no pending round trip can rewrite."""
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        self.assertIn("window.__blinkLast", app.READ_URL)  # only initialised there
        self.assertIn("window.__blinkLast", app.READ_TOUCH)
        self.assertIn("window.__blinkRev", app.READ_TOUCH)
        # a mark is made by a listener that only fires on a changed value. Nothing
        # watches the document, so a click, a focus or opening the editor marks nothing
        for missing in ("addEventListener", "closest", "keyup", "click", ".blk-acc",
                        ".blk-qrow"):
            self.assertNotIn(missing, app.READ_URL)
        self.assertIn('"rows"', app.STAMP_ROWS)
        self.assertIn('"json"', app.STAMP_JSON)
        for stamp in (app.STAMP_ROWS, app.STAMP_JSON):
            self.assertIn("window.__blinkRev = (window.__blinkRev || 0) + 1", stamp)
            self.assertIn("a[0] = String(window.__blinkRev)", stamp)
        src = inspect.getsource(app.playground_tab)
        self.assertIn("js=READ_TOUCH", src)
        self.assertIn("js=STAMP_ROWS", src)
        self.assertIn("js=STAMP_JSON", src)
        # it is read at the head of the chain, before anything is validated or run
        chain = src.split("def live(trigger):")[1].split("def ")[0]
        self.assertLess(chain.index("READ_TOUCH"), chain.index("waiting"))
        self.assertLess(chain.index("waiting"), chain.index("decide"))

    def test_a_revision_is_stamped_on_every_change_and_every_load(self):
        """Opening the editor is not an edit; a preset and a hand-off are loads."""
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        src = inspect.getsource(app.playground_tab)
        for line in ("field.input(from_rows", "questions.input(from_json",
                     "add_btn.click(add_row"):
            stamped = src.split(line)[1].split(")\n")[0]
            self.assertIn("js=STAMP_", stamped, line)
        # presets, the blank starter and the row remover all load or change a request
        for line in ("live(btn.click(", "live(blank_btn.click(", "drop.click("):
            self.assertIn("js=STAMP_", src.split(line)[1].split("\n\n")[0], line)
        self.assertIn("js=STAMP_JSON", inspect.getsource(app.build))  # the hand-off

    def test_an_overtaken_sync_never_writes_its_result_back(self):
        """M2: validation held up on the network, landing after a newer preset."""
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            wired = handlers(demo)
            from_json, from_rows = wired["from_json"], wired["from_rows"]
            seen = {}
            fresh = from_json("2", seen, self.ui.questions_json(
                {"kept": {"type": "noul", "instructions": "?"}}), 1)
            self.assertEqual(fresh[0]["questions"], {"kept": {
                "type": "noul", "instructions": "?"}})
            self.assertEqual(fresh[1], 1)  # one row, the request it just validated
            # the older paste finally lands: every output is left exactly as it was
            late = from_json("1", seen, '{"obsolete": {"type": "noul"}}', 1)
            self.assertEqual(len(late), len(fresh))
            self.assertTrue(all(x == gr_skip() for x in late), late[:3])
            # and so is an older row sync, including the error it would have raised
            rows = [dict(self.ui.BLANK_ROW, name="a", ask="?", options="only: one")]
            late_rows = from_rows("1", seen, 1, *app._flatten(rows))
            self.assertEqual(list(late_rows), [gr_skip()] * 3)
            # the next edit is newer again, and is taken
            good = [dict(self.ui.BLANK_ROW, name="a", ask="?",
                         options="one: 1\ntwo: 2")]
            taken = from_rows("3", seen, 1, *app._flatten(good))
            self.assertEqual(list(taken[0]["questions"]), ["a"])
            self.assertEqual(taken[2], "")
        finally:
            demo.close()

    def test_an_overtaken_load_leaves_the_rows_it_found(self):
        """A stale preset, hand-off or Add question must not reshape a newer request."""
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            wired = handlers(demo)
            seen = {"request": 5}
            rows = [dict(self.ui.BLANK_ROW, name="a", ask="?", options="one: 1\ntwo: 2")]
            late = wired["add_row"]("4", seen, {}, 1, *app._flatten(rows))
            self.assertEqual(list(late), [gr_skip()] * (app.SLOTS * 5 + 1))
            self.assertEqual(seen["request"], 5)  # nothing older moves the mark
            added = wired["add_row"]("6", seen, {}, 1, *app._flatten(rows))
            self.assertEqual(added[0], 2)
            # the hand-off enters the same way
            draft = {"state": "x", "model": "owner/one",
                     "questions": {"a": {"type": "noul", "instructions": "?"}}}
            stale = app.into_playground("5", seen, draft)
            self.assertEqual(list(stale), [gr_skip()] * (6 + app.SLOTS * 5))
        finally:
            demo.close()

    def test_an_overtaken_run_never_paints_over_a_newer_one(self):
        """The answer column belongs to the newest request that asked for it."""
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            wired = handlers(demo)
            decide, waiting = wired["decide"], wired["waiting"]
            text = self.ui.questions_json({"a": {"type": "noul", "instructions": "?"}})
            flat = app._flatten(self.ui.questions_to_rows(json.loads(text)))
            seen = {}
            self.assertIn("blk-answers", decide("x", None, "json", "4", seen, text, 1,
                                                *flat))
            self.assertEqual(waiting("json", "3", seen, text, 1, *flat), gr_skip())
            self.assertEqual(decide("x", None, "json", "3", seen, text, 1, *flat),
                             gr_skip())
        finally:
            demo.close()

    def test_a_revision_is_a_number_however_it_arrives(self):
        for raw, want in ((None, 0), ("", 0), ("7", 7), (7, 7), (" 7 ", 7), ("x", 0)):
            self.assertEqual(self.ui.as_rev(raw), want, raw)
        seen = {}
        self.assertTrue(self.ui.newest(seen, "request", "2"))
        self.assertEqual(self.ui.applied(seen, "request"), 2)
        self.assertFalse(self.ui.newest(seen, "request", "1"))
        self.assertEqual(self.ui.applied(seen, "request"), 2)  # the mark never goes back
        self.assertTrue(self.ui.newest(seen, "request", "2"))  # the same one may finish
        self.assertTrue(self.ui.newest(seen, "answers", "1"))  # a surface of its own
        self.assertTrue(self.ui.newest(None, "request", "1"))  # nothing to order against

    def test_the_latency_hook_is_off_unless_asked_for(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        self.assertEqual(app.SLOW_SYNC, 0)
        with unittest.mock.patch.dict(os.environ, {"BLINK_SLOW_SYNC": "1.5"}):
            self.assertEqual(reload_app().SLOW_SYNC, 1.5)

    def test_a_blocked_request_never_reaches_the_model(self):
        import ui

        calls = []
        real = blink.decide
        blink.decide = lambda *a, **k: calls.append(a) or real(*a, **k)
        try:
            held = ui.request_from_json("{oops")
            html = (ui.problems_html(held["problems"]) if held["problems"]
                    else ui.run_questions("x", held["questions"]))
        finally:
            blink.decide = real
        self.assertEqual(calls, [])
        self.assertIn(ui.PROBLEM["json"], html)


class TestAttributionAndRows(unittest.TestCase):
    """A result names the model that produced it, and a row appears one at a time."""

    def setUp(self):
        import ui

        self.ui = ui

    def test_the_answer_names_the_model_that_answered(self):
        q = {"urgent": {"type": "noul", "instructions": "Now?"}}
        out = blink.decide("x", q)
        html = self.ui.render_answers("x", q, out)
        self.assertIn(self.ui.short_model(out["meta"]["model"]), html)
        self.assertIn("blk-by", html)

    def test_a_stale_column_says_what_to_do(self):
        html = self.ui.stale_html()
        self.assertIn(__import__('html').escape(self.ui.STALE_LABEL), html)
        self.assertNotIn("blk-answers", html)
        self.assertNotIn("%", html)

    def test_a_tab_move_and_a_row_restore_are_two_calls(self):
        """A Tabs update remounts the rows, so a restore sent with it is overwritten."""
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        src = inspect.getsource(app.build)
        self.assertIn("def restore_rows(n, held)", src)
        self.assertIn("restore_rows, back_in, pg[\"slots\"]", src)
        move = src.split('def go(slug, model, case=""):')[1].split("def ")[0]
        self.assertNotIn("_restore", move)  # the move itself never carries one
        self.assertIn("tabs.select(tab_selected", src)

    def test_a_model_change_forgets_the_answer_ask_was_holding(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        with serving("owner/one", "owner/two"):
            app = reload_app()
            src = inspect.getsource(app.build)
        self.assertIn('dict(draft, answers=None)', src)
        self.assertIn("held_in = [] if ask is None or mine else", src)

    def test_asks_empty_state_names_an_action_ask_has(self):
        self.assertIn(self.ui.ASK["stale"], self.ui.stale_html(self.ui.ASK["stale"]))
        self.assertNotIn("Decide", self.ui.ASK["stale"])
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        self.assertIn('ui.stale_html(ui.ASK["stale"])', inspect.getsource(app.ask_tab))

    def test_a_deep_link_sets_both_halves_of_the_address(self):
        """Leaving the case empty made the sync delete the case the link asked for."""
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        src = inspect.getsource(app.build)
        opened = src.split("def opened_by_url(")[1].split("\n        demo.load")[0]
        self.assertIn('case or ""', opened)
        self.assertIn("at_case", src)
        self.assertIn("tab === ", app.SYNC_URL)  # a case belongs to its own tab
        self.assertIn(json.dumps(self.ui.CASE_TAB), app.SYNC_URL)

    def test_a_model_change_clears_every_other_answer(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        with serving("owner/one", "owner/two"):
            app = reload_app()
            src = inspect.getsource(app.build)
        self.assertIn("[ui.stale_html(w) for w in words]", src)
        self.assertIn("stale = [p for j, p in enumerate(panels) if j != i]", src)

    def test_a_global_model_change_reaches_every_other_selector(self):
        """S1: one update dict shared between selectors arrives empty at all but one."""
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        with serving("owner/one", "owner/two"):
            app = reload_app()
            demo = app.build()
            try:
                wired = [f.fn for f in demo.fns.values()
                         if getattr(getattr(f, "fn", None), "__name__", "") == "switched"]
                self.assertTrue(wired)
                for switched in wired:
                    out = switched("owner/two")
                    picks = [x for x in out if isinstance(x, dict)
                             and x.get("value") == "owner/two"]
                    self.assertTrue(picks)
                    # distinct objects: gradio pops "value" out of the dict it is handed
                    self.assertEqual(len({id(p) for p in picks}), len(picks))
                    for pick in picks:
                        self.assertEqual(pick.pop("value"), "owner/two")
                    self.assertTrue(all("value" not in p for p in picks))
            finally:
                demo.close()

    def test_asks_panel_keeps_its_own_words_when_another_tab_switches(self):
        """S3: the cross-tab handler must not tell the ask panel to click Decide."""
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        with serving("owner/one", "owner/two"):
            app = reload_app()
            demo = app.build()
            try:
                said = set()
                for f in demo.fns.values():
                    fn = getattr(f, "fn", None)
                    if getattr(fn, "__name__", "") != "switched":
                        continue
                    said.update(x for x in fn("owner/two")
                                if isinstance(x, str) and "blk-empty" in x)
            finally:
                demo.close()
        asked = [x for x in said if self.ui.ASK["stale"] in x]
        self.assertTrue(asked)  # the ask panel is reached, in its own words
        for html in asked:
            self.assertNotIn("Decide", html)
            self.assertNotIn(results_escape(self.ui.STALE_LABEL), html)

    def test_the_ask_tab_answers_again_with_the_new_model(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        with serving("owner/one", "owner/two"):
            app = reload_app()
            src = inspect.getsource(app.ask_tab)
        self.assertIn("def redecide(", src)
        self.assertIn("model_in.input(", src)
        self.assertNotIn("author.draft", src.split("def redecide(")[1])

    def test_an_overtaken_ask_never_paints_under_a_newer_model(self):
        """S2: an ask in flight, a model chosen inside Ask, then the old one lands."""
        need_gradio(self)
        self.addCleanup(reload_app)
        with serving("owner/one", "owner/two"):
            app = reload_app()
            demo = app.build()
            try:
                wired = handlers(demo)
                flow = wired["ask_flow"]
                seen = {}
                steps = flow("1", seen, "How many r in strawberry", "owner/one")
                next(steps)                                # drafting
                wired["picked_model"]("2", seen, None)     # a model chosen meanwhile
                self.assertEqual(list(steps), [])          # nothing else is painted
                # left alone, the same ask shows its draft and its answer as before
                done = list(flow("3", {}, "How many r in strawberry", "owner/one"))
                self.assertEqual(len(done), 3)
                painted = [v for v in done[-1].values() if isinstance(v, str)]
                carried = [v for v in done[-1].values() if isinstance(v, dict)]
                self.assertTrue(any("blk-answers" in v for v in painted))
                # the answer is filed under the model that produced it
                self.assertEqual([c["model"] for c in carried], ["owner/one"])
            finally:
                demo.close()

    def test_a_second_model_answers_only_while_it_is_still_the_choice(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        with serving("owner/one", "owner/two"):
            app = reload_app()
            demo = app.build()
            try:
                redecide = handlers(demo)["redecide"]
                draft = {"state": "x", "model": "owner/one",
                         "questions": {"a": {"type": "noul", "instructions": "?"}}}
                seen = {"ask": 9}
                answers, kept = redecide("8", seen, draft, "owner/two")
                self.assertEqual(answers, gr_skip())
                self.assertEqual(kept, gr_skip())
                answers, kept = redecide("9", seen, draft, "owner/two")
                self.assertIn("blk-answers", answers)
                self.assertEqual(kept["model"], "owner/two")
            finally:
                demo.close()

    def test_the_hand_off_carries_only_the_answer_that_matches(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        with serving("owner/one", "owner/two"):
            app = reload_app()
            demo = app.build()
            try:
                draft = {"state": "x", "model": "owner/one",
                         "answers": '<div class="blk-answers">counted</div>',
                         "questions": {"a": {"type": "noul", "instructions": "?"}}}
                same = app.into_playground("1", {}, draft, "owner/one")
                other = app.into_playground("1", {}, draft, "owner/two")
            finally:
                demo.close()
        self.assertIn("counted", same[5])
        self.assertNotIn("counted", other[5])
        self.assertIn(results_escape(self.ui.STALE_LABEL), other[5])

    def test_add_question_adds_exactly_one_row(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            src = inspect.getsource(app.playground_tab)
            add = src.split("def add_row(")[1].split("def ")[0]
        finally:
            demo.close()
        # the count is carried, never guessed from values a hidden row already has
        self.assertIn("n + 1", add)
        self.assertIn("shown", src)
        self.assertNotIn("any(str(v).strip()", src)

    def test_the_row_count_is_its_own_state(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            states = [b for b in demo.blocks.values() if isinstance(b, gr.State)]
            self.assertTrue(any(b.value == 3 for b in states))  # the invoice preset
        finally:
            demo.close()

    def test_filling_can_lock_the_rows(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        rows = [dict(app.ui.BLANK_ROW, name="a")]
        open_ = app._fill(rows, 1, locked=False)
        shut = app._fill(rows, 1, locked=True)
        self.assertTrue(open_[app.SLOTS]["interactive"])
        self.assertFalse(shut[app.SLOTS]["interactive"])
        self.assertTrue(open_[0]["visible"])
        self.assertFalse(open_[1]["visible"])

    def test_the_claim_about_generated_text_follows_the_tab(self):
        plain, drafting = self.ui.masthead(), self.ui.masthead(drafting=True)
        self.assertIn(self.ui.MAST["lede"], plain)
        self.assertIn(self.ui.MAST["ask_lede"], drafting)
        self.assertNotIn("No generated text", drafting)
        # no token-count chip on any tab: it read as jargon
        for head in (plain, drafting):
            self.assertNotIn("generated tokens", head)
            self.assertNotIn("writes no text", head)
        self.assertNotIn("chip", self.ui.MAST)
        # the model tag survives either way
        self.assertIn("owner/model", self.ui.masthead("owner/model", drafting=True))

    def test_the_ask_tab_switches_the_claim(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        src = inspect.getsource(app.build)
        self.assertIn("masthead_for(slug, case, model)", src)
        self.assertIn("drafting=tab == ui.ASK_TAB", inspect.getsource(app.masthead_for))


class TestLayout(unittest.TestCase):
    """One centred column, one left edge, and the editor folded below Decide."""

    def css(self) -> str:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css"),
                  encoding="utf-8") as fh:
            return fh.read()

    def test_the_column_is_centred_with_even_gutters(self):
        css = self.css()
        self.assertIn("margin-left: auto !important", css)
        self.assertIn("margin-right: auto !important", css)
        self.assertIn("max-width: 1280px !important", css)
        # one gutter on both sides, 16px on a phone and 32px from a tablet up: gradio
        # scopes media queries inside its content wrapper, so the gutter scales by clamp
        self.assertIn("padding: 0 clamp(16px, 4vw, 32px) 8px !important", css)

    def test_every_block_shares_the_same_left_edge(self):
        css = self.css()
        head = css[css.index(".gradio-container .block,"):]
        head = head[: head.index("}")]
        for sel in (".gradio-container .form", ".gradio-container .html-container"):
            self.assertIn(sel, head)
        self.assertIn("padding-left: 0 !important", head)
        self.assertIn("padding-right: 0 !important", head)

    def test_narrow_widths_stack_and_wrap(self):
        css = self.css()
        self.assertIn("@media (max-width: 860px)", css)
        self.assertIn("flex-direction: column !important", css)
        self.assertIn("@media (max-width: 720px)", css)
        self.assertIn("flex-wrap: wrap !important", css)

    def test_no_multi_line_values_in_the_rules_we_added(self):
        """gradio's css scoper drops a declaration whose value wraps."""
        css = self.css()
        for i, line in enumerate(css.splitlines(), 1):
            stripped = line.strip()
            if stripped.endswith("(") and not stripped.startswith(("@", "/*", "*")):
                self.fail(f"style.css:{i} opens a wrapped value: {stripped}")

    def test_the_state_then_the_questions_then_decide(self):
        """The owner read an answer as being about the state; the order has to say otherwise."""
        need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            rows = sorted(
                (b for b in demo.blocks.values()
                 if "blk-qrow" in (getattr(b, "elem_classes", None) or [])),
                key=lambda b: b._id,
            )
            decide = next(b for b in demo.blocks.values()
                          if "blk-decide" in (getattr(b, "elem_classes", None) or []))
            state = next(b for b in demo.blocks.values()
                         if "blk-state" in (getattr(b, "elem_classes", None) or []))
            acc = next(b for b in demo.blocks.values()
                       if "blk-acc" in (getattr(b, "elem_classes", None) or []))
            self.assertLess(state._id, rows[0]._id)
            self.assertLess(rows[0]._id, decide._id)
            self.assertLess(acc._id, decide._id)  # Decide comes after every input
            self.assertFalse(acc.open)  # only the JSON view is folded
        finally:
            demo.close()

    def test_the_editor_is_still_a_working_json_editor(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            code = next(b for b in demo.blocks.values() if isinstance(b, gr.Code))
            self.assertEqual(code.language, "json")
            self.assertNotEqual(code.interactive, False)
            self.assertTrue(code.wrap_lines)
            self.assertEqual(code.value, examples.PLAYGROUND_QUESTIONS)
            acc = next(b for b in demo.blocks.values()
                       if "blk-acc" in (getattr(b, "elem_classes", None) or []))
            self.assertFalse(acc.open)
            self.assertIn("blk-acc", acc.elem_classes)
        finally:
            demo.close()

    def test_the_answer_ends_in_one_footer(self):
        import ui

        q = {"urgent": {"type": "noul", "instructions": "Now?"}}
        out = blink.decide("x", q)
        html = ui.render_answers("x", q, out)
        self.assertEqual(html.count('class="blk-answer-foot"'), 1)
        foot = html[html.index('class="blk-answer-foot"'):]
        self.assertIn("blk-stats", foot)
        self.assertIn("blk-more", foot)


class TestModelSwitcher(unittest.TestCase):
    """One model: the page looks exactly as before. Two: a control that reaches decide."""

    A = "thegovind/blink-4b"
    B = "thegovind/blink-mimo-9b"
    Q = {"urgent": {"type": "noul", "instructions": "Today?"}}
    QTEXT = json.dumps(Q)

    def test_short_names_come_from_the_id(self):
        import ui

        self.assertEqual(ui.short_model("thegovind/blink-mimo-9b"), "blink-mimo-9b")
        self.assertEqual(ui.short_model("blink-4b"), "blink-4b")

    def test_a_single_model_shows_no_control(self):
        gr = need_gradio(self)
        import ui

        self.addCleanup(reload_app)
        app = reload_app()
        self.assertFalse(app.MULTI)
        self.assertEqual(len(ui.model_choices()), 1)
        with gr.Blocks() as page:
            widget = app.model_switch()
        page.close()
        self.assertIsInstance(widget, gr.State)
        demo = app.build()
        self.assertFalse(
            any("blk-seg" in (getattr(b, "elem_classes", None) or []) for b in demo.blocks.values())
        )
        demo.close()

    def test_two_models_give_a_segmented_control(self):
        gr = need_gradio(self)
        import ui

        self.addCleanup(reload_app)
        with serving(self.A, self.B):
            app = reload_app()
            self.assertTrue(app.MULTI)
            self.assertEqual(
                ui.model_choices(), [("blink-4b", self.A), ("blink-mimo-9b", self.B)]
            )
            with gr.Blocks() as page:
                widget = app.model_switch()
            page.close()
            self.assertIsInstance(widget, gr.Radio)
            self.assertEqual(widget.value, self.A)
            self.assertIn("blk-seg", widget.elem_classes)
            self.assertEqual([c[1] for c in widget.choices], [self.A, self.B])
            demo = app.build()
            segs = [b for b in demo.blocks.values()
                    if "blk-seg" in (getattr(b, "elem_classes", None) or [])]
            # one on each tab a visitor can run: playground, ask, every use case
            self.assertEqual(len(segs), 2 + len(examples.USE_CASES))
            demo.close()

    def test_decide_answers_as_the_chosen_model(self):
        with serving(self.A, self.B):
            self.assertEqual(blink.models(), [self.A, self.B])
            self.assertEqual(blink.decide("x", self.Q)["meta"]["model"], self.A)
            self.assertEqual(blink.decide("x", self.Q, model=self.B)["meta"]["model"], self.B)
            with self.assertRaises(blink.BlinkError):
                blink.decide("x", self.Q, model="someone/else")

    def test_each_model_renders_its_own_answer(self):
        import ui

        with serving(self.A, self.B):
            first = ui.run_playground("Card charged twice.", self.QTEXT, self.A)
            second = ui.run_playground("Card charged twice.", self.QTEXT, self.B)
            self.assertIn(self.A, first)
            self.assertIn(self.B, second)
            self.assertNotEqual(first, second)
            self.assertIn("blk-answers", second)

    def test_the_default_model_answers_the_same_as_before(self):
        alone = blink.decide("Card charged twice.", self.Q)["answers"]
        with serving(self.A, self.B):
            chosen = blink.decide("Card charged twice.", self.Q, model=self.A)["answers"]
        self.assertEqual(alone, chosen)

    def test_the_masthead_tag_follows_the_selection(self):
        import ui

        self.assertIn(results_escape(self.B), ui.masthead(self.B))
        self.assertIn(results_escape(self.A), ui.masthead(self.A))

    def test_the_use_case_runner_takes_a_model(self):
        import ui

        case = examples.USE_CASES[0]
        with serving(self.A, self.B):
            html = ui.run_use_case(case, case.examples[0].state, self.B)
        self.assertIn(self.B, html)
        self.assertIn("blk-verdict", html)

    def test_the_api_accepts_a_model(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        with serving(self.A, self.B):
            app = reload_app()
            self.assertEqual(app.systemone("x", self.Q)["meta"]["model"], self.A)
            self.assertEqual(
                app.systemone("x", self.Q, None, self.B)["meta"]["model"], self.B
            )
            self.assertIn("model", inspect.signature(app.systemone).parameters)

    def test_a_switch_re_runs_that_tab_live(self):
        need_gradio(self)
        self.addCleanup(reload_app)
        with serving(self.A, self.B):
            app = reload_app()
            src = inspect.getsource(app)
        self.assertIn("model_in.input(", src)
        self.assertIn("[state, model_in, touched, rev, seen, *run_in]", src)
        self.assertIn("[state, model_in]", src)


class TestLatencyElement(unittest.TestCase):
    """latency is null until replay.json exists; the element must simply not be there."""

    LAT = {
        "requests": 21,
        "p50_ms": 64.0,
        "max_ms": 141.2,
        "generated_tokens": 0,
        "caption": "End to end per request, bundled examples.",
    }

    def test_results_json_latency_is_null_or_complete(self):
        import results

        lat = results.load().get("latency")
        if lat is not None:
            self.assertEqual(set(self.LAT), set(lat))

    def test_strip_reports_the_recorded_numbers(self):
        import results

        html = results.latency_strip(self.LAT)
        self.assertIn("64", html)
        self.assertIn("141.2", html)
        self.assertNotIn("accelerator", html)
        self.assertIn("bundled examples", html)
        self.assertNotIn("generated", html)

    def test_results_never_count_generated_tokens(self):
        import ui

        for block in ui.results_blocks():
            self.assertNotIn("tokens generated", block)
            self.assertNotIn("</b> generated", block)

    def test_app_builds_with_latency_null(self):
        gr = need_gradio(self)
        import results

        self.addCleanup(reload_app)  # put the unpatched data back for later tests
        data = dict(results.load(), latency=None)
        with unittest.mock.patch.object(results, "load", lambda: data):
            app = reload_app()
            self.assertIsNone(app.DATA["latency"])
            with gr.Blocks() as page:
                app.results_tab()
            page.close()

    def test_app_builds_with_latency_present(self):
        gr = need_gradio(self)
        import results

        self.addCleanup(reload_app)
        data = dict(results.load(), latency=self.LAT)
        with unittest.mock.patch.object(results, "load", lambda: data):
            app = reload_app()
            self.assertEqual(app.DATA["latency"], self.LAT)
            with gr.Blocks() as page:
                app.results_tab()
            page.close()

    def test_whole_app_assembles(self):
        need_gradio(self)
        app = reload_app()
        demo = app.build()
        self.assertTrue(demo.blocks)
        demo.close()


class TestReplay(unittest.TestCase):
    """Replay mode serves recorded logits; temperature and assembly still run live."""

    def setUp(self):
        import tempfile

        import examples

        self.examples = examples
        self.q = {
            "queue": {"type": "choice", "instructions": "Owner?", "criteria": {"billing": "b", "tech": "t"}},
            "urgent": {"type": "noul", "instructions": "Today?"},
        }
        self.state = "My card was charged twice."
        rec = {
            blink.request_key(self.state, self.q): {
                "names": ["t"],
                "logits": {"queue": [2.0, 0.0], "urgent": [0.0, 1.0]},
                "input_tokens": 321,
                "latency_ms": 41.5,
            }
        }
        fd, self.path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump({"model": "m/x", "hardware": "accelerator", "requests": rec}, fh)
        self.eng = blink.ReplayEngine(self.path, temperature=1.0)
        self._saved = blink._ENGINE
        blink._ENGINE = self.eng

    def tearDown(self):
        blink._ENGINE = self._saved
        os.unlink(self.path)

    def test_hit_uses_recorded_logits_and_latency(self):
        out = blink.decide(self.state, self.q)
        p = out["answers"]["queue"]["probabilities"]
        self.assertAlmostEqual(p["billing"], 1 / (1 + pow(2.718281828459045, -2.0)), places=6)
        self.assertEqual(out["answers"]["queue"]["choice"], "billing")
        self.assertLess(out["answers"]["urgent"]["noul"], 0.5)
        self.assertEqual(out["meta"]["engine"], "replay")
        self.assertEqual(out["meta"]["latency_ms"], 41.5)
        self.assertEqual(out["meta"]["recorded_on"], "accelerator")
        self.assertEqual(out["meta"]["input_tokens"], 321)

    def test_temperature_still_applies(self):
        hot = blink.decide(self.state, self.q, temperature=4.0)["answers"]["queue"]["probabilities"]
        cold = blink.decide(self.state, self.q)["answers"]["queue"]["probabilities"]
        self.assertLess(hot["billing"], cold["billing"])

    def test_key_ignores_edge_whitespace_but_not_order(self):
        self.assertEqual(blink.request_key(" a\r\nb \n", self.q), blink.request_key("a\nb", self.q))
        flipped = dict(reversed(list(self.q.items())))
        self.assertNotEqual(blink.request_key(self.state, flipped), blink.request_key(self.state, self.q))

    def test_miss_raises_a_blink_error(self):
        with self.assertRaises(blink.BlinkError):
            blink.decide("something else entirely", self.q)

    def test_miss_is_a_replay_miss_that_names_the_way_out_and_no_hardware(self):
        with self.assertRaises(blink.ReplayMiss) as caught:
            blink.decide("something else entirely", self.q)
        msg = str(caught.exception)
        self.assertIn("blink.py", msg)
        self.assertNotIn("accelerator", msg)
        self.assertNotIn("GPU", msg)
        self.assertLess(len(msg), 260)
        self.assertTrue(issubclass(blink.ReplayMiss, blink.BlinkError))

    def test_missing_hardware_still_gives_a_sentence(self):
        eng = blink.ReplayEngine(self.path, temperature=1.0)
        eng.hardware = ""
        with self.assertRaises(blink.ReplayMiss) as caught:
            eng.logits("nothing recorded", self.q)
        self.assertIn("blink.py", str(caught.exception))

    def test_a_miss_renders_as_a_notice_not_an_error(self):
        need_gradio(self)
        app = reload_app()
        app.blink._ENGINE = self.eng
        try:
            html = app.run_playground("an input nobody recorded", json.dumps(self.q))
        finally:
            app.blink._ENGINE = self._saved
        self.assertIn("blk-notice", html)
        self.assertNotIn("blk-warn", html)
        self.assertIn("blink.py", html)

    def test_a_hit_still_renders_answers(self):
        need_gradio(self)
        app = reload_app()
        app.blink._ENGINE = self.eng
        try:
            html = app.run_playground(self.state, json.dumps(self.q))
        finally:
            app.blink._ENGINE = self._saved
        self.assertIn("blk-answers", html)
        self.assertNotIn("blk-notice", html)

    def _recording(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay.json")
        if not os.path.exists(path):
            self.skipTest("replay.json not recorded yet")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)["requests"]

    def test_bundled_replay_covers_every_example(self):
        rec = self._recording()
        ex = self.examples
        for case in ex.USE_CASES:
            if case.key in PENDING_REPLAY:
                continue
            for e in case.examples:
                self.assertIn(blink.request_key(e.state, case.questions), rec, f"{case.key}/{e.label}")
        for label, s, qtext in ex.PLAYGROUND_PRESETS:
            self.assertIn(blink.request_key(blink.as_state(s), json.loads(qtext)), rec, label)

    def test_pending_replay_list_is_still_needed(self):
        rec = self._recording()
        for key in PENDING_REPLAY:
            case = self.examples.CASES_BY_KEY[key]
            covered = all(
                blink.request_key(e.state, case.questions) in rec for e in case.examples
            )
            self.assertFalse(covered, f"{key} is recorded now; drop it from PENDING_REPLAY")

    def test_an_unrecorded_example_degrades_to_the_notice(self):
        import ui

        # any bundled case misses the one-request fixture recording above
        case = self.examples.CASES_BY_KEY["nextclick"]
        saved = blink._ENGINE
        blink._ENGINE = self.eng
        try:
            html = ui.run_use_case(case, case.examples[0].state, prefer="saved")
        finally:
            blink._ENGINE = saved
        self.assertIn("blk-notice", html)
        self.assertNotIn("blk-answers", html)


class TestStaticBuild(unittest.TestCase):
    """The static Space is only allowed to exist when a real recording covers every request."""

    def setUp(self):
        import shutil
        import tempfile

        import build_static
        import record_replay

        self.bs = build_static
        self.bundled = record_replay.bundled_requests()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.replay = os.path.join(self.tmp, "replay.json")
        self.out = os.path.join(self.tmp, "static")
        write_synthetic_replay(self.replay)

    def built(self) -> tuple[str, str]:
        self.bs.build(self.out, self.replay, synthetic=True)
        with open(os.path.join(self.out, "index.html"), encoding="utf-8") as fh:
            html = fh.read()
        with open(os.path.join(self.out, "style.css"), encoding="utf-8") as fh:
            css = fh.read()
        return html, css

    def test_refuses_when_the_recording_is_missing(self):
        with self.assertRaises(self.bs.BuildError) as caught:
            self.bs.build(self.out, os.path.join(self.tmp, "nothing.json"))
        self.assertIn("record_replay.py", str(caught.exception))
        self.assertFalse(os.path.exists(self.out))

    def test_refuses_a_recording_that_misses_a_bundled_request(self):
        partial = os.path.join(self.tmp, "partial.json")
        write_synthetic_replay(partial, drop=2)
        with self.assertRaises(self.bs.BuildError) as caught:
            self.bs.build(self.out, partial, synthetic=True)
        self.assertIn("missing", str(caught.exception))
        self.assertFalse(os.path.exists(self.out))

    def test_refuses_a_synthetic_recording_without_the_flag(self):
        with self.assertRaises(self.bs.BuildError) as caught:
            self.bs.build(self.out, self.replay)
        self.assertIn("--allow-synthetic", str(caught.exception))

    def test_cli_reports_a_refusal_and_writes_nothing(self):
        import contextlib
        import io

        partial = os.path.join(self.tmp, "partial-cli.json")
        write_synthetic_replay(partial, drop=2)  # independent of whether space/replay.json exists
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = self.bs.main(["--out", self.out, "--allow-synthetic", partial])
        self.assertEqual(code, 2)
        self.assertIn("build_static:", err.getvalue())
        self.assertFalse(os.path.exists(self.out))

    def test_synthetic_marker_is_visible_in_the_page_and_the_readme(self):
        html, _ = self.built()
        self.assertIn(self.bs.SYNTHETIC_NOTE, html)
        self.assertIn('data-synthetic="1"', html)
        self.assertIn("synthetic preview</title>", html)
        with open(os.path.join(self.out, "README.md"), encoding="utf-8") as fh:
            self.assertIn(self.bs.SYNTHETIC_NOTE, fh.read())

    def test_a_real_recording_leaves_no_marker(self):
        real = os.path.join(self.tmp, "real.json")
        data = write_synthetic_replay(real)
        data["engine"] = "TorchEngine (bf16 weights, FP32 label projection, sdpa)"
        with open(real, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        self.bs.build(self.out, real)
        with open(os.path.join(self.out, "index.html"), encoding="utf-8") as fh:
            html = fh.read()
        self.assertNotIn(self.bs.SYNTHETIC_NOTE, html)
        self.assertNotIn("data-synthetic", html)

    def test_every_bundled_request_is_pre_rendered(self):
        html, _ = self.built()
        for name, state, qs in self.bundled:
            key = blink.request_key(state, qs)
            self.assertIn(f'data-key="{key}"', html, name)
        import examples

        chips = len(examples.PLAYGROUND_PRESETS) + sum(len(c.examples) for c in examples.USE_CASES)
        self.assertEqual(html.count('class="blk-out'), chips + 1 + len(examples.USE_CASES))

    def test_answers_carry_the_saved_run_tag_and_no_mock_banner(self):
        html, _ = self.built()
        self.assertIn(" ms recorded call \u00b7 saved run", html)
        self.assertNotIn("accelerator", html)
        self.assertNotIn("Mock engine", html)
        self.assertIn("blk-notice", html)

    def test_every_tab_and_both_figures_are_in_the_page(self):
        import examples

        html, _ = self.built()
        for label in ("Playground", "Use cases", "Results", "How it works"):
            self.assertIn(f">{label}</button>", html)
        for case in examples.USE_CASES:
            self.assertIn(f'id="pane-case-{case.key}"', html)
        self.assertIn("Decision Index by system", html)
        import results

        pa = results.load()["pareto"]
        self.assertIn(f'{pa["x_label"]} against {pa["y_label"]}', html)
        self.assertNotIn("gradio_client", html)

    def test_no_remote_resources(self):
        html, css = self.built()
        with open(os.path.join(self.out, "app.js"), encoding="utf-8") as fh:
            js = fh.read()
        remote = re.findall(r'(?:src|srcset)="(https?://[^"]+)"', html)
        remote += re.findall(r'<link[^>]+href="(https?://[^"]+)"', html)
        remote += re.findall(r'url\(\s*["\']?(https?://[^)"\']+)', html + css)
        remote += re.findall(r'@import\s+(?:url\()?["\'](https?://[^"\']+)', css)
        self.assertEqual(remote, [])
        for url in re.findall(r"https?://[^\s\"'<)]+", html + css + js):
            self.assertRegex(url, r"^https://(huggingface\.co|github\.com|thegovind\.github\.io)/")

    def test_demo_label_and_footer_are_on_the_static_page(self):
        import ui

        html, _ = self.built()
        self.assertEqual(
            html.count(ui.DEMO_LABEL.split("\u00b7")[0].strip()), 1 + len(examples.USE_CASES)
        )
        self.assertNotIn("GPU", html)
        self.assertIn("Weights are for non-commercial research and evaluation", html)
        self.assertNotIn("Personal research release", html)
        self.assertEqual(html.count('class="blk-foot"'), 1)

    def test_static_page_carries_the_qualified_claims(self):
        html, _ = self.built()
        for phrase in (
            "Decision Index 0.1",
            "The live board moved to 0.2 on 2026-09-24",
            "no 0.2 result here",
            "3,000-request DI-S sample picked the prompt and checkpoints",
            "development proxies computed here",
            "No hard-accuracy advantage is claimed",
            "Longer inputs are refused, not truncated",
            ">saved run \u00b7 blink-4b</em>",
        ):
            self.assertIn(phrase, html, phrase)
        self.assertNotIn("same kind of 0.7", html)
        self.assertNotIn("simulated", html)
        self.assertIn("Weights are for non-commercial research and evaluation; app code is Apache-2.0.", html)
        self.assertEqual(html.count("Apache-2.0"), 1)

    def test_writes_a_static_space_readme(self):
        self.built()
        with open(os.path.join(self.out, "README.md"), encoding="utf-8") as fh:
            head = fh.read().split("---")[1]
        self.assertIn("sdk: static", head)
        self.assertIn("app_file: index.html", head)
        self.assertIn("option probabilities", head)
        self.assertNotIn("sdk_version", head)
        self.assertNotIn("suggested_hardware", head)
        for model in ("thegovind/blink-4b", "thegovind/blink-27b"):
            self.assertIn(model, head)

    def test_page_stays_light(self):
        self.built()
        total = sum(
            os.path.getsize(os.path.join(self.out, f)) for f in os.listdir(self.out)
        )
        self.assertLess(total, 3 * 1024 * 1024)


class TestSelectionDisclosure(unittest.TestCase):
    def test_results_page_discloses_selection_next_to_the_headline(self):
        import ui

        html = "".join(ui.results_blocks())
        self.assertIn("The 3,000-request DI-S sample picked the prompt and checkpoints", html)
        self.assertIn("post-selection", html)
        self.assertLess(html.index("DI-S sample"), html.index("<svg"))


class TestManyModels(unittest.TestCase):
    """BLINK_MODELS: several models side by side, each with its own engine and recording."""

    OTHER = "thegovind/blink-mimo-9b"

    def setUp(self):
        self._specs, self._engine, self._engines = blink.MODEL_SPECS, blink._ENGINE, dict(blink._ENGINES)
        blink.MODEL_SPECS = [(blink.MODEL_ID, None), (self.OTHER, "abc123")]
        blink._ENGINES.clear()
        self.q = {"queue": {"type": "choice", "instructions": "Route it.",
                            "criteria": {"billing": "Charges", "technical": "Bugs"}}}

    def tearDown(self):
        blink.MODEL_SPECS, blink._ENGINE = self._specs, self._engine
        blink._ENGINES.clear()
        blink._ENGINES.update(self._engines)

    def test_specs_parse_repo_and_revision(self):
        with unittest.mock.patch.dict(os.environ, {"BLINK_MODELS": " a/one , b/two@rev9 ,, "}):
            self.assertEqual(blink._model_specs(), [("a/one", None), ("b/two", "rev9")])
        with unittest.mock.patch.dict(os.environ, {"BLINK_MODELS": ""}):
            self.assertEqual(blink._model_specs(), [(blink.MODEL_ID, blink.MODEL_REVISION)])

    def test_models_lists_default_first(self):
        self.assertEqual(blink.models(), [blink.MODEL_ID, self.OTHER])

    def test_each_model_has_its_own_recording(self):
        self.assertEqual(blink.replay_path(), blink.REPLAY_PATH)
        self.assertEqual(blink.replay_path(blink.MODEL_ID), blink.REPLAY_PATH)
        self.assertTrue(blink.replay_path(self.OTHER).endswith("replay-blink-mimo-9b.json"))

    def test_unknown_model_is_a_blink_error(self):
        with self.assertRaises(blink.BlinkError) as caught:
            blink.decide("x", self.q, model="someone/else")
        self.assertIn(self.OTHER, str(caught.exception))

    def test_decide_routes_to_the_chosen_model(self):
        blink._ENGINE = blink.MockEngine(blink.MODEL_ID)
        a = blink.decide("My card was charged twice.", self.q)
        b = blink.decide("My card was charged twice.", self.q, model=self.OTHER)
        self.assertEqual(a["meta"]["model"], blink.MODEL_ID)
        self.assertEqual(b["meta"]["model"], self.OTHER)
        self.assertIs(blink.engine(self.OTHER), blink.engine(self.OTHER))
        self.assertNotEqual(a["answers"]["queue"]["probabilities"], b["answers"]["queue"]["probabilities"])

    def test_default_mock_numbers_do_not_move(self):
        state = "My card was charged twice."
        self.assertEqual(blink.MockEngine().logits(state, self.q), blink.MockEngine(blink.MODEL_ID).logits(state, self.q))

    def test_warm_builds_every_engine(self):
        blink._ENGINE = blink.MockEngine(blink.MODEL_ID)
        engines = blink.warm()
        self.assertEqual([e.model_id for e in engines], blink.models())

    def _recording(self, model):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump({"model": model, "requests": {}}, fh)
        self.addCleanup(os.unlink, path)
        return path

    def test_a_recording_from_another_model_is_not_used(self):
        path = self._recording("Qwen/Qwen3.5-4B")
        with unittest.mock.patch.object(blink, "replay_path", lambda m=None: path):
            self.assertIsNone(blink._matching_replay(blink.MODEL_ID))
        path = self._recording(blink.MODEL_ID)
        with unittest.mock.patch.object(blink, "replay_path", lambda m=None: path):
            self.assertEqual(blink._matching_replay(blink.MODEL_ID).model_id, blink.MODEL_ID)

    def test_hybrid_without_a_recording_runs_live(self):
        class Live:
            model_id, temperature, calls = blink.MODEL_ID, 1.0, 0

            def logits(self, state, questions, bias=None):
                Live.calls += 1
                return {"queue": [1.0, 0.0]}, 7

        eng = blink.HybridEngine(None, Live())
        self.assertEqual(eng.hardware, "")
        eng.logits("x", self.q, prefer="saved")
        self.assertEqual((Live.calls, eng.last_source), (1, "torch"))


class TestMastheadBase(unittest.TestCase):
    """The base-model tag follows the selected model."""

    def test_each_model_names_its_own_base(self):
        import ui

        self.assertIn("Qwen/Qwen3.5-4B", ui.masthead("thegovind/blink-4b"))
        mimo = ui.masthead("thegovind/blink-mimo-9b")
        self.assertIn("XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", mimo)
        self.assertNotIn("Qwen/Qwen3.5-4B ·", mimo)

    def test_unknown_model_falls_back_to_the_small_base(self):
        import ui

        self.assertIn(ui.DATA["model"]["base_small"], ui.masthead("someone/else"))


class TestSystemoneEndpoint(unittest.TestCase):
    """The JSON API: questions as an object or a JSON string; blink's refusals come back as their reason."""

    def setUp(self):
        need_gradio(self)
        self.app = reload_app()
        self.q = {"queue": {"type": "choice", "instructions": "Route it.",
                            "criteria": {"billing": "Charges", "technical": "Bugs"}}}

    def test_object_and_json_string_give_the_same_answers(self):
        a = self.app.systemone("My card was charged twice.", self.q)
        b = self.app.systemone("My card was charged twice.", json.dumps(self.q))
        self.assertEqual(a["answers"], b["answers"])

    def test_bad_json_is_a_readable_error(self):
        import gradio as gr

        with self.assertRaises(gr.Error) as caught:
            self.app.systemone("x", "{not json")
        self.assertIn("not valid JSON", str(caught.exception))

    def test_a_blink_refusal_keeps_its_reason(self):
        import gradio as gr

        with self.assertRaises(gr.Error) as caught:
            self.app.systemone("x", {})
        self.assertIn("non-empty", str(caught.exception))

    def test_the_response_carries_typesafes_fields_then_meta(self):
        out = self.app.systemone("My card was charged twice.", self.q)
        self.assertEqual(list(out), ["model", "answers", "usage", "meta"])
        self.assertEqual(out["model"], out["meta"]["model"])
        self.assertEqual(out["usage"], {"input_tokens": out["meta"]["input_tokens"], "output_tokens": 0})
        self.assertEqual(out["answers"], blink.decide("My card was charged twice.", self.q)["answers"])

    def test_typesafe_model_names_get_the_default_model(self):
        import gradio as gr

        for name in ("jev-latest", "JEV-1.13.0", "jev-preview", " jev "):
            with self.subTest(name=name):
                self.assertEqual(self.app.systemone("x", self.q, None, name)["model"], blink.MODEL_ID)
        with self.assertRaises(gr.Error) as caught:  # a blink model this deployment doesn't serve is still an error
            self.app.systemone("x", self.q, None, "thegovind/blink-9b")
        self.assertIn("unknown model", str(caught.exception))

    def test_the_state_may_be_json(self):
        state = {"message": "My card was charged twice.", "order": "A-104"}
        want = blink.decide(state, self.q)["answers"]
        self.assertEqual(self.app.systemone(state, self.q)["answers"], want)
        self.assertEqual(self.app.systemone(json.dumps(state), self.q)["answers"], want)


class TestApiTab(unittest.TestCase):
    """The API tab: its copy lives in api_doc, and every example is the real thing."""

    def setUp(self):
        import api_doc
        import ui

        self.d, self.ui = api_doc, ui

    def test_the_tab_has_a_slug_and_deep_links(self):
        self.assertIn(("api", "API"), self.ui.TABS)
        for raw in ("?tab=api", "#api", "?tab=API", "?__theme=dark&tab=api"):
            self.assertEqual(self.ui.parse_deep_link(raw), ("api", None))

    def test_the_example_response_is_blink_4bs_saved_answer(self):
        """What serve.py returns for EXAMPLE_REQUEST, from blink-4b's saved logits, rounded to three decimals."""
        req, saved = self.d.EXAMPLE_REQUEST, blink._ENGINE
        blink._ENGINE = blink.ReplayEngine(blink.replay_path("thegovind/blink-4b"))
        try:
            out = blink.decide(req["state"], req["questions"])
        finally:
            blink._ENGINE = saved
        self.assertEqual((out["meta"]["model"], out["meta"]["engine"]), ("thegovind/blink-4b", "replay"))

        def r3(x):
            if isinstance(x, float):
                return round(x, 3)
            if isinstance(x, dict):
                return {k: r3(v) for k, v in x.items()}
            return x

        served = {"model": self.d.SERVED_AS, "answers": out["answers"],
                  "usage": {"input_tokens": out["meta"]["input_tokens"], "output_tokens": 0}}
        self.assertEqual(json.dumps(r3(served)), json.dumps(self.d.EXAMPLE_RESPONSE))  # values and order

    def test_every_example_request_is_one_blink_accepts(self):
        for qs in (self.d.EXAMPLE_REQUEST["questions"], self.d.SHORT_REQUEST["questions"], self.d.SPACE_QUESTIONS):
            blink.validate(qs)
        self.assertEqual(self.d.EXAMPLE_REQUEST["questions"], json.loads(examples.PLAYGROUND_QUESTIONS))

    def test_the_refusal_is_blinks_own(self):
        with self.assertRaises(blink.BlinkError) as caught:
            blink.validate(self.d.REFUSED_REQUEST["questions"])
        reason = str(caught.exception)
        self.assertEqual(self.d.REFUSED_BODY, {
            "error": reason, "detail": [{"loc": ["body", "questions", "urgency"], "msg": reason,
                                         "type": "value_error"}]})
        self.assertEqual(self.d.UNAUTHORIZED_BODY["error"], self.d.UNAUTHORIZED_BODY["detail"])

    def test_json_blocks_round_trip(self):
        for obj in (self.d.EXAMPLE_REQUEST, self.d.EXAMPLE_RESPONSE, self.d.REFUSED_REQUEST, self.d.REFUSED_BODY,
                    self.d.UNAUTHORIZED_BODY, self.d.MODELS_BODY, self.d.SHORT_REQUEST):
            self.assertEqual(json.loads(self.d.pretty(obj)), obj)
        body = self.d.HTTP_CLIENT.split("<<'EOF'\n", 1)[1].rsplit("\nEOF", 1)[0]
        self.assertEqual(json.loads(body), self.d.SHORT_REQUEST)
        self.assertIn(f"{self.d.SERVER_URL}/v1/systemone", self.d.HTTP_CLIENT)

    def test_the_space_examples_match_the_endpoint(self):
        gr = need_gradio(self)
        del gr
        self.addCleanup(reload_app)
        app = reload_app()
        self.assertEqual(list(inspect.signature(app.systemone).parameters),
                         ["state", "questions", "temperature", "model"])
        data = json.loads(re.search(r"-d '(\{.*\})'", self.d.SPACE_CURL).group(1))["data"]
        self.assertEqual(data, [self.d.SPACE_STATE, self.d.SPACE_QUESTIONS, None, "thegovind/blink-4b"])
        self.assertIn("/gradio_api/call/v1_systemone", self.d.SPACE_CURL)
        self.assertIn('api_name="/v1_systemone"', self.d.SPACE_PYTHON)
        out = app.systemone(*data)
        self.assertEqual(list(out), ["model", "answers", "usage", "meta"])
        for key in ('"model": "thegovind/blink-4b"', '"output_tokens": 0', '"meta"'):
            self.assertIn(key, self.d.SPACE_REPLY)
        # the documented reply is a live one: the same structure, with the two timings a live engine adds
        lines = self.d.SPACE_REPLY.splitlines()
        self.assertEqual(lines[0], "event: complete")
        (reply,) = json.loads(lines[1].removeprefix("data: "))
        self.assertEqual(list(reply), list(out))
        self.assertEqual({q: list(a) for q, a in reply["answers"].items()}, {q: list(a) for q, a in out["answers"].items()})
        self.assertEqual(list(reply["usage"]), ["input_tokens", "output_tokens"])
        self.assertEqual(list(reply["meta"]), list(out["meta"]) + ["model_ms", "prefill_tokens"])
        self.assertEqual(reply["model"], "thegovind/blink-4b")
        noul = reply["answers"]["urgent"]
        self.assertAlmostEqual(noul["probabilities"]["yes"] + noul["probabilities"]["no"], 1.0, places=3)
        self.assertEqual(noul["noul"], round(noul["noul"], 3))
        self.assertIn("latency_ms and model_ms vary", lines[-1])

    def test_python_snippets_compile(self):
        for name in ("PYTHON_CLIENT", "SPACE_PYTHON"):
            compile(getattr(self.d, name), name, "exec")

    def test_downloads_pin_the_code_revision(self):
        """Every model download the tab shows gets the code revision that has this serve.py and blink.py."""
        self.assertEqual(self.d.CODE_REVISION, "v1.4")
        texts = [v for v in vars(self.d).values() if isinstance(v, str)]
        texts += [s for v in self.d.COPY.values() for s in (v if isinstance(v, tuple) else (v,)) if isinstance(s, str)]
        lines = [line for t in texts for line in t.splitlines() if "hf download" in line or "hf_hub_download" in line]
        self.assertTrue(lines)
        for line in lines:
            self.assertIn("--revision v1.4", line)

    def test_the_how_it_works_snippet_uses_the_same_revision(self):
        """How it works runs blink.py in-process; it downloads the same code revision the API tab pins."""
        snippet = self.ui.API_SNIPPET
        self.assertIn(f'os.environ["BLINK_REVISION"] = "{self.d.CODE_REVISION}"', snippet)
        self.assertIn(f'hf_hub_download("thegovind/blink-4b", "blink.py", revision="{self.d.CODE_REVISION}")', snippet)
        self.assertNotIn("v1.0", snippet)
        self.assertNotIn("@REVISION@", snippet)
        compile(snippet, "API_SNIPPET", "exec")
        self.assertIn(self.ui.esc(snippet), "".join(self.ui.how_blocks()))

    def test_the_table_is_complete(self):
        marks = {k for k, _ in self.d.COPY["legend"]}
        self.assertEqual(len(self.d.TABLE_COLS), 4)
        for row in self.d.TABLE_ROWS:
            self.assertEqual(len(row), 4, row)
            self.assertIn(row[2][0], marks)
            self.assertIn(row[3][0], marks)

    def test_the_blocks_show_all_the_copy(self):
        html = "".join(self.ui.api_blocks())

        def strings(x):
            if isinstance(x, str):
                yield x
            elif isinstance(x, (tuple, list)):
                for v in x:
                    yield from strings(v)

        for key, value in self.d.COPY.items():
            for s in strings(value):
                with self.subTest(key=key, s=s[:40]):
                    self.assertIn(self.ui.esc(s), html)
        for row in self.d.TABLE_ROWS:
            self.assertIn(self.ui.esc(row[1]), html)
        self.assertEqual("blk-copytag" in html, self.d.DRAFT)
        for code in (self.d.SERVER_RUN, self.d.DOCKER_RUN, self.d.PYTHON_CLIENT, self.d.JS_CLIENT,
                     self.d.HTTP_CLIENT, self.d.SPACE_PYTHON, self.d.SPACE_CURL):
            self.assertIn(self.ui.esc(code), html)

    def test_the_tab_is_built(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            tabs = {b.id: b.label for b in demo.blocks.values() if isinstance(b, gr.Tab)}
        finally:
            demo.close()
        self.assertEqual(tabs.get("api"), "API")

    def test_every_local_module_the_space_imports_is_staged(self):
        import ast

        try:
            import stage_space
        except ImportError:
            self.skipTest("stage_space.py, the Space upload tool, is not in this repository")

        here = os.path.dirname(os.path.abspath(__file__))
        for f in (f for f in stage_space.FILES if f.endswith(".py")):
            with open(os.path.join(here, f), encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            for name in names:
                if os.path.exists(os.path.join(here, f"{name}.py")):
                    with self.subTest(file=f, module=name):
                        self.assertIn(f"{name}.py", stage_space.FILES)


class TestProjectLinks(unittest.TestCase):
    """The code and the docs one click from every tab, and the API docs from the API tab."""

    GITHUB = "https://github.com/thegovind/blink"
    DOCS = "https://thegovind.github.io/blink/"
    WIRE = "https://thegovind.github.io/blink/api/"
    PAIR = [("GitHub", GITHUB, "_blank", "noopener"), ("Docs", DOCS, "_blank", "noopener")]

    def setUp(self):
        import api_doc
        import ui

        self.d, self.ui = api_doc, ui

    @staticmethod
    def links(html: str) -> list[tuple]:
        """(label, href, target, rel) for every link, in page order."""
        from html.parser import HTMLParser

        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.found, self.at, self.text = [], None, ""

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    self.at, self.text = dict(attrs), ""

            def handle_data(self, data):
                if self.at is not None:
                    self.text += data

            def handle_endtag(self, tag):
                if tag == "a" and self.at is not None:
                    a = self.at
                    self.found.append((self.text.strip(), a.get("href"), a.get("target"), a.get("rel")))
                    self.at = None

        parser = Links()
        parser.feed(html)
        parser.close()
        return parser.found

    def fold(self, html: str, summary: str) -> str:
        start = html.index(f"<summary>{summary}</summary>")
        return html[start: html.index("</details>", start)]

    def test_the_addresses_and_labels(self):
        self.assertEqual(self.ui.PROJECT_LINKS, (("GitHub", self.GITHUB), ("Docs", self.DOCS)))
        self.assertEqual((self.ui.GITHUB_URL, self.ui.DOCS_URL), (self.GITHUB, self.DOCS))
        self.assertEqual(self.d.WIRE_FORMAT_URL, self.WIRE)
        self.assertTrue(self.d.WIRE_FORMAT_URL.startswith(self.ui.DOCS_URL))
        self.assertEqual(self.d.COPY["wire_docs"], "API docs")

    def test_the_masthead_carries_them_for_every_model(self):
        """The masthead sits above the tabs and is redrawn on every move and model change, always with the pair."""
        for model in [None, *blink.models(), "someone/else"]:
            for drafting in (False, True):
                with self.subTest(model=model, drafting=drafting):
                    head = self.ui.masthead(model, drafting=drafting)
                    self.assertEqual(self.links(head), self.PAIR)
                    row = head[head.index('<div class="blk-toprow">'): head.index('<p class="blk-lede">')]
                    self.assertIn('<div class="blk-toplinks">', row)
                    self.assertLess(row.index("<h1>"), row.index("blk-toplinks"))

    def test_the_masthead_is_outside_every_tab(self):
        gr = need_gradio(self)
        self.addCleanup(reload_app)
        app = reload_app()
        demo = app.build()
        try:
            heads = [b for b in demo.blocks.values()
                     if isinstance(b, gr.HTML) and 'class="blk-toplinks"' in str(b.value or "")]
            self.assertEqual(len(heads), 1)
            node, chain = heads[0], []
            while node is not None:
                chain.append(type(node).__name__)
                node = getattr(node, "parent", None)
            self.assertEqual(chain[-1], "Blocks")
            self.assertNotIn("Tab", chain)
            self.assertNotIn("Tabs", chain)
            self.assertEqual(self.links(heads[0].value), self.PAIR)
        finally:
            demo.close()

    def test_the_api_tab_links_the_api_docs_by_the_lede(self):
        first = self.ui.api_blocks()[0]
        self.assertEqual(self.links(first), [("API docs", self.WIRE, "_blank", "noopener")])
        self.assertLess(first.index(self.ui.esc(self.d.COPY["lede"])), first.index(self.WIRE))

    def test_the_api_steps_carry_the_source_and_the_docs(self):
        steps = self.ui.api_steps()
        self.assertEqual(self.links(steps), self.PAIR)
        head = steps[: steps.index('<ol class="blk-steps">')]
        self.assertIn(self.ui.esc(self.d.COPY["steps_label"]), head)
        self.assertIn('class="blk-links"', head)
        self.assertIn(steps, "".join(self.ui.api_blocks()))

    def test_how_it_works_links_them_where_it_runs_the_model(self):
        how = "".join(self.ui.how_blocks())
        run = self.fold(how, "Running it yourself")
        self.assertEqual(self.links(run), self.PAIR)
        self.assertIn(self.ui.esc(self.ui.API_SNIPPET), run)
        self.assertLess(run.index("blk-pre"), run.index('class="blk-links"'))
        rest = how.replace(run, "")
        self.assertFalse([a for a in self.links(rest) if a[1] in (self.GITHUB, self.DOCS)])

    def test_every_new_link_opens_in_a_new_tab_under_its_own_label(self):
        page = (self.ui.masthead() + "".join(self.ui.api_blocks()) + "".join(self.ui.how_blocks())
                + "".join(self.ui.home_blocks()) + self.ui.home_external() + self.ui.footer())
        ours = [a for a in self.links(page) if a[1] in (self.GITHUB, self.DOCS, self.WIRE)]
        self.assertEqual(len(ours), 2 + 1 + 2 + 2)  # masthead, API lede, API steps, How it works
        self.assertEqual({(a[0], a[1]) for a in ours},
                         {("GitHub", self.GITHUB), ("Docs", self.DOCS), ("API docs", self.WIRE)})
        for label, _, target, rel in ours:
            with self.subTest(label=label):
                self.assertEqual(target, "_blank")
                self.assertIn("noopener", rel.split())

    def test_the_space_card_links_them(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
            card = fh.read()
        self.assertTrue(card.startswith("---\n"))
        front, body = card[4:].split("\n---\n", 1)
        line = ("**Code:** [GitHub](https://github.com/thegovind/blink) \u00b7 "
                "**Docs:** [thegovind.github.io/blink](https://thegovind.github.io/blink/)")
        self.assertEqual(body.count(line), 1)
        self.assertIn(f"\n{line}\n", body)
        self.assertNotIn("github", front.lower())
        try:
            import yaml
        except ImportError:  # pragma: no cover - PyYAML comes with gradio
            self.skipTest("PyYAML is not installed")
        meta = yaml.safe_load(front)
        self.assertEqual((meta["title"], meta["sdk"], meta["app_file"]), ("blink", "gradio", "app.py"))
        self.assertEqual(str(meta["sdk_version"]), "6.28.0")
        self.assertIn("thegovind/blink-4b", meta["models"])
        self.assertIn("thegovind/blink-mimo-9b", meta["models"])


class TestCopyRules(unittest.TestCase):
    """The standing rules for what the Space says: its own voice, its links and its card."""

    # the card's YAML front matter, byte for byte, as deployed; new README prose must leave it untouched
    FRONT_MATTER_SHA256 = "4e100ce58e0cc36ebf0ef1b491f35ff5befe01579872fd39b9a6b9858aa54606"
    FIRST_PERSON = re.compile(r"\b(?:we|our|ours|us|ourselves)\b", re.IGNORECASE)

    def setUp(self):
        import api_doc
        import results
        import ui

        self.d, self.results, self.ui = api_doc, results, ui
        self.here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.here, "README.md"), encoding="utf-8") as fh:
            self.card = fh.read()

    def pages(self) -> str:
        data = self.results.load()
        return (self.ui.masthead() + self.ui.masthead(drafting=True) + "".join(self.ui.home_blocks(data))
                + self.ui.home_external() + "".join(self.ui.card_face(t, line) for _, t, line in self.ui.HOME_LINKS)
                + "".join(self.ui.results_blocks(data)) + "".join(self.ui.how_blocks())
                + "".join(self.ui.api_blocks()) + self.ui.footer())

    def voice(self) -> dict:
        """Everything written in the Space's own voice. Sample inputs (the states a request sends) are data."""
        import html as htmllib

        def strings(x):
            if isinstance(x, str):
                yield x
            elif isinstance(x, dict):
                for k, v in x.items():
                    if k not in ("state", "kind", "id"):
                        yield from strings(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    yield from strings(v)

        data = self.results.load()
        page = (self.ui.masthead() + self.ui.masthead(drafting=True) + "".join(self.ui.home_blocks(data))
                + self.ui.home_external() + "".join(self.ui.card_face(t, line) for _, t, line in self.ui.HOME_LINKS)
                + "".join(self.ui.results_blocks(data)) + "".join(self.ui.how_blocks()) + self.ui.footer()
                + self.ui.playground_note() + self.ui.ask_note_line()
                + "".join(self.ui.use_case_note(c) for c in examples.USE_CASES))
        return {
            "the page": htmllib.unescape(re.sub(r"<[^>]+>", " ", page)),
            "api_doc.COPY": " | ".join(strings(self.d.COPY)) + " | " + " | ".join(strings(self.d.TABLE_ROWS)),
            "results.json": " | ".join(strings(data)),
            "README.md": self.card.split("\n---\n", 1)[1],
            "use cases": " | ".join(s for c in examples.USE_CASES
                                    for s in (c.title, c.blurb, c.state_label, c.policy_note or "",
                                              *(ex.label for ex in c.examples))),
        }

    def test_no_we_our_or_us_in_the_spaces_own_voice(self):
        for where, text in self.voice().items():
            with self.subTest(where=where):
                self.assertEqual(sorted({m.group(0).lower() for m in self.FIRST_PERSON.finditer(text)}), [])

    def test_no_per_model_docs_pages_are_linked(self):
        try:
            import stage_space
        except ImportError:
            self.skipTest("stage_space.py, the Space upload tool, is not in this repository")

        texts = [self.pages()]
        for name in stage_space.FILES:
            with open(os.path.join(self.here, name), encoding="utf-8") as fh:
                texts.append(fh.read())
        for text in texts:
            self.assertNotIn("github.io/blink/models", text)

    def test_how_it_works_links_each_models_card(self):
        how = "".join(self.ui.how_blocks())
        cards = {n for n, _ in self.ui.MODEL_CARDS}
        self.assertEqual(cards, {"blink-4b", "blink-27b", "blink-mimo-9b"})
        self.assertLessEqual({m.split("/")[-1] for m in blink.models()}, cards)  # every served model included
        for name, href in self.ui.MODEL_CARDS:
            with self.subTest(model=name):
                self.assertEqual(href, f"https://huggingface.co/thegovind/{name}")
                self.assertIn(f'<a href="{href}" target="_blank" rel="noopener">{name} card</a>', how)

    def test_every_outside_link_opens_in_a_new_tab(self):
        anchors = re.findall(r"<a\b[^>]*>", self.pages())
        outside = [a for a in anchors if re.search(r'href="https?://', a)]
        self.assertGreaterEqual(len(outside), 15)
        for a in outside:
            with self.subTest(link=a[:90]):
                self.assertIn('target="_blank"', a)
                self.assertIn("noopener", re.search(r'rel="([^"]*)"', a).group(1).split())

    def test_the_card_front_matter_is_byte_identical(self):
        import hashlib

        self.assertTrue(self.card.startswith("---\n"))
        front = self.card[: self.card.index("\n---\n", 4) + len("\n---\n")]
        self.assertEqual(hashlib.sha256(front.encode("utf-8")).hexdigest(), self.FRONT_MATTER_SHA256)

    def test_every_licence_line_says_non_commercial(self):
        """Wherever the weights' terms are stated they say "non-commercial": footer, card, How it works, static card."""
        import build_static

        how = "".join(self.ui.how_blocks())
        run = how[how.index("<summary>Running it yourself</summary>"):]
        lines = {
            "footer": self.ui.footer(),
            "README.md": self.card.rstrip("\n").rsplit("\n---\n", 1)[1],
            "How it works": run[: run.index("</details>")],
            "static README": build_static.space_readme(False),
        }
        for where, text in lines.items():
            with self.subTest(where=where):
                self.assertIn("non-commercial", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
