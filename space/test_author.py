"""Tests for author.py (free-form asks → one blink request). Runs without a model: BLINK_MOCK=1 python3 -m unittest test_author"""
import importlib
import json
import os
import unittest

os.environ.setdefault("BLINK_MOCK", "1")

import author  # noqa: E402
import blink  # noqa: E402


class TestFirstJson(unittest.TestCase):
    def test_plain_and_prefixed(self):
        self.assertEqual(author.first_json('{"a": 1}'), {"a": 1})
        self.assertEqual(author.first_json('Sure: {"a": 1} trailing'), {"a": 1})

    def test_repairs_missing_closers_only(self):
        self.assertEqual(author.first_json('{"a": {"b": [1, 2'), {"a": {"b": [1, 2]}})
        raw = ('{"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", '
               '"criteria": {"x": "X", "y": "Y"}}}')  # one brace short, as seen on ZeroGPU
        self.assertEqual(author.first_json(raw)["questions"]["q"]["criteria"], {"x": "X", "y": "Y"})

    def test_braces_inside_strings_are_ignored(self):
        self.assertEqual(author.first_json('{"a": "has } brace", "b": {"c": 1}'), {"a": "has } brace", "b": {"c": 1}})

    def test_unrepairable(self):
        self.assertIsNone(author.first_json("no json"))
        self.assertIsNone(author.first_json('{"a": "ends inside a string'))
        self.assertIsNone(author.first_json('{"a": 1,, }'))


class TestNormalise(unittest.TestCase):
    def q(self, **kw):
        return {"state": "s", "questions": {"q": {"instructions": "Which?", **kw}}}

    def test_choice_dict_list_slugs_dedup_unquote(self):
        out = author.normalise(self.q(type="choice", criteria=["1", "2", "3", "3"]), "ask")
        self.assertEqual(out["questions"]["q"]["criteria"], {"1": "1", "2": "2", "3": "3"})
        out = author.normalise(self.q(type="choice", criteria={"Pay Invoice!": "'Pay it'", "b": '"Ignore"'}), "ask")
        self.assertEqual(out["questions"]["q"]["criteria"], {"pay_invoice": "Pay it", "b": "Ignore"})

    def test_colliding_keys_never_merge_options(self):
        out = author.normalise(self.q(type="choice", criteria={"a": "first", "a!": "second", "a_2": "third"}), "ask")
        crit = out["questions"]["q"]["criteria"]
        self.assertEqual(list(crit.values()), ["first", "second", "third"])
        self.assertEqual(len(set(crit)), 3)
        names = author.normalise({"state": "s", "questions": {
            "Q": {"type": "noul", "instructions": "x"}, "q": {"type": "noul", "instructions": "y"},
            "q_2": {"type": "noul", "instructions": "z"}}}, "ask")["questions"]
        self.assertEqual(len(names), 3)
        self.assertEqual([v["instructions"] for v in names.values()], ["x", "y", "z"])

    def test_noul_and_aliases(self):
        out = author.normalise(self.q(type="yes/no"), "ask")
        self.assertEqual(out["questions"]["q"], {"type": "noul", "instructions": "Which?"})
        out = author.normalise(self.q(type="noul", criteria={"yes": "Y", "no": "N"}), "ask")
        self.assertEqual(out["questions"]["q"]["criteria"], {"true": "Y", "false": "N"})

    def test_score_levels(self):
        out = author.normalise(self.q(type="score", criteria=["low", "mid", "high"]), "ask")
        self.assertEqual(out["questions"]["q"]["criteria"], ["low", "mid", "high"])
        with self.assertRaises(author.AuthorError):
            author.normalise(self.q(type="score", criteria=["only one"]), "ask")
        with self.assertRaises(author.AuthorError):
            author.normalise(self.q(type="score", criteria=[str(i) for i in range(11)]), "ask")

    def test_empty_state_falls_back_to_ask_and_result_is_valid(self):
        out = author.normalise({"state": " ", "questions": {"Letter Count": {"type": "choice", "instructions": "How many?",
                                                                          "criteria": ["1", "2", "3"]}}}, "How many r?")
        self.assertEqual(out["state"], "How many r?")
        self.assertIn("letter_count", out["questions"])
        blink.validate(out["questions"])

    def test_rejections(self):
        bad = [
            None, {}, {"questions": {}}, {"error": "not a decision"},
            self.q(type="choice", criteria=["only"]),
            self.q(type="choice", criteria=[str(i) for i in range(author.MAX_DRAFT_OPTIONS + 1)]),
            self.q(type="choice"), self.q(type="ranking", criteria=["a", "b"]),
            {"state": "s", "questions": {"q": {"type": "noul", "instructions": " "}}},
            {"state": "s", "questions": {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(4)}},
        ]
        for obj in bad:
            with self.subTest(obj=obj), self.assertRaises(author.AuthorError):
                author.normalise(obj, "ask")

    def test_not_a_decision_message(self):
        with self.assertRaises(author.AuthorError) as cm:
            author.normalise({"error": "not a decision"}, "write a poem")
        self.assertIn("few possible answers", str(cm.exception))


class TestAsk(unittest.TestCase):
    def test_limits(self):
        for ask in ("", "   ", None):
            with self.subTest(ask=ask), self.assertRaises(author.AuthorError):
                author.check_ask(ask)
        with self.assertRaises(author.AuthorError):
            author.check_ask("x" * (author.MAX_ASK_CHARS + 1))
        self.assertEqual(author.check_ask("  hi  "), "hi")


class TestMockDraft(unittest.TestCase):
    def test_strawberry(self):
        d = author.draft("How many r in strawberry")
        self.assertEqual(d["state"], "strawberry")
        (q,) = d["questions"].values()
        self.assertEqual((q["type"], list(q["criteria"])), ("choice", ["1", "2", "3", "4"]))
        self.assertEqual(d["author"]["model"], "mock")
        blink.validate(d["questions"])

    def test_types(self):
        cases = {"Is 91 a prime number?": "noul", "How angry is this customer?": "score",
                 "Which is healthier: oatmeal or a donut?": "choice"}
        for ask, qtype in cases.items():
            with self.subTest(ask=ask):
                (q,) = author.draft(ask)["questions"].values()
                self.assertEqual(q["type"], qtype)

    def test_refuses_non_decisions_with_raw(self):
        with self.assertRaises(author.AuthorError) as cm:
            author.draft("write me a poem about cats")
        self.assertIn("not a decision", cm.exception.raw)

    def test_draft_runs_through_blink(self):
        d = author.draft("How many r in strawberry")
        out = blink.decide(d["state"], d["questions"])
        (a,) = out["answers"].values()
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0, places=6)


class TestConfig(unittest.TestCase):
    def test_prompt_shots_are_valid_requests(self):
        for _, answer in author.SHOTS:
            if "error" in answer:
                continue
            blink.validate(answer["questions"])
            json.dumps(answer)
        self.assertEqual(author.messages("hi")[-1], {"role": "user", "content": "hi"})

    def test_enabled_switch_and_revision_pin(self):
        env = {k: os.environ.get(k) for k in ("BLINK_AUTHOR", "BLINK_MOCK", "BLINK_AUTHOR_REVISION")}
        try:
            os.environ.pop("BLINK_AUTHOR_REVISION", None)
            os.environ["BLINK_MOCK"] = "0"
            os.environ["BLINK_AUTHOR"] = "off"
            self.assertFalse(importlib.reload(author).enabled())
            os.environ.pop("BLINK_AUTHOR")
            mod = importlib.reload(author)
            self.assertTrue(mod.enabled())
            self.assertEqual(mod.AUTHOR_MODEL, "Qwen/Qwen3.5-4B")
            self.assertEqual(mod.AUTHOR_REVISION, "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
            os.environ["BLINK_AUTHOR"] = "Qwen/Qwen3.5-2B"
            self.assertIsNone(importlib.reload(author).AUTHOR_REVISION)
        finally:
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(author)


if __name__ == "__main__":
    unittest.main()
