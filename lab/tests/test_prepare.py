"""examples/train/prepare.py: row checks, the votes/outcome shortcuts, the group-disjoint split and the leak refusal."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("blink_prepare", REPO / "examples" / "train" / "prepare.py")
prepare = importlib.util.module_from_spec(SPEC)
sys.modules["blink_prepare"] = prepare
SPEC.loader.exec_module(prepare)

CHOICE = {"type": "choice", "instructions": "Which remedy?",
          "criteria": {"refund": "Return payment", "replace": "Send a replacement"}}
NOUL = {"type": "noul", "instructions": "Reply today?"}
SCORE = {"type": "score", "instructions": "Impact?", "criteria": ["None", "Partial", "Complete"]}


def row(**kw):
    base = {"state": "A fictional parcel arrived damaged.", "question": CHOICE, "gold": "refund"}
    base.update(kw)
    return base


class TestNormalize(unittest.TestCase):
    def test_votes_become_a_normalised_target(self):
        out = prepare.normalize({"state": "s", "question": CHOICE, "votes": {"refund": 3, "replace": 1}})
        self.assertEqual(out["target"], {"refund": 0.75, "replace": 0.25})
        self.assertNotIn("votes", out)
        self.assertNotIn("gold", out)
        self.assertEqual(out["meta"]["label_from"], "votes")

    def test_outcome_becomes_gold(self):
        out = prepare.normalize({"state": "s", "question": CHOICE, "outcome": "replace"})
        self.assertEqual(out["gold"], "replace")
        self.assertEqual(out["meta"]["label_from"], "outcome")

    def test_yes_no_and_score_labels_in_the_trainers_form(self):
        self.assertEqual(prepare.normalize({"state": "s", "question": NOUL, "gold": True})["gold"], "yes")
        self.assertEqual(prepare.normalize({"state": "s", "question": NOUL, "outcome": False})["gold"], "no")
        self.assertEqual(prepare.normalize({"state": "s", "question": NOUL, "gold": "Yes"})["gold"], "yes")
        self.assertEqual(prepare.normalize({"state": "s", "question": SCORE, "gold": 2})["gold"], "2")

    def test_target_wins_over_gold_like_the_trainer(self):
        out = prepare.normalize(row(target={"replace": 2, "refund": 0}))
        self.assertEqual(out["target"], {"replace": 1.0})
        self.assertNotIn("gold", out)

    def test_bad_rows_are_refused_by_name(self):
        cases = [
            (row(id="x1", gold="exchange"), "x1"),
            (row(id="x2", target={"exchange": 1}), "offered"),
            (row(id="x3", target={"refund": -1, "replace": 2}), "nonnegative"),
            (row(id="x4", weight=-0.5), "weight"),
            ({"id": "x5", "state": "s", "question": CHOICE, "votes": {"refund": 1}, "gold": "refund"}, "not both"),
            ({"id": "x6", "state": "s", "questions": {"q": CHOICE}, "gold": "refund"}, "one 'question'"),
            ({"id": "x7", "question": CHOICE, "gold": "refund"}, "state"),
            ({"id": "x8", "state": "s", "question": {"type": "score", "criteria": ["only one"]}, "gold": "0"}, "2-10"),
        ]
        for bad, needle in cases:
            with self.subTest(bad=bad["id"]):
                with self.assertRaises(ValueError) as ctx:
                    prepare.normalize(bad, "rows.jsonl:1")
                self.assertIn(needle, str(ctx.exception))

    def test_ids_are_stable_when_missing(self):
        a = prepare.normalize(row())
        b = prepare.normalize(row())
        self.assertEqual(a["id"], b["id"])
        self.assertTrue(a["id"].startswith("row-"))


class TestSplit(unittest.TestCase):
    def rows(self):
        out = []
        for g in range(20):
            for k in range(3):
                out.append(prepare.normalize(row(id=f"g{g}-{k}", state=f"state {g}-{k}", group=f"g{g}")))
        return out

    def test_groups_never_straddle_and_the_split_is_repeatable(self):
        train, dev = prepare.split(self.rows(), 0.2, seed=7)
        self.assertFalse({r["group"] for r in train} & {r["group"] for r in dev})
        self.assertEqual(len(train) + len(dev), 60)
        self.assertGreaterEqual(len(dev), 12)
        self.assertLessEqual(len(dev), 15)
        again = prepare.split(self.rows(), 0.2, seed=7)[1]
        self.assertEqual([r["id"] for r in dev], [r["id"] for r in again])

    def test_identical_states_stay_together_without_groups(self):
        rows = [prepare.normalize(row(id=f"r{i}", state=f"state {i // 2}")) for i in range(20)]
        train, dev = prepare.split(rows, 0.3)
        self.assertFalse({r["state"] for r in train} & {r["state"] for r in dev})

    def test_one_group_cannot_be_split(self):
        rows = [prepare.normalize(row(id=f"r{i}", group="same")) for i in range(5)]
        with self.assertRaises(ValueError):
            prepare.split(rows, 0.2)


class TestCommandLine(unittest.TestCase):
    def write(self, path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_fraction_split_writes_both_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.write(tmp / "rows.jsonl", [row(id=f"r{i}", state=f"s{i}") for i in range(30)]
                       + [{"state": f"v{i}", "question": NOUL, "votes": {"yes": 2, "no": 1}} for i in range(10)])
            self.assertEqual(prepare.main([str(tmp / "rows.jsonl"), "--out-dir", str(tmp / "out"),
                                           "--dev-fraction", "0.25"]), 0)
            train = [json.loads(x) for x in (tmp / "out" / "train.jsonl").read_text().splitlines()]
            dev = [json.loads(x) for x in (tmp / "out" / "dev.jsonl").read_text().splitlines()]
            self.assertEqual(len(train) + len(dev), 40)
            self.assertTrue(all("votes" not in r for r in train + dev))

    def test_a_dev_file_that_repeats_training_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.write(tmp / "train.jsonl", [row(id=f"t{i}", state=f"s{i}") for i in range(5)])
            self.write(tmp / "dev.jsonl", [row(id="d0", state="s3", gold="replace")])
            self.assertEqual(prepare.main([str(tmp / "train.jsonl"), "--dev", str(tmp / "dev.jsonl"),
                                           "--out-dir", str(tmp / "out")]), 1)
            self.assertFalse((tmp / "out").exists())

    def test_bad_json_names_the_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "rows.jsonl").write_text(json.dumps(row()) + "\n{not json\n")
            with self.assertRaises(ValueError) as ctx:
                prepare.load(str(tmp / "rows.jsonl"))
            self.assertIn("rows.jsonl:2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
