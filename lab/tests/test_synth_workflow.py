from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO / "space") not in sys.path:
    sys.path.insert(0, str(REPO / "space"))

import blink  # noqa: E402
from jevlab import synth_workflow as sw  # noqa: E402


def rows_by_family():
    out = {}
    seed = 1000
    while set(out) != set(sw.FAMILIES):
        rows = sw.rows_for_seed(seed)
        out.setdefault(rows[0]["meta"]["family"], rows)
        seed += 1
    return out


class TestSynthWorkflow(unittest.TestCase):
    def test_each_family_recomputes_from_rendered_fields(self):
        for family, rows in rows_by_family().items():
            self.assertGreaterEqual(len(rows), 5, family)
            self.assertLessEqual(len(rows), 30, family)
            for row in rows:
                fields = sw.parse_state_fields(row["state"])
                self.assertTrue(fields, row["id"])
                gold, target = sw.recompute_row(row)
                self.assertEqual(gold, row["gold"], row["id"])
                if target is None:
                    self.assertIsNone(row["target"], row["id"])
                else:
                    self.assertIsNotNone(row["target"], row["id"])
                    self.assertAlmostEqual(sum(row["target"].values()), 1.0, places=7)
                    self.assertEqual(set(target), set(row["target"]))
                    for key, value in target.items():
                        self.assertAlmostEqual(value, row["target"][key], places=7)

    def test_cant_tell_only_when_required_field_absent(self):
        rows = sw.generate_rows(range(1000, 1060))
        seen = 0
        for row in rows:
            q = row["question"]
            if q["type"] != "choice" or "cant_tell" not in q["criteria"]:
                continue
            seen += 1
            fields = sw.parse_state_fields(row["state"])
            required = row["meta"]["spec"].get("required", [])
            self.assertTrue(required, row["id"])
            self.assertTrue(any(k not in fields for k in required), row["id"])
            self.assertEqual(row["gold"], "cant_tell", row["id"])
        self.assertGreater(seen, 0)

    def test_question_options_are_unique_and_validate(self):
        for row in sw.generate_rows(range(1000, 1020)):
            q = row["question"]
            if q["type"] == "choice":
                keys = list(q["criteria"])
            elif q["type"] == "noul":
                keys = ["yes", "no"]
            else:
                keys = [str(i) for i in range(len(q["criteria"]))]
            self.assertEqual(len(keys), len(set(keys)), row["id"])
            self.assertIn(row["gold"], keys, row["id"])
            blink.validate({"q": q})

    def test_generation_is_deterministic_per_seed(self):
        first = sw.rows_for_seed(1234)
        second = sw.rows_for_seed(1234)
        self.assertEqual(
            json.dumps(first, sort_keys=True, ensure_ascii=False),
            json.dumps(second, sort_keys=True, ensure_ascii=False),
        )
        self.assertNotEqual(first[0]["id"], sw.rows_for_seed(1235)[0]["id"])

    def test_split_seed_documents_are_disjoint(self):
        train = sw.generate_rows(range(1000, 1010))
        dev = sw.generate_rows(range(9000, 9010))
        self.assertTrue({r["meta"]["seed"] for r in train}.isdisjoint({r["meta"]["seed"] for r in dev}))
        self.assertTrue({r["meta"]["group"] for r in train}.isdisjoint({r["meta"]["group"] for r in dev}))


if __name__ == "__main__":
    unittest.main()
