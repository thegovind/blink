from __future__ import annotations

import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class StubRenderer:
    @staticmethod
    def tok(prompt, add_special_tokens=False):
        return {"input_ids": [1, 2]}

    def render(self, state, question, order):
        from jevlab.render import question_options

        keys = [key for key, _ in question_options(question)]
        self.last_keys = [keys[i] for i in order]
        return "prompt", self.last_keys, [65 + i for i in range(len(keys))]


class TrainingExampleTests(unittest.TestCase):
    def test_synthetic_rows_cover_every_type_and_dev_split(self):
        from jevlab.render import question_options

        path = ROOT.parent / "examples/train/sample.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        self.assertEqual({row["question"]["type"] for row in rows[:16]}, {"choice", "noul", "score"})
        self.assertEqual({row["question"]["type"] for row in rows[16:]}, {"choice", "noul", "score"})
        for row in rows:
            with self.subTest(id=row["id"]):
                self.assertEqual(row["src"], "synthetic")
                self.assertNotIn("data:image/", json.dumps(row["state"]))
                keys = {key for key, _ in question_options(row["question"])}
                if "target" in row:
                    self.assertLessEqual(set(row["target"]), keys)
                    self.assertGreater(sum(row["target"].values()), 0)
                else:
                    self.assertIn(row["gold"], keys)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
class TrainTests(unittest.TestCase):
    def test_encode_gold_soft_target_and_score_order(self):
        from jevlab import train

        rd = StubRenderer()
        question = {"type": "choice", "criteria": {"open": "Open it", "close": "Close it"}}
        hard = train.encode(rd, {"question": question, "gold": "open"}, random.Random(0), 4)
        self.assertEqual(dict(zip(rd.last_keys, hard["t"])), {"open": 1.0, "close": 0.0})

        soft = train.encode(rd, {"question": question, "target": {"open": 3, "close": 1},
                                 "weight": 2.0, "src": "synthetic"}, random.Random(0), 4)
        self.assertEqual(sorted(soft["t"]), [0.25, 0.75])
        self.assertEqual(soft["w"], 2.0)
        self.assertEqual(soft["src"], "synthetic")

        yes_no = train.encode(rd, {"question": {"type": "noul"}, "gold": "no"}, random.Random(1), 4)
        self.assertEqual(dict(zip(rd.last_keys, yes_no["t"])), {"yes": 0.0, "no": 1.0})

        score = {"type": "score", "criteria": ["low", "medium", "high"]}
        levels = train.encode(rd, {"question": score, "gold": "2"}, random.Random(1), 4)
        self.assertEqual(levels["t"], [0.0, 0.0, 1.0])
        self.assertEqual(levels["cand"], [65, 66, 67])

    def test_invalid_targets_and_gold_fail_even_when_prompt_is_long(self):
        from jevlab import train

        q = {"type": "choice", "criteria": {"open": "Open it", "close": "Close it"}}
        cases = [
            ({"gold": "missing"}, "gold must be an offered"),
            ({"target": {}}, "target must be finite"),
            ({"target": {"open": 1, "missing": 1}}, "target must use offered"),
            ({"target": {"open": -1, "close": 2}}, "target must be finite"),
            ({"target": {"open": float("nan")}}, "target must be finite"),
            ({"gold": "open", "weight": float("inf")}, "weight must be finite"),
        ]
        for supervision, error in cases:
            with self.subTest(supervision=supervision), self.assertRaisesRegex(ValueError, error):
                train.encode(StubRenderer(), {"id": "bad", "question": q, **supervision}, random.Random(0), 1)
        self.assertIsNone(train.encode(StubRenderer(), {"question": q, "gold": "open"}, random.Random(0), 1))

    def test_cpu_restricted_label_loss_and_gradients(self):
        import torch
        from jevlab import train

        torch.manual_seed(7)

        class Body(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(8, 4)

            def forward(self, input_ids, use_cache):
                return SimpleNamespace(last_hidden_state=self.emb(input_ids))

        body = Body()
        head = torch.randn(8, 4)
        examples = [
            {"ids": [1, 2], "cand": [5, 6], "t": [1.0, 0.0]},
            {"ids": [3], "cand": [5, 6], "t": [0.25, 0.75]},
        ]
        losses, logps = train.forward_batch(body, head, examples, 0, torch.device("cpu"))
        expected = torch.stack([
            -(torch.tensor(example["t"]) * torch.log_softmax(head[example["cand"]] @
                                                              body.emb.weight[example["ids"][-1]], dim=-1)).sum()
            for example in examples
        ])
        torch.testing.assert_close(losses, expected)
        self.assertEqual(len(logps), 2)
        losses.sum().backward()
        self.assertTrue(torch.isfinite(body.emb.weight.grad).all())
        self.assertGreater(float(body.emb.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
