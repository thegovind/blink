"""examples/rl/reward_demo.py: an accuracy reward makes the policy overconfident; proper scores keep it calibrated."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("blink_reward_demo", REPO / "examples" / "rl" / "reward_demo.py")
demo = importlib.util.module_from_spec(SPEC)
sys.modules["blink_reward_demo"] = demo
SPEC.loader.exec_module(demo)


class TestRewardDemo(unittest.TestCase):
    def test_accuracy_reward_is_overconfident_and_proper_scores_are_not(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                r = demo.run(situations=60, steps=12000, seed=seed)
                grpo, log, brier = r["accuracy reward (GRPO)"], r["log score"], r["Brier score"]
                self.assertGreater(grpo["confidence"] - grpo["hit_rate"], 0.15)
                self.assertGreater(grpo["calibration_error"], 0.15)
                self.assertLess(log["calibration_error"], 0.08)
                self.assertLess(brier["calibration_error"], 0.10)
                self.assertLess(log["expected_log_loss"], grpo["expected_log_loss"])
                self.assertLess(grpo["entropy"], log["entropy"])

    def test_training_longer_makes_the_accuracy_reward_more_certain_not_more_right(self):
        r = demo.longer(situations=60, steps=8000, seed=0)
        conf = [r[("accuracy reward (GRPO)", n)]["confidence"] for n in (2000, 8000, 32000)]
        gap = [r[("accuracy reward (GRPO)", n)]["confidence"] - r[("accuracy reward (GRPO)", n)]["hit_rate"]
               for n in (2000, 8000, 32000)]
        self.assertLess(conf[0], conf[1])
        self.assertLess(conf[1], conf[2])
        self.assertLess(gap[0], gap[2])
        self.assertLess(r[("log score", 32000)]["calibration_error"], 0.05)

    def test_gradients_match_finite_differences(self):
        import math
        import random

        rng = random.Random(3)
        z = [0.3, -0.2, 0.9, 0.1]
        y = 2

        def log_score(zz):
            return math.log(demo.softmax(zz)[y])

        def brier(zz):
            p = demo.softmax(zz)
            return -0.5 * sum((pk - (1.0 if k == y else 0.0)) ** 2 for k, pk in enumerate(p))

        for fn, grad in ((log_score, demo.grad_log_score), (brier, demo.grad_brier)):
            g = grad(demo.softmax(z), y, rng)
            for j in range(len(z)):
                zp = list(z)
                zm = list(z)
                zp[j] += 1e-6
                zm[j] -= 1e-6
                self.assertAlmostEqual(g[j], (fn(zp) - fn(zm)) / 2e-6, places=5)


if __name__ == "__main__":
    unittest.main()
