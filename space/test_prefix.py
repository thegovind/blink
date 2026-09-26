"""Shared-prefix prefill: logic tests, plus an exact parity test on a tiny random Qwen3.5 (DeltaNet + full attention).

  python3 -m unittest test_prefix                                     # logic tests (torch tests skip without torch)
  uv run --with torch==2.13.0 --with transformers==5.17.0 python -m unittest test_prefix
"""
import os
import random
import unittest

os.environ.setdefault("BLINK_MOCK", "1")

import blink  # noqa: E402

try:
    import torch
    from transformers.models.qwen3_5 import configuration_qwen3_5 as cq
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
except Exception:  # noqa: BLE001
    torch = None


class TestSharedPrefixLen(unittest.TestCase):
    def test_cases(self):
        f = blink._shared_prefix_len
        self.assertEqual(f([[1, 2, 3]]), 0)                              # one question: nothing to share
        self.assertEqual(f([[1, 2, 3, 4], [1, 2, 9, 9], [1, 2, 3]]), 2)
        self.assertEqual(f([[5, 6], [7, 8]]), 0)
        self.assertEqual(f([[1, 2, 3], [1, 2, 3]]), 2)                   # identical: each keeps its last token
        self.assertEqual(f([[1, 2, 3, 4, 5], [1, 2, 3]]), 2)             # shorter one keeps one token of its own

    def test_matches_brute_force(self):
        rng = random.Random(7)
        for _ in range(200):
            base = [rng.randrange(5) for _ in range(rng.randrange(1, 12))]
            seqs = [base[: rng.randrange(0, len(base) + 1)] + [rng.randrange(5) for _ in range(rng.randrange(1, 6))]
                    for _ in range(rng.randrange(2, 6))]
            p = 0
            while all(len(s) > p for s in seqs) and len({s[p] for s in seqs}) == 1:
                p += 1
            self.assertEqual(blink._shared_prefix_len(seqs), min(p, min(len(s) for s in seqs) - 1))


class TestBatchesMaxRows(unittest.TestCase):
    def test_cap_and_default(self):
        self.assertEqual([len(b) for b in blink._batches([10] * 7, 1000, max_rows=3)], [3, 3, 1])
        self.assertEqual([len(b) for b in blink._batches([10] * 7, 1000)], [7])
        self.assertEqual(sorted(i for b in blink._batches([5, 50, 7, 30], 60) for i in b), [0, 1, 2, 3])


class TestPrefillTokens(unittest.TestCase):
    def test_counts(self):
        prefix = list(range(1000))
        seqs = [prefix + [tok] * n for tok, n in ((7, 10), (8, 20), (9, 30))]  # tails differ from their first token
        self.assertEqual(blink._prefill_tokens(seqs, prefix_cache=False), 3000 + 60)
        self.assertEqual(blink._prefill_tokens(seqs, prefix_cache=True), 1000 + 60)   # prefix once, tails each
        short = [[1, 2, 3, 9], [1, 2, 3, 8]]                                           # under PREFIX_MIN_TOKENS
        self.assertEqual(blink._prefill_tokens(short, prefix_cache=True), 8)
        self.assertEqual(blink._prefill_tokens([prefix], prefix_cache=True), 1000)     # one question


@unittest.skipIf(torch is None, "torch/transformers not installed")
class TestTinyQwen35Parity(unittest.TestCase):
    """Cached-prefix logits must equal full-prompt logits (FP32, tiny random model, CPU reference kernels)."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cfg = cq.Qwen3_5TextConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            tie_word_embeddings=False, max_position_embeddings=4096,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                             "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        )
        cls.model = mq.Qwen3_5ForCausalLM(cfg).eval()

        class Eng:
            pass

        cls.eng = Eng()
        cls.eng.model, cls.eng.pad_id, cls.eng.token_budget = cls.model, 0, 4096
        cls.key = id(cls.eng)
        blink._LIVE[cls.key] = cls.eng
        rng = random.Random(3)
        prefix = [rng.randrange(1, 128) for _ in range(90)]
        cls.seqs = [prefix + [rng.randrange(1, 128) for _ in range(n)] for n in (5, 17, 9, 23, 1, 12)]
        cls.cands = [[rng.randrange(1, 128) for _ in range(k)] for k in (2, 3, 4, 5, 2, 7)]

    def run_forward(self, prefix_cache, min_tokens=blink.PREFIX_MIN_TOKENS, kv_tokens=blink.PREFIX_KV_TOKENS, budget=4096):
        saved = (blink.PREFIX_MIN_TOKENS, blink.PREFIX_KV_TOKENS)
        blink.PREFIX_MIN_TOKENS, blink.PREFIX_KV_TOKENS = min_tokens, kv_tokens
        self.eng.prefix_cache, self.eng.token_budget = prefix_cache, budget
        try:
            out, _ = blink._forward(self.key, self.seqs, self.cands)
        finally:
            blink.PREFIX_MIN_TOKENS, blink.PREFIX_KV_TOKENS = saved
        return out

    def assert_close(self, a, b, tol=2e-4):
        for x, y in zip(a, b):
            self.assertEqual(len(x), len(y))
            self.assertLess(max(abs(p - q) for p, q in zip(x, y)), tol)

    def test_shared_prefix_matches_full_prompts(self):
        full = self.run_forward(prefix_cache=False)
        shared = self.run_forward(prefix_cache=True, min_tokens=8)
        self.assert_close(full, shared)

    def test_several_tail_batches(self):
        full = self.run_forward(prefix_cache=False)
        # at most 2 tails per batch (kv cap) and a small budget: several expanded copies of the prefix cache
        shared = self.run_forward(prefix_cache=True, min_tokens=8, kv_tokens=2 * (90 + 23), budget=40)
        self.assert_close(full, shared)

    def test_full_path_is_batch_invariant(self):
        # the reference itself: one-question-per-forward vs batched full prompts agree
        self.assert_close(self.run_forward(prefix_cache=False, budget=4096), self.run_forward(prefix_cache=False, budget=120))

    def test_threshold_keeps_plain_path(self):
        self.assertEqual(self.run_forward(prefix_cache=True, min_tokens=10_000), self.run_forward(prefix_cache=False))

    def test_expand_cache_copies(self):
        prefix = torch.tensor([self.seqs[0][:90]])
        with torch.no_grad():
            cache = self.model.model(input_ids=prefix, use_cache=True).past_key_values
        before = [t.clone() for layer in cache.layers for t in _tensors(layer)]
        big = blink._expand_cache(cache, 3)
        for layer in big.layers:
            for t in _tensors(layer):
                self.assertEqual(t.shape[0], 3)
        after = [t for layer in cache.layers for t in _tensors(layer)]
        self.assertTrue(all(torch.equal(a, b) and a.shape[0] == 1 for a, b in zip(before, after)))


def _tensors(layer):
    out = []
    for name in ("keys", "values"):
        t = getattr(layer, name, None)
        if torch.is_tensor(t) and t.numel():
            out.append(t)
    for name in ("conv_states", "recurrent_states"):
        held = getattr(layer, name, None)
        vals = held.values() if isinstance(held, dict) else (held or [])
        out += [v for v in vals if torch.is_tensor(v) and v.numel()]
    return out


if __name__ == "__main__":
    unittest.main()
