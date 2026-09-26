"""E1 serving arms on a tiny random Qwen3.5 (FP32, CPU): the batched readout (H2) and cross-request packing (H1)
must reproduce the release path.

  uv run --with torch==2.13.0 --with transformers==5.17.0 python -m unittest test_readout
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


@unittest.skipIf(torch is None, "torch/transformers not installed")
class TestServingArms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1)
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
        model = mq.Qwen3_5ForCausalLM(cfg).eval()
        rng = random.Random(11)

        class Eng:
            pass

        cls.eng = eng = Eng()
        eng.model, eng.pad_id, eng.token_budget, eng.prefix_cache = model, 0, 4096, False
        eng.key = id(eng)
        blink._LIVE[eng.key] = eng
        # three requests of different shapes: (state tokens, [(question tokens, candidate ids)])
        cls.requests = []
        for n_q in (3, 1, 5):
            state = [rng.randrange(1, 128) for _ in range(rng.randrange(20, 60))]
            qs = {}
            for j in range(n_q):
                tail = [rng.randrange(1, 128) for _ in range(rng.randrange(1, 15))]
                qs[f"q{j}"] = (state + tail, rng.sample(range(1, 128), rng.randrange(2, 9)))
            cls.requests.append(qs)

        def render(state, questions):
            return [{"qkey": k, "keys": [str(c) for c in v[1]], "ids": v[0], "cand": v[1]} for k, v in questions.items()]

        eng.render = render
        eng.logits_rendered = lambda works: blink.TorchEngine.logits_rendered(eng, works)

    def forward(self, seqs, cands, readout):
        saved = blink.READOUT
        blink.READOUT = readout
        try:
            out, _ = blink._forward(self.eng.key, seqs, cands)
        finally:
            blink.READOUT = saved
        return out

    def test_batched_readout_matches_rows(self):
        seqs = [v[0] for r in self.requests for v in r.values()]
        cands = [v[1] for r in self.requests for v in r.values()]
        for budget in (4096, 90):  # one forward, and several
            self.eng.token_budget = budget
            a, b = self.forward(seqs, cands, "rows"), self.forward(seqs, cands, "batched")
            for x, y in zip(a, b):
                self.assertEqual(len(x), len(y))
                self.assertLess(max(abs(p - q) for p, q in zip(x, y)), 1e-5)
                self.assertEqual(max(range(len(x)), key=x.__getitem__), max(range(len(y)), key=y.__getitem__))
        self.eng.token_budget = 4096

    def test_logits_many_matches_one_at_a_time(self):
        single = [blink.TorchEngine.logits(self.eng, None, r)[0] for r in self.requests]
        many = blink.TorchEngine.logits_many(self.eng, [(None, r) for r in self.requests])
        self.assertEqual(len(many), len(self.requests))
        for r, s, (m, n_tok) in zip(self.requests, single, many):
            self.assertEqual(set(m), set(r))                    # each request gets exactly its own questions back
            self.assertEqual(n_tok, sum(len(v[0]) for v in r.values()))
            for q in r:
                self.assertEqual(len(m[q]), len(r[q][1]))
                self.assertLess(max(abs(p - q2) for p, q2 in zip(s[q], m[q])), 2e-4)

    def test_logits_many_empty(self):
        self.assertEqual(blink.TorchEngine.logits_many(self.eng, []), [])


class TestDecideManyAndBatcher(unittest.TestCase):
    """Mock engine: decide_many and the Batcher must give every request exactly what decide() gives it."""

    REQS = [
        ("Invoice 7 is overdue.", {"pay": {"type": "noul", "instructions": "Should we pay now?"}}),
        ("Ticket: login fails.", {"route": {"type": "choice", "instructions": "Which team?",
                                            "criteria": {"auth": "Sign-in", "billing": "Charges"}},
                                  "urgency": {"type": "score", "instructions": "How urgent?",
                                              "criteria": ["low", "medium", "high"]}}),
        ("Short note.", {"ok": {"type": "noul", "instructions": "Is this fine?"}}),
    ]

    def test_matches_decide(self):
        many = blink.decide_many(self.REQS)
        for (state, qs), got in zip(self.REQS, many):
            want = blink.decide(state, qs)
            self.assertEqual(got["answers"], want["answers"])
            self.assertEqual(got["meta"]["input_tokens"], want["meta"]["input_tokens"])

    def test_bad_request_is_isolated(self):
        bad = ("x", {"q": {"type": "score", "instructions": "?", "criteria": ["only one"]}})
        many = blink.decide_many([self.REQS[0], bad, self.REQS[2]])
        self.assertIsInstance(many[1], blink.BlinkError)
        self.assertEqual(many[0]["answers"], blink.decide(*self.REQS[0])["answers"])
        self.assertEqual(many[2]["answers"], blink.decide(*self.REQS[2])["answers"])

    def test_batcher_returns_each_callers_own_result(self):
        import concurrent.futures
        b = blink.Batcher(window_s=0.05, max_requests=8)
        jobs = self.REQS * 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futs = [pool.submit(b.submit, s, q) for s, q in jobs]
            got = [f.result(timeout=30) for f in futs]
        for (s, q), g in zip(jobs, got):
            self.assertEqual(set(g["answers"]), set(q))
            self.assertEqual(g["answers"], blink.decide(s, q)["answers"])
        with self.assertRaises(blink.BlinkError):
            b.submit("x", {})



def deep_list(depth: int) -> list:
    """Nested lists `depth` deep: rendering them overflows the Python stack (a RecursionError, not a BlinkError)."""
    root = cur = []
    for _ in range(depth):
        cur.append([])
        cur = cur[0]
    return root


class FakeBatchEngine(blink.MockEngine):
    """The mock engine's numbers behind the torch engine's batching interface (render + logits_rendered). Its numbers
    depend only on the request, so packing requests together must not change any answer. Hooks for tests: `gate`
    (called once per packed call, e.g. to hold the worker), `fail_packed`, `explode` (a state whose rendering fails
    with a RuntimeError) and `drop` (a state whose rows the packed call leaves out)."""

    name = "torch"

    def __init__(self, model_id: str = blink.MODEL_ID, revision=None, temperature: float = blink.TEMPERATURE,
                 max_chars: int = 2000):
        super().__init__(model_id)
        self.max_chars, self.packed, self.gate = max_chars, [], None
        self.fail_packed, self.explode, self.drop = False, "__explode__", "__drop__"

    def render(self, state, questions: dict):
        if state == self.explode:
            raise RuntimeError("simulated rendering failure")
        work = []
        for qkey, q in questions.items():
            items = blink.question_options(q)
            prompt = blink.user_message(state, q, blink.LABEL_POOL[: len(items)], items)
            if len(prompt) > self.max_chars:
                raise blink.BlinkError(f"question {qkey!r} renders over the context length")
            work.append({"qkey": qkey, "state": state, "q": q})
        return work

    def logits_rendered(self, works: list):
        self.packed.append(len(works))
        if self.gate is not None:
            self.gate()
        if self.fail_packed:
            raise RuntimeError("CUDA out of memory (simulated)")
        out = []
        for work in works:
            rows, n = {}, 0
            for w in work:
                if w["state"] == self.drop:
                    continue
                r, k = blink.MockEngine.logits(self, w["state"], {w["qkey"]: w["q"]})
                rows.update(r)
                n += k
            out.append((rows, n))
        return out

    def logits_many(self, requests: list):
        return self.logits_rendered([self.render(state, questions) for state, questions in requests])

    def logits(self, state, questions: dict, bias: dict | None = None):
        self.render(state, questions)  # the same limits and failures as the packed path
        return blink.MockEngine.logits(self, state, questions)


def hold_worker(eng):
    """Make the next packed call wait: returns (entered, release) events."""
    import threading

    entered, release = threading.Event(), threading.Event()

    def gate():
        eng.gate = None
        entered.set()
        release.wait(30)

    eng.gate = gate
    return entered, release


class WithFakeEngine(unittest.TestCase):
    REQS = TestDecideManyAndBatcher.REQS
    SCHEMA_BAD = ("x", {"q": {"type": "score", "instructions": "?", "criteria": ["only one"]}})
    TOO_LONG = ("y" * 5000, {"q": {"type": "noul", "instructions": "Is this long?"}})
    DEEP = ("z", {"q": {"type": "noul", "instructions": deep_list(3000)}})

    def setUp(self):
        self.saved, self.eng = blink._ENGINE, FakeBatchEngine()
        blink._ENGINE = self.eng

    def tearDown(self):
        blink._ENGINE = self.saved

    def assert_same_as_decide(self, got, state, qs):
        want = blink.decide(state, qs)
        self.assertEqual(got["answers"], want["answers"])
        self.assertEqual(got["meta"]["input_tokens"], want["meta"]["input_tokens"])


class TestDecideManyIsolation(WithFakeEngine):
    """decide_many through the packed path (an engine with logits_rendered): every request gets exactly what decide()
    gives it, and one request's failure, of any kind and at any step, stays with that request."""

    def test_matches_decide_in_one_packed_call(self):
        many = blink.decide_many(self.REQS)
        self.assertEqual(self.eng.packed, [3])
        for (state, qs), got in zip(self.REQS, many):
            self.assert_same_as_decide(got, state, qs)
            self.assertEqual(got["meta"]["batched_requests"], 3)

    def test_mixed_requests_are_isolated(self):
        reqs = [self.REQS[0], self.SCHEMA_BAD, self.TOO_LONG, self.DEEP, self.REQS[1]]
        many = blink.decide_many(reqs)
        self.assertEqual(self.eng.packed, [2])  # only the two healthy requests reach the packed forward
        self.assert_same_as_decide(many[0], *self.REQS[0])
        self.assert_same_as_decide(many[4], *self.REQS[1])
        self.assertIsInstance(many[1], blink.BlinkError)                 # schema: HTTP 422
        self.assertIsInstance(many[2], blink.BlinkError)                 # over the context length: 422
        self.assertIsInstance(many[3], RecursionError)                   # not a BlinkError: 500, for this caller only
        with self.assertRaises(RecursionError):
            blink.decide(*self.DEEP)                                     # the release path fails it the same way

    def test_rendering_error_of_another_kind_is_isolated(self):
        many = blink.decide_many([self.REQS[0], ("__explode__", self.REQS[2][1]), self.REQS[2]])
        self.assertIsInstance(many[1], RuntimeError)
        self.assert_same_as_decide(many[0], *self.REQS[0])
        self.assert_same_as_decide(many[2], *self.REQS[2])

    def test_failed_packed_call_answers_each_request_alone(self):
        self.eng.fail_packed = True
        many = blink.decide_many(self.REQS + [self.TOO_LONG])
        self.assertEqual(self.eng.packed, [3])
        for (state, qs), got in zip(self.REQS, many):
            self.assert_same_as_decide(got, state, qs)
            self.assertEqual(got["meta"]["batched_requests"], 1)
        self.assertIsInstance(many[3], blink.BlinkError)

    def test_answer_assembly_error_is_isolated(self):
        many = blink.decide_many([self.REQS[0], ("__drop__", self.REQS[2][1]), self.REQS[1]])
        self.assertIsInstance(many[1], KeyError)
        self.assert_same_as_decide(many[0], *self.REQS[0])
        self.assert_same_as_decide(many[2], *self.REQS[1])

    def test_repeated_question_keys_across_requests(self):
        reqs = [(f"Case {i}: the parcel is {'late' if i % 2 else 'fine'}.",
                 {"q": {"type": "noul", "instructions": "Is it late?"},
                  "r": {"type": "choice", "instructions": "Who handles it?",
                        "criteria": {"ship": "Shipping", "bill": "Billing"}}}) for i in range(6)]
        many = blink.decide_many(reqs)
        self.assertEqual(self.eng.packed, [6])
        for (state, qs), got in zip(reqs, many):
            self.assert_same_as_decide(got, state, qs)


class TestBatcherLimits(WithFakeEngine):
    """The Batcher's queue is bounded, its settings are checked before any thread starts, and a worker failure
    reaches every waiting caller instead of leaving them blocked."""

    def run_all(self, b, jobs):
        import concurrent.futures

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(jobs)))
        return pool, [pool.submit(b.submit, s, q) for s, q in jobs]

    def wait_for(self, cond, timeout=10.0):
        import time

        end = time.monotonic() + timeout
        while not cond():
            if time.monotonic() > end:
                self.fail("timed out")
            time.sleep(0.005)

    def test_mixed_batch_through_the_batcher(self):
        b = blink.Batcher(window_s=0.05, max_requests=16)
        entered, release = hold_worker(self.eng)
        pool, first = self.run_all(b, [self.REQS[0]])
        self.assertTrue(entered.wait(10))
        jobs = [self.REQS[1], self.DEEP, ("__explode__", self.REQS[2][1]), self.REQS[2]]
        pool2, futs = self.run_all(b, jobs)
        self.wait_for(lambda: b.q.qsize() == len(jobs))
        with self.assertRaises(blink.BlinkError):          # invalid: rejected at once, even with the worker busy
            b.submit(*self.SCHEMA_BAD)
        release.set()
        self.assert_same_as_decide(first[0].result(timeout=30), *self.REQS[0])
        self.assert_same_as_decide(futs[0].result(timeout=30), *self.REQS[1])
        self.assertIsInstance(futs[1].exception(timeout=30), RecursionError)
        self.assertIsInstance(futs[2].exception(timeout=30), RuntimeError)
        self.assert_same_as_decide(futs[3].result(timeout=30), *self.REQS[2])
        self.assertEqual(self.eng.packed, [1, 2])        # the four queued requests were one batch; two rendered
        pool.shutdown(), pool2.shutdown()

    def test_queue_is_bounded(self):
        b = blink.Batcher(window_s=0.001, max_requests=1, max_queued=2)
        entered, release = hold_worker(self.eng)
        pool, futs = self.run_all(b, [self.REQS[0]])
        self.assertTrue(entered.wait(10))
        pool2, more = self.run_all(b, [self.REQS[1], self.REQS[2]])
        self.wait_for(lambda: b.q.qsize() == 2)
        with self.assertRaises(blink.BlinkBusy):
            b.submit(*self.REQS[2])
        release.set()
        for (state, qs), f in zip(self.REQS, futs + more):
            self.assert_same_as_decide(f.result(timeout=30), state, qs)
        pool.shutdown(), pool2.shutdown()

    def test_settings_are_checked_before_the_worker_starts(self):
        import threading

        def workers():
            return sum(t.name == "blink-batcher" for t in threading.enumerate())

        before = workers()
        for kw in ({"window_s": 0}, {"window_s": -0.005}, {"window_s": float("inf")}, {"window_s": float("nan")},
                   {"window_s": 1.5}, {"window_s": 0.005, "max_requests": 0},
                   {"window_s": 0.005, "max_requests": 65}, {"window_s": 0.005, "max_queued": 0}):
            with self.assertRaises(ValueError, msg=str(kw)):
                blink.Batcher(**kw)
        self.assertEqual(workers(), before)

    def test_worker_failure_reaches_the_callers(self):
        from unittest import mock

        b = blink.Batcher(window_s=0.001)
        with mock.patch.object(blink, "decide_many", side_effect=RuntimeError("engine gone")):
            with self.assertRaisesRegex(RuntimeError, "engine gone"):
                b.submit(*self.REQS[0])
        self.assert_same_as_decide(b.submit(*self.REQS[0]), *self.REQS[0])  # the worker keeps serving
        with mock.patch.object(blink, "decide_many", side_effect=SystemExit):  # kills the worker thread
            with self.assertRaisesRegex(RuntimeError, "lost this request"):
                b.submit(*self.REQS[0])
        self.wait_for(lambda: not b.alive())
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            b.submit(*self.REQS[0])


@unittest.skipIf(torch is None, "torch/transformers not installed")
class TestDecideManyOnTheTorchPath(unittest.TestCase):
    """decide_many against decide() through the torch engine's own render, logits and logits_rendered, on the tiny
    random model with a character-level tokenizer: several token-budget forwards, question keys repeated across
    requests, and bad requests mixed in."""

    @classmethod
    def setUpClass(cls):
        import string

        TestServingArms.setUpClass.__func__(cls)  # builds cls.eng (the tiny model) as in TestServingArms

        class TinyEngine(blink.TorchEngine):
            def __init__(self, model):
                self.model_id, self.temperature, self.token_budget = "tiny", 1.0, 1 << 20
                self.model, self.pad_id, self.prefix_cache = model, 0, False
                self.tok = lambda s, add_special_tokens=False: {"input_ids": [ord(c) % 100 + 27 for c in s]}
                self.labels, self.label_ids = list(string.ascii_uppercase), list(range(1, 27))
                self.key = id(self)
                blink._LIVE[self.key] = self

            def _wrap(self, user):
                return blink.SYSTEM + user

        cls.tiny = TinyEngine(cls.eng.model)

    def setUp(self):
        self.saved = blink._ENGINE, blink.MAX_INPUT_TOKENS
        blink._ENGINE, blink.MAX_INPUT_TOKENS = self.tiny, 1200

    def tearDown(self):
        blink._ENGINE, blink.MAX_INPUT_TOKENS = self.saved

    @staticmethod
    def probs(ans):
        return ans.get("probabilities") or {"yes": ans["noul"], "no": 1 - ans["noul"]}

    def test_matches_decide(self):
        reqs = [(f"Order {i}: {'late and damaged' if i % 2 else 'on time'} parcel.",
                 {"late": {"type": "noul", "instructions": "Is it late?"},
                  "team": {"type": "choice", "instructions": "Who handles it?",
                           "criteria": {"ship": "Shipping", "bill": "Billing", "ret": "Returns"}},
                  "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]}})
                for i in range(4)]
        bad = {1: ("x", {"q": {"type": "score", "instructions": "?", "criteria": ["one"]}}),
               3: ("w" * 2000, {"q": {"type": "noul", "instructions": "Too long?"}})}
        mixed = list(reqs)
        for i, r in bad.items():
            mixed.insert(i, r)
        for budget in (1 << 20, 900):  # one forward, and several
            self.tiny.token_budget = budget
            many = blink.decide_many(mixed)
            for i, ((state, qs), got) in enumerate(zip(mixed, many)):
                if i in bad:
                    self.assertIsInstance(got, blink.BlinkError, (budget, i))
                    continue
                want = blink.decide(state, qs)
                self.assertEqual(got["meta"]["input_tokens"], want["meta"]["input_tokens"])
                self.assertEqual(set(got["answers"]), set(qs))
                for q in qs:
                    a, w = self.probs(got["answers"][q]), self.probs(want["answers"][q])
                    self.assertEqual(max(a, key=a.get), max(w, key=w.get))
                    self.assertLess(max(abs(a[k] - w[k]) for k in w), 1e-5)


if __name__ == "__main__":
    unittest.main()
