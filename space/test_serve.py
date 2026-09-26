"""serve.py over real HTTP with a stand-in engine (no GPU, no weights): every request gets its own status, answers
and usage with batching off (the release path) and on; an overloaded batching queue answers 529; bad settings stop
the server before anything loads. The wire tests hold serve.py to the TypeSafe API's format (answers, usage, model
list, Bearer key, error bodies, request id) and to v1.1: every v1.1 response comes back byte for byte, apart from
the fields added since (confidence on score answers, detail on error bodies). TestOfficialSDK runs TypeSafe's own
Python SDK against the server when it's installed (uv run --with typesafe-sdk ...).

  cd release && BLINK_MOCK=1 python -m unittest test_serve
"""
import concurrent.futures
import contextlib
import hashlib
import http.client
import importlib.util
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
# research tree: release/serve.py beside space/; public tree: space/test_serve.py with serve.py at the root
# serve.py's folder goes on the path first, then blink.py's in front of it, so `import blink` finds space/blink.py
# even where the public tree's root also holds a `blink` package
for d in (HERE, os.path.join(HERE, "..")):
    if os.path.exists(os.path.join(d, "serve.py")):
        sys.path.insert(0, d)
        break
for d in (os.path.join(HERE, "..", "space"), HERE):
    if os.path.exists(os.path.join(d, "blink.py")):
        sys.path.insert(0, d)
        break
os.environ.setdefault("BLINK_MOCK", "1")

import blink  # noqa: E402
import serve  # noqa: E402
import test_readout  # noqa: E402  (a module import, so its tests don't run again here)

FakeBatchEngine, hold_worker = test_readout.FakeBatchEngine, test_readout.hold_worker

REQS = test_readout.TestDecideManyAndBatcher.REQS
SCHEMA_BAD = ("x", {"q": {"type": "score", "instructions": "?", "criteria": ["only one"]}})
EXPLODE = ("__explode__", REQS[2][1])  # its rendering fails with a RuntimeError: HTTP 500, for this request only
SAVED_ENGINE = None


def setUpModule():
    global SAVED_ENGINE
    SAVED_ENGINE = blink._ENGINE


def tearDownModule():
    blink._ENGINE = SAVED_ENGINE


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port: int, state, questions):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        c.request("POST", "/v1/systemone", json.dumps({"state": state, "questions": questions}),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"{}"), dict(r.getheaders())
    finally:
        c.close()


def health(port: int) -> dict:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        c.request("GET", "/healthz")
        return json.loads(c.getresponse().read())
    finally:
        c.close()


def wait_for(cond, timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise AssertionError("timed out")
        time.sleep(0.01)


class Server:
    """serve.main() in a daemon thread, with FakeBatchEngine (or `engine`) in place of the torch engine. The newest
    server's engine is blink._ENGINE, which is the one blink.decide() compares against."""

    def __init__(self, *argv: str, engine=None, env: dict | None = None):
        self.port = free_port()
        args = ["serve.py", "--model", tempfile.mkdtemp(), "--port", str(self.port), *argv]
        with mock.patch.object(blink, "TorchEngine", engine or FakeBatchEngine), mock.patch.object(sys, "argv", args), \
                mock.patch.dict(os.environ, env or {}), contextlib.redirect_stdout(io.StringIO()):
            threading.Thread(target=serve.main, daemon=True).start()
            end = time.monotonic() + 30
            while True:
                try:
                    self.health = health(self.port)
                    break
                except OSError:
                    if time.monotonic() > end:
                        raise
                    time.sleep(0.02)
        self.engine = blink._ENGINE
        self.model = self.health["model"]


class TestServe(unittest.TestCase):
    def check(self, jobs, got, want_codes):
        self.assertEqual([code for code, _, _ in got], want_codes)
        for (state, qs), (code, body, _) in zip(jobs, got):
            if code != 200:
                self.assertIn("error", body)
                continue
            want = blink.decide(state, qs)
            self.assertEqual(body["answers"], want["answers"])
            self.assertEqual(body["usage"], {"input_tokens": want["meta"]["input_tokens"], "output_tokens": 0})

    def test_release_path(self):
        srv = Server()
        self.assertIsNone(srv.health["batching"])
        self.assertEqual(set(srv.health), {"ok", "model", "revision", "weights_verified", "hub_offline", "warmup",
                                           "kernels", "versions", "batching", "api_key_required"})
        self.assertIs(srv.health["api_key_required"], False)
        jobs = [REQS[0], SCHEMA_BAD, EXPLODE, REQS[1], REQS[2]]
        with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
            got = list(pool.map(lambda j: post(srv.port, *j), jobs))
        self.check(jobs, got, [200, 422, 500, 200, 200])
        self.assertEqual(srv.engine.packed, [])  # one request at a time; never packed

    def test_batching_keeps_each_request_to_itself(self):
        srv = Server("--batch-window-ms", "20")
        b = srv.health["batching"]
        self.assertEqual((b["window_ms"], b["max_requests"], b["max_queued"], b["worker_alive"], b["queued"]),
                         (20.0, 16, 64, True, 0))
        entered, release = hold_worker(srv.engine)  # the next requests queue up and are decided as one batch
        jobs = [REQS[1], EXPLODE, REQS[2]]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            first = pool.submit(post, srv.port, *REQS[0])
            self.assertTrue(entered.wait(10))
            futs = [pool.submit(post, srv.port, *j) for j in jobs]
            wait_for(lambda: health(srv.port)["batching"]["queued"] == len(jobs))
            code, body, _ = post(srv.port, *SCHEMA_BAD)  # invalid: answered at once, even with the worker busy
            self.assertEqual(code, 422)
            release.set()
            got = [first.result(30)] + [f.result(30) for f in futs]
        self.check([REQS[0]] + jobs, got, [200, 200, 500, 200])
        self.assertEqual(srv.engine.packed, [1, 2])  # the three queued requests were one batch; two rendered

    def test_overload_answers_529(self):
        """TypeSafe's "overloaded" status, with Retry-After (its SDKs retry any 5xx and honour the header)."""
        srv = Server("--batch-window-ms", "1", "--max-batch-requests", "1", "--max-queued-requests", "1")
        entered, release = hold_worker(srv.engine)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            first = pool.submit(post, srv.port, *REQS[0])
            self.assertTrue(entered.wait(10))
            second = pool.submit(post, srv.port, *REQS[1])
            wait_for(lambda: health(srv.port)["batching"]["queued"] == 1)
            code, body, headers = post(srv.port, *REQS[2])
            self.assertEqual(code, 529)
            self.assertEqual(headers.get("Retry-After"), "1")
            self.assertIn("error", body)
            self.assertEqual(body["detail"], body["error"])
            release.set()
            self.check([REQS[0], REQS[1]], [first.result(30), second.result(30)], [200, 200])

    def test_bad_settings_stop_before_loading(self):
        class Loaded(Exception):
            pass

        def load(*args, **kwargs):
            raise Loaded

        root = tempfile.mkdtemp()
        bad = (["--batch-window-ms", "inf"], ["--batch-window-ms", "nan"], ["--batch-window-ms", "-1"],
               ["--batch-window-ms", "1001"], ["--batch-window-ms", "5", "--max-batch-requests", "0"],
               ["--max-batch-requests", "65"], ["--max-queued-requests", "0"],
               ["--api-key", "two words"], ["--api-key", "caf\u00e9"], ["--api-key", "tab\there"])
        with mock.patch.object(blink, "TorchEngine", load), mock.patch.dict(os.environ), \
                contextlib.redirect_stderr(io.StringIO()):
            for argv in bad:
                with self.subTest(argv=argv), mock.patch.object(sys, "argv", ["serve.py", "--model", root, *argv]):
                    with self.assertRaises(SystemExit):
                        serve.main()
            os.environ["BLINK_BATCH_WINDOW_MS"] = "fast"
            with mock.patch.object(sys, "argv", ["serve.py", "--model", root, "--batch-window-ms", "0"]):
                with self.assertRaises(Loaded):  # an explicit 0 wins over a bad environment value
                    serve.main()
            with mock.patch.object(sys, "argv", ["serve.py", "--model", root]):
                with self.assertRaises(SystemExit):
                    serve.main()
            del os.environ["BLINK_BATCH_WINDOW_MS"]
            os.environ["BLINK_API_KEY"] = "not a key"
            with mock.patch.object(sys, "argv", ["serve.py", "--model", root]):
                with self.assertRaises(SystemExit):  # the environment's key is checked like the flag's
                    serve.main()
            with mock.patch.object(sys, "argv", ["serve.py", "--model", root, "--api-key", "ok-key"]):
                with self.assertRaises(Loaded):  # an explicit key wins over a bad environment value
                    serve.main()


# --- the wire format: TypeSafe's API, and v1.1 byte for byte ---------------------------------------------------------


class PathFreeEngine(FakeBatchEngine):
    """FakeBatchEngine with numbers that don't depend on the served path (the mock salts its numbers by model id), so
    a response can be compared with one recorded in another run. Every wire test uses it, so whichever server's
    engine blink.decide() last picked up, the numbers are the same."""

    def __init__(self, *args, **kwargs):
        super().__init__(blink.MODEL_ID)


def call(port: int, method: str, path: str, body=None, headers: dict | None = None):
    """One raw request on a fresh connection: (status, lower-cased headers, body text)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        c.request(method, path, body.encode("utf-8") if isinstance(body, str) else body,
                  {"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read().decode("utf-8")
    finally:
        c.close()


def J(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


# The API reference's examples (docs.typesafe.ai/api), its quickstart request, and what JevBench's typesafe adapter and
# the Decision Index kit's http engine send. These build exactly the request bodies V11 was recorded with.
DOC_NOUL = {"state": "Help! My payouts have been failing for 3 days.", "model": "jev-latest",
            "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                                        "criteria": {"true": "Explicitly time-sensitive",
                                                     "false": "No urgency expressed"}}}}
DOC_CHOICE = {"state": "Help! My payouts have been failing for 3 days.", "model": "jev-latest",
              "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                                           "criteria": {"billing": "Payments, invoicing, refunds",
                                                        "technical": "Bugs, outages, integrations",
                                                        "sales": "Pricing, upgrades, new accounts"}}}}
DOC_SCORE = {"state": "Help! My payouts have been failing for 3 days.", "model": "jev-latest",
             "questions": {"frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                                           "criteria": ["Calm", "Frustrated", "Very angry"]}}}
QUICKSTART = {
    "state": "Hi, I've been trying to connect my payment account for 3 days and the integration keeps failing. "
             "I'm losing sales. Please help ASAP.",
    "model": "jev-latest",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this",
                       "criteria": {"billing": "Payment or subscription issues",
                                    "technical": "Bugs or integration problems",
                                    "sales": "Pricing or account questions"}},
        "frustration": {"type": "score", "instructions": "How frustrated the customer appears",
                        "criteria": ["Calm, just stating facts", "Frustrated but civil",
                                     "Very angry, strong language"]},
        "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"},
    },
}
STRUCTURED = {
    "state": {"messages": [{"from": "customer", "text": "I was charged twice for order A-104."},
                           {"from": "agent", "text": "Sorry about that, checking now."}],
              "order": {"id": "A-104", "total": 42.5}},
    "model": "jev-1.13.0",
    "questions": {
        "same_person": {"type": "noul",
                        "instructions": {"record": {"name": "Ada", "city": "Oakland"},
                                         "question": "Is `messages[0]` from the person in `record`?"}},
        "tone": {"type": "choice", "instructions": "What is the customer's tone?",
                 "criteria": {"calm": None, "frustrated": None, "angry": None}},
        "severity": {"type": "score", "instructions": ["How severe is the issue?", "Consider money lost."],
                     "criteria": ["Cosmetic", {"level": "Money at risk", "examples": ["double charge"]},
                                  "Account locked"]},
        "orders": {"type": "choice", "instructions": "Which department?",
                   "criteria": {"orders": {"what": "Order status or returns", "not_for": "Payments"},
                                "billing": ["Charges", "Refunds"]}},
    },
}
JEVBENCH = {"state": "The package arrived two weeks late and the box was crushed.", "model": "jev-latest",
            "questions": {"decision": {"type": "choice", "instructions": "What is the main complaint?",
                                       "criteria": {"late": "Late delivery", "damaged": "Damaged item",
                                                    "wrong": "Wrong item"}}}}
DIKIT = {"model": "default", "state": {"question": "2 + 2 = ?", "answer": "4"},
         "questions": {"q0": {"type": "noul", "instructions": "Is the answer correct?"},
                       "q1": {"type": "score", "instructions": "How confident should a grader be?",
                              "criteria": ["not at all", "somewhat", "very", "certain"]}}}
UNICODE = {"state": "Caf\u00e9 \u2615 \u2014 \u8acb\u6c42\u304c\u4e8c\u91cd\u3067\u3059\u3002 \u00dcn\u00efc\u00f6d\u00e9 \U0001f680",
           "model": "jev-latest",
           "questions": {"lang": {"type": "choice", "instructions": "Which language dominates?",
                                  "criteria": {"ja": "\u65e5\u672c\u8a9e", "fr": "Fran\u00e7ais", "en": "English"}}}}
NO_MODEL = {"state": "Invoice 7 is overdue.", "questions": {"pay": {"type": "noul", "instructions": "Pay now?"}}}
NO_STATE = {"model": "jev-latest", "questions": {"pay": {"type": "noul", "instructions": "Pay now?"}}}
NOUL_YES_NO = {"state": "Refund please", "model": "jev-latest",
               "questions": {"refund": {"type": "noul", "instructions": "Refund request?",
                                        "criteria": {"yes": "Asks for money back", "no": "Anything else"}}}}
CHOICE_LIST = {"state": "Bonjour", "questions": {"lang": {"type": "choice", "instructions": "Language?",
                                                          "criteria": ["english", "french"]}}}
TEN_LEVELS = {"state": "ok", "questions": {"s": {"type": "score", "instructions": "Rate",
                                                 "criteria": [f"level {i}" for i in range(10)]}}}
MANY_OPTIONS = {"state": "pick", "questions": {"c": {"type": "choice", "instructions": "Which?",
                                                     "criteria": {f"o{i}": None for i in range(255)}}}}

WIRE = [
    ("doc_noul", "POST", "/v1/systemone", J(DOC_NOUL)),
    ("doc_choice", "POST", "/v1/systemone", J(DOC_CHOICE)),
    ("doc_score", "POST", "/v1/systemone", J(DOC_SCORE)),
    ("quickstart", "POST", "/v1/systemone", J(QUICKSTART)),
    ("structured", "POST", "/v1/systemone", J(STRUCTURED)),
    ("jevbench", "POST", "/v1/systemone", J(JEVBENCH)),
    ("dikit", "POST", "/v1/systemone", J(DIKIT)),
    ("unicode", "POST", "/v1/systemone", J(UNICODE)),
    ("no_model", "POST", "/v1/systemone", J(NO_MODEL)),
    ("no_state", "POST", "/v1/systemone", J(NO_STATE)),
    ("noul_yes_no", "POST", "/v1/systemone", J(NOUL_YES_NO)),
    ("choice_list", "POST", "/v1/systemone", J(CHOICE_LIST)),
    ("ten_levels", "POST", "/v1/systemone", J(TEN_LEVELS)),
    ("many_options", "POST", "/v1/systemone", J(MANY_OPTIONS)),
    ("trailing_slash", "POST", "/v1/systemone/", J(NO_MODEL)),
    ("bad_json", "POST", "/v1/systemone", '{"state": "x", "questions": '),
    ("not_object", "POST", "/v1/systemone", '["x"]'),
    ("no_questions", "POST", "/v1/systemone", J({"state": "x", "model": "jev-latest"})),
    ("empty_questions", "POST", "/v1/systemone", J({"state": "x", "questions": {}})),
    ("question_not_object", "POST", "/v1/systemone", J({"state": "x", "questions": {"q": "Is it?"}})),
    ("bad_type", "POST", "/v1/systemone", J({"state": "x", "questions": {"q": {"type": "rank",
                                                                             "instructions": "?"}}})),
    ("one_level", "POST", "/v1/systemone", J({"state": "x", "questions": {"q": {"type": "score", "instructions": "?",
                                                                              "criteria": ["only one"]}}})),
    ("eleven_levels", "POST", "/v1/systemone", J({"state": "x", "questions": {"q": {
        "type": "score", "instructions": "?", "criteria": [str(i) for i in range(11)]}}})),
    ("choice_no_criteria", "POST", "/v1/systemone", J({"state": "x", "questions": {"q": {
        "type": "choice", "instructions": "?"}}})),
    ("too_many_options", "POST", "/v1/systemone", J({"state": "x", "questions": {"q": {
        "type": "choice", "instructions": "?", "criteria": {f"o{i}": None for i in range(256)}}}})),
    ("too_many_questions", "POST", "/v1/systemone", J({"state": "x", "questions": {
        f"q{i}": {"type": "noul", "instructions": "?"} for i in range(513)}})),
    ("too_long", "POST", "/v1/systemone", J({"state": "y" * 5000, "questions": {"q": {"type": "noul",
                                                                                     "instructions": "Long?"}}})),
    ("explode", "POST", "/v1/systemone", J({"state": "__explode__", "questions": {"q": {"type": "noul",
                                                                                       "instructions": "?"}}})),
    ("post_unknown_path", "POST", "/v1/other", J(NO_MODEL)),
    ("get_unknown_path", "GET", "/nope", None),
]
WIRE_BY_NAME = {name: (method, path, body) for name, method, path, body in WIRE}

# Recorded from v1.1 (serve.py f578fea0, blink.py da7a4b73, as released) with PathFreeEngine, the served path
# written as {MODEL}: name -> (sha256 of the request body, first 16 hex; status; response body).
V11 = {
    'doc_noul': ('082269c4d2e35650', 200, '{"model": "{MODEL}", "answers": {"is_urgent": {"type": "noul", "noul": 0.4930999634050107, "probabilities": {"yes": 0.4930999634050107, "no": 0.5069000365949893}}}, "usage": {"input_tokens": 101, "output_tokens": 0}}'),
    'doc_choice': ('887f9803fcedd9f9', 200, '{"model": "{MODEL}", "answers": {"department": {"type": "choice", "choice": "technical", "probabilities": {"billing": 0.19769939764680702, "technical": 0.6461568465584113, "sales": 0.15614375579478165}, "confidence": 0.469235269837617}}, "usage": {"input_tokens": 126, "output_tokens": 0}}'),
    'doc_score': ('5a60a7984f8c7e58', 200, '{"model": "{MODEL}", "answers": {"frustration": {"type": "score", "score": 0.9681059827522747, "probabilities": {"0": 0.14333382730613417, "1": 0.7452263626354569, "2": 0.11143981005840889}, "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"}, "choice": "1"}}, "usage": {"input_tokens": 111, "output_tokens": 0}}'),
    'quickstart': ('04c824f97034c0dd', 200, '{"model": "{MODEL}", "answers": {"department": {"type": "choice", "choice": "sales", "probabilities": {"billing": 0.3189249486537832, "technical": 0.29545841393309735, "sales": 0.3856166374131195}, "confidence": 0.0784249561196793}, "frustration": {"type": "score", "score": 0.7596752809359448, "probabilities": {"0": 0.3954196616516803, "1": 0.4494853957606944, "2": 0.15509494258762518}, "legend": {"0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language"}, "choice": "1"}, "is_urgent": {"type": "noul", "noul": 0.4793237857761251, "probabilities": {"yes": 0.4793237857761251, "no": 0.5206762142238749}}}, "usage": {"input_tokens": 408, "output_tokens": 0}}'),
    'structured': ('da5ee66fe2754b00', 200, '{"model": "{MODEL}", "answers": {"same_person": {"type": "noul", "noul": 0.35423407618100855, "probabilities": {"yes": 0.35423407618100855, "no": 0.6457659238189914}}, "tone": {"type": "choice", "choice": "frustrated", "probabilities": {"calm": 0.31815070128630896, "frustrated": 0.5549699592383517, "angry": 0.12687933947533936}, "confidence": 0.3324549388575275}, "severity": {"type": "score", "score": 1.162019094267827, "probabilities": {"0": 0.24712689261294601, "1": 0.34372712050628107, "2": 0.40914598688077297}, "legend": {"0": "Cosmetic", "1": "{\\n  \\"level\\": \\"Money at risk\\",\\n  \\"examples\\": [\\n    \\"double charge\\"\\n  ]\\n}", "2": "Account locked"}, "choice": "2"}, "orders": {"type": "choice", "choice": "orders", "probabilities": {"orders": 0.9034090649096072, "billing": 0.09659093509039289}, "confidence": 0.8068181298192143}}, "usage": {"input_tokens": 616, "output_tokens": 0}}'),
    'jevbench': ('ecfd3b1cf902f9fd', 200, '{"model": "{MODEL}", "answers": {"decision": {"type": "choice", "choice": "late", "probabilities": {"late": 0.807635505460649, "damaged": 0.05965758066236418, "wrong": 0.13270691387698688}, "confidence": 0.7114532581909735}}, "usage": {"input_tokens": 114, "output_tokens": 0}}'),
    'dikit': ('6217cc70e9b4d61b', 200, '{"model": "{MODEL}", "answers": {"q0": {"type": "noul", "noul": 0.2503796704682207, "probabilities": {"yes": 0.2503796704682207, "no": 0.7496203295317793}}, "q1": {"type": "score", "score": 1.3842511140761045, "probabilities": {"0": 0.3068232648866341, "1": 0.2073084636408323, "2": 0.2806621639823284, "3": 0.20520610749020515}, "legend": {"0": "not at all", "1": "somewhat", "2": "very", "3": "certain"}, "choice": "0"}}, "usage": {"input_tokens": 208, "output_tokens": 0}}'),
    'unicode': ('08b2ce6a1ffcb2d1', 200, '{"model": "{MODEL}", "answers": {"lang": {"type": "choice", "choice": "fr", "probabilities": {"ja": 0.32417081931053016, "fr": 0.39765027458977253, "en": 0.27817890609969737}, "confidence": 0.09647541188465882}}, "usage": {"input_tokens": 96, "output_tokens": 0}}'),
    'no_model': ('a7c0c9c0b6c535c0', 200, '{"model": "{MODEL}", "answers": {"pay": {"type": "noul", "noul": 0.3947939798759009, "probabilities": {"yes": 0.3947939798759009, "no": 0.6052060201240991}}}, "usage": {"input_tokens": 78, "output_tokens": 0}}'),
    'no_state': ('264e53df20d1eb2c', 200, '{"model": "{MODEL}", "answers": {"pay": {"type": "noul", "noul": 0.5223238723821921, "probabilities": {"yes": 0.5223238723821921, "no": 0.4776761276178079}}}, "usage": {"input_tokens": 73, "output_tokens": 0}}'),
    'noul_yes_no': ('f2fc597815eb867a', 200, '{"model": "{MODEL}", "answers": {"refund": {"type": "noul", "noul": 0.5226508439264278, "probabilities": {"yes": 0.5226508439264278, "no": 0.47734915607357226}}}, "usage": {"input_tokens": 87, "output_tokens": 0}}'),
    'choice_list': ('a82ba0c447ec028a', 200, '{"model": "{MODEL}", "answers": {"lang": {"type": "choice", "choice": "french", "probabilities": {"english": 0.25335323338669563, "french": 0.7466467666133043}, "confidence": 0.4932935332266086}}, "usage": {"input_tokens": 77, "output_tokens": 0}}'),
    'ten_levels': ('1d86e78ee35715c8', 200, '{"model": "{MODEL}", "answers": {"s": {"type": "score", "score": 3.957160196556809, "probabilities": {"0": 0.05772047530178102, "1": 0.09943126303262649, "2": 0.06355722389602497, "3": 0.23992457760640531, "4": 0.18867173165896653, "5": 0.14068073424157002, "6": 0.059690049542021846, "7": 0.054755192808686085, "8": 0.0487952590509935, "9": 0.04677349286092429}, "legend": {"0": "level 0", "1": "level 1", "2": "level 2", "3": "level 3", "4": "level 4", "5": "level 5", "6": "level 6", "7": "level 7", "8": "level 8", "9": "level 9"}, "choice": "3"}}, "usage": {"input_tokens": 183, "output_tokens": 0}}'),
    'many_options': ('f4671a0757a368f5', 422, '{"error": "question \'c\' renders over the context length"}'),
    'trailing_slash': ('a7c0c9c0b6c535c0', 200, '{"model": "{MODEL}", "answers": {"pay": {"type": "noul", "noul": 0.3947939798759009, "probabilities": {"yes": 0.3947939798759009, "no": 0.6052060201240991}}}, "usage": {"input_tokens": 78, "output_tokens": 0}}'),
    'bad_json': ('8cefef9f11a9e565', 400, '{"error": "invalid JSON: Expecting value: line 1 column 29 (char 28)"}'),
    'not_object': ('cd65ea2c2ad99e94', 400, '{"error": "the body must be a JSON object"}'),
    'no_questions': ('4c8ff925ada7b9a3', 422, '{"error": "questions must be a non-empty object"}'),
    'empty_questions': ('4e8918111232970a', 422, '{"error": "questions must be a non-empty object"}'),
    'question_not_object': ('d4f02dd1670945b2', 422, '{"error": "question \'q\' must be an object"}'),
    'bad_type': ('efd885edc3f5aa77', 422, '{"error": "unsupported question type \'rank\'"}'),
    'one_level': ('edd37cb4ac837bb4', 422, '{"error": "a score takes 2 to 10 levels"}'),
    'eleven_levels': ('a8b422bda343fbf0', 422, '{"error": "a score takes 2 to 10 levels"}'),
    'choice_no_criteria': ('625d8cd71c3e2d30', 422, '{"error": "choice needs criteria"}'),
    'too_many_options': ('8f0997551aacbf08', 422, '{"error": "256 options per choice; supported 1-255"}'),
    'too_many_questions': ('aea81be16e6f4c8f', 422, '{"error": "513 questions; supported 1-512"}'),
    'too_long': ('f3212cfcc17c3b3c', 422, '{"error": "question \'q\' renders over the context length"}'),
    'explode': ('5d1d23469bd53191', 500, '{"error": "RuntimeError: simulated rendering failure"}'),
    'post_unknown_path': ('a7c0c9c0b6c535c0', 404, '{"error": "not found"}'),
    'get_unknown_path': ('e3b0c44298fc1c14', 404, '{"error": "not found"}'),
}


# Python 3.12 made float sum() compensated, which moves some last digits; these were recorded the same way on 3.11.
V11_BEFORE_3_12 = {
    'doc_choice': ('887f9803fcedd9f9', 200, '{"model": "{MODEL}", "answers": {"department": {"type": "choice", "choice": "technical", "probabilities": {"billing": 0.197699397646807, "technical": 0.6461568465584113, "sales": 0.15614375579478162}, "confidence": 0.469235269837617}}, "usage": {"input_tokens": 126, "output_tokens": 0}}'),
    'quickstart': ('04c824f97034c0dd', 200, '{"model": "{MODEL}", "answers": {"department": {"type": "choice", "choice": "sales", "probabilities": {"billing": 0.3189249486537832, "technical": 0.29545841393309735, "sales": 0.3856166374131195}, "confidence": 0.0784249561196793}, "frustration": {"type": "score", "score": 0.7596752809359449, "probabilities": {"0": 0.3954196616516804, "1": 0.4494853957606945, "2": 0.1550949425876252}, "legend": {"0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language"}, "choice": "1"}, "is_urgent": {"type": "noul", "noul": 0.4793237857761251, "probabilities": {"yes": 0.4793237857761251, "no": 0.5206762142238749}}}, "usage": {"input_tokens": 408, "output_tokens": 0}}'),
    'jevbench': ('ecfd3b1cf902f9fd', 200, '{"model": "{MODEL}", "answers": {"decision": {"type": "choice", "choice": "late", "probabilities": {"late": 0.8076355054606491, "damaged": 0.05965758066236418, "wrong": 0.13270691387698685}, "confidence": 0.7114532581909736}}, "usage": {"input_tokens": 114, "output_tokens": 0}}'),
    'dikit': ('6217cc70e9b4d61b', 200, '{"model": "{MODEL}", "answers": {"q0": {"type": "noul", "noul": 0.2503796704682207, "probabilities": {"yes": 0.2503796704682207, "no": 0.7496203295317793}}, "q1": {"type": "score", "score": 1.384251114076105, "probabilities": {"0": 0.30682326488663414, "1": 0.20730846364083233, "2": 0.28066216398232846, "3": 0.20520610749020518}, "legend": {"0": "not at all", "1": "somewhat", "2": "very", "3": "certain"}, "choice": "0"}}, "usage": {"input_tokens": 208, "output_tokens": 0}}'),
    'unicode': ('08b2ce6a1ffcb2d1', 200, '{"model": "{MODEL}", "answers": {"lang": {"type": "choice", "choice": "fr", "probabilities": {"ja": 0.32417081931053016, "fr": 0.39765027458977253, "en": 0.2781789060996974}, "confidence": 0.09647541188465882}}, "usage": {"input_tokens": 96, "output_tokens": 0}}'),
    'ten_levels': ('1d86e78ee35715c8', 200, '{"model": "{MODEL}", "answers": {"s": {"type": "score", "score": 3.9571601965568086, "probabilities": {"0": 0.05772047530178102, "1": 0.09943126303262649, "2": 0.06355722389602497, "3": 0.23992457760640531, "4": 0.18867173165896653, "5": 0.14068073424157002, "6": 0.059690049542021846, "7": 0.054755192808686085, "8": 0.0487952590509935, "9": 0.04677349286092429}, "legend": {"0": "level 0", "1": "level 1", "2": "level 2", "3": "level 3", "4": "level 4", "5": "level 5", "6": "level 6", "7": "level 7", "8": "level 8", "9": "level 9"}, "choice": "3"}}, "usage": {"input_tokens": 183, "output_tokens": 0}}'),
}
GOLDEN = {**V11, **V11_BEFORE_3_12} if sys.version_info < (3, 12) else V11
# recorded on Linux (glibc); elsewhere exp() may round differently in the last digit, so floats are compared
# to 1e-12 there and everything else exactly
EXACT_FLOATS = sys.platform.startswith("linux")


def strip_added(body: dict) -> dict:
    """A response without the fields added since v1.1 (detail on errors, confidence on score answers), with the
    rest in the order v1.1 wrote it."""
    out = {k: v for k, v in body.items() if k != "detail"}
    if isinstance(out.get("answers"), dict):
        out["answers"] = {q: {f: v for f, v in a.items() if not (f == "confidence" and a.get("type") == "score")}
                          for q, a in out["answers"].items()}
    return out


def jevbench_parse(qtype: str, labels, parsed: dict) -> dict:
    """What JevBench's stock typesafe adapter reads from a 200 (jevbench/adapters/typesafe.py): exact-label probs."""
    ans = parsed["answers"]["decision"]
    assert ans["type"] == qtype, "native answer type mismatch"
    if qtype == "noul":
        p = ans["noul"]
        assert not isinstance(p, bool) and isinstance(p, (int, float)) and 0.0 <= p <= 1.0
        return {"yes": p, "no": 1.0 - p}
    if qtype == "choice":
        assert ans.get("choice") in labels
    assert isinstance(ans.get("probabilities"), dict)
    return ans["probabilities"]


def dikit_validate(questions: dict, response: dict) -> None:
    """The Decision Index kit's response check (decision_index/engines/base.py validate) for choice and noul."""
    import math

    assert set(response.get("answers", {})) == set(questions), "Question keys mismatch"
    for k, q in questions.items():
        a = response["answers"][k]
        assert a.get("type") == q["type"], "Type mismatch"
        if q["type"] == "choice":
            assert a.get("choice") in q["criteria"]
            p = a.get("probabilities", {})
            assert set(p) == set(q["criteria"]) and all(math.isfinite(v) and 0 <= v <= 1 for v in p.values())
            assert abs(sum(p.values()) - 1) <= 0.01
        elif q["type"] == "noul":
            assert math.isfinite(a["noul"]) and 0 <= a["noul"] <= 1


# the Decision Index kit counts a 400/413/422 whose body names one of these as "unsupported", not as an error
DI_CAPACITY_MARKERS = ("options per choice", "a score takes 2 to 10 levels", "maximum context length")


class TestWire(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = Server(engine=PathFreeEngine)

    def post(self, body, headers=None, srv=None, path="/v1/systemone"):
        code, head, text = call((srv or self.srv).port, "POST", path, J(body) if isinstance(body, dict) else body,
                                headers)
        return code, head, json.loads(text) if text else None

    def assert_typesafe(self, questions: dict, body: dict) -> None:
        """TypeSafe's response schema (openapi.json SystemOneResponse), with the strict types its Python SDK
        checks; plus the documented meaning of each number."""
        self.assertEqual(list(body)[:3], ["model", "answers", "usage"])
        self.assertIsInstance(body["model"], str)
        self.assertEqual(set(body["usage"]), {"input_tokens", "output_tokens"})
        self.assertIs(type(body["usage"]["input_tokens"]), int)
        self.assertEqual(body["usage"]["output_tokens"], 0)
        self.assertEqual(list(body["answers"]), list(questions))
        for key, q in questions.items():
            a = body["answers"][key]
            self.assertEqual(a["type"], q["type"])
            if q["type"] == "noul":
                self.assertIs(type(a["noul"]), float)
                self.assertTrue(0.0 <= a["noul"] <= 1.0)
                self.assertEqual(a["probabilities"], {"yes": a["noul"], "no": a["probabilities"]["no"]})
                continue
            crit = q["criteria"]
            keys = list(crit) if q["type"] == "choice" else [str(i) for i in range(len(crit))]
            probs = a["probabilities"]
            self.assertEqual(list(probs), keys)
            self.assertTrue(all(type(p) is float and 0.0 <= p <= 1.0 for p in probs.values()))
            self.assertAlmostEqual(sum(probs.values()), 1.0)
            n, top = len(keys), max(probs.values())
            self.assertIs(type(a["confidence"]), float)
            self.assertAlmostEqual(a["confidence"], (n * top - 1) / (n - 1))  # docs.typesafe.ai/confidence
            if q["type"] == "choice":
                self.assertEqual(probs[a["choice"]], top)
            else:
                self.assertTrue(0.0 <= a["confidence"] <= 1.0)
                self.assertAlmostEqual(a["score"], sum(i * probs[k] for i, k in enumerate(keys)))
                self.assertEqual(a["legend"], {k: blink.text(crit[i]) for i, k in enumerate(keys)})
                self.assertEqual(probs[a["choice"]], top)

    def assert_same_json(self, got_text: str, want_text: str) -> None:
        """Byte for byte; off Linux, floats to 1e-12 (see EXACT_FLOATS) with keys, order and the rest exact."""
        if EXACT_FLOATS:
            return self.assertEqual(got_text, want_text)

        def same(g, w):
            if isinstance(w, float) and type(g) is float:
                return abs(g - w) <= 1e-12 * max(1.0, abs(w))
            if isinstance(w, dict):
                return isinstance(g, dict) and list(g) == list(w) and all(same(g[k], w[k]) for k in w)
            if isinstance(w, list):
                return isinstance(g, list) and len(g) == len(w) and all(same(a, b) for a, b in zip(g, w, strict=True))
            return type(g) is type(w) and g == w

        self.assertTrue(same(json.loads(got_text), json.loads(want_text)), f"{got_text}\n!=\n{want_text}")

    def test_v11_requests_come_back_byte_for_byte(self):
        """Every recorded v1.1 request gets v1.1's status and body, byte for byte, once the added fields are taken
        out; nothing else is added. One request at a time and with batching on."""
        for srv in (self.srv, Server("--batch-window-ms", "5", engine=PathFreeEngine)):
            for name, method, path, body in WIRE:
                sha, status, want = GOLDEN[name]
                with self.subTest(name=name, batching=srv.health["batching"] is not None):
                    self.assertEqual(hashlib.sha256((body or "").encode("utf-8")).hexdigest()[:16], sha)
                    code, head, text = call(srv.port, method, path, body, {"Authorization": "Bearer any-key"})
                    self.assertEqual(code, status)
                    self.assertEqual(head["content-type"], "application/json")
                    got, old = json.loads(text), json.loads(want)
                    self.assert_same_json(J(strip_added(got)), want.replace("{MODEL}", srv.model))
                    if status == 200:
                        self.assertEqual(list(got), list(old))
                        for q, a in got["answers"].items():
                            added = set(a) - set(old["answers"][q])
                            self.assertEqual(added, {"confidence"} if a["type"] == "score" else set())
                    else:
                        self.assertEqual(list(got), ["error", "detail"])

    def test_answers_have_typesafes_shape(self):
        for req in (DOC_NOUL, DOC_CHOICE, DOC_SCORE, QUICKSTART, STRUCTURED, DIKIT, UNICODE, TEN_LEVELS):
            with self.subTest(questions=list(req["questions"])):
                code, head, body = self.post(req)
                self.assertEqual(code, 200)
                self.assert_typesafe(req["questions"], body)
                self.assertEqual(body["model"], self.srv.model)

    def test_model_field_is_accepted_and_not_used(self):
        want = self.post(DOC_CHOICE)[2]
        for model in ("jev-latest", "jev-preview", "jev-1.13.0", "~typesafe/jev-latest", "", None):
            with self.subTest(model=model):
                req = {k: v for k, v in DOC_CHOICE.items() if k != "model"}
                if model is not None:
                    req["model"] = model
                code, head, body = self.post(req)
                self.assertEqual(code, 200)
                self.assertEqual(body, want)  # the served model answers and is named, whatever was asked for

    def test_request_id_on_every_response(self):
        seen = []
        for method, path, body in (("POST", "/v1/systemone", J(DOC_NOUL)), ("POST", "/v1/systemone", "{"),
                                   ("POST", "/v1/systemone", J({"state": "x", "questions": {}})),
                                   ("POST", "/v1/systemone", WIRE_BY_NAME["explode"][2]),
                                   ("POST", "/v1/nope", "{}"), ("GET", "/nope", None), ("GET", "/v1/models", None),
                                   ("GET", "/healthz", None)):
            with self.subTest(method=method, path=path):
                code, head, text = call(self.srv.port, method, path, body)
                rid = head.get("x-typesafe-request-id", "")
                self.assertRegex(rid, r"^[0-9a-f]{32}$")
                seen.append(rid)
        self.assertEqual(len(set(seen)), len(seen))

    def test_models_list(self):
        for path in ("/v1/models", "/v1/models/"):
            with self.subTest(path=path):
                code, head, text = call(self.srv.port, "GET", path)
                self.assertEqual(code, 200)
                body = json.loads(text)
                self.assertEqual(list(body), ["models"])
                (entry,) = body["models"]
                self.assertEqual(list(entry), ["name", "description", "release_date"])
                self.assertTrue(all(isinstance(v, str) for v in entry.values()))
                self.assertEqual(entry["name"], self.srv.model)  # the name every answer reports

    def test_open_by_default(self):
        for headers in ({}, {"Authorization": "Bearer any-key"}, {"Authorization": "Basic dXNlcjpwdw=="},
                        {"Authorization": "Bearer"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.post(DOC_NOUL, headers)[0], 200)
                self.assertEqual(call(self.srv.port, "GET", "/v1/models", headers=headers)[0], 200)

    def assert_key_required(self, srv, key: str) -> None:
        self.assertIs(srv.health["api_key_required"], True)
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": f"Basic {key}"},
                        {"Authorization": key}, {"Authorization": f"Bearer {key}x"}, {"Authorization": "Bearer "}):
            with self.subTest(headers=headers):
                code, head, body = self.post(DOC_NOUL, headers, srv=srv)
                self.assertEqual(code, 401)
                self.assertEqual(head.get("www-authenticate"), "Bearer")
                self.assertEqual(head.get("connection"), "close")  # the unread body isn't left on the connection
                self.assertEqual(body["detail"], body["error"])
                self.assertIn("API key", body["error"])
                code, head, text = call(srv.port, "GET", "/v1/models", headers=headers)
                self.assertEqual(code, 401)
        for headers in ({"Authorization": f"Bearer {key}"}, {"Authorization": f"bearer  {key} "}):
            with self.subTest(headers=headers):
                code, head, body = self.post(DOC_NOUL, headers, srv=srv)
                self.assertEqual(code, 200)
                self.assert_typesafe(DOC_NOUL["questions"], body)
                self.assertEqual(call(srv.port, "GET", "/v1/models", headers=headers)[0], 200)
        self.assertEqual(call(srv.port, "GET", "/healthz")[0], 200)  # health checks need no key

    def test_api_key_flag(self):
        self.assert_key_required(Server("--api-key", "s3cret-key.1", engine=PathFreeEngine), "s3cret-key.1")

    def test_api_key_from_environment(self):
        self.assert_key_required(Server(engine=PathFreeEngine, env={"BLINK_API_KEY": "env-key"}), "env-key")
        srv = Server("--api-key", "flag-key", engine=PathFreeEngine, env={"BLINK_API_KEY": "env-key"})
        self.assertEqual(self.post(DOC_NOUL, {"Authorization": "Bearer flag-key"}, srv=srv)[0], 200)
        self.assertEqual(self.post(DOC_NOUL, {"Authorization": "Bearer env-key"}, srv=srv)[0], 401)
        srv = Server(engine=PathFreeEngine, env={"BLINK_API_KEY": "  "})
        self.assertIs(srv.health["api_key_required"], False)

    def test_error_bodies(self):
        """400/422: detail is TypeSafe's (FastAPI's) validation list, locating the refusal; otherwise the reason.
        The reasons themselves are v1.1's (test_v11_requests_come_back_byte_for_byte)."""
        where = {"bad_json": ["body"], "not_object": ["body"], "no_questions": ["body", "questions"],
                 "empty_questions": ["body", "questions"], "question_not_object": ["body", "questions", "q"],
                 "bad_type": ["body", "questions", "q"], "one_level": ["body", "questions", "q"],
                 "eleven_levels": ["body", "questions", "q"], "choice_no_criteria": ["body", "questions", "q"],
                 "too_many_options": ["body", "questions", "q"], "too_many_questions": ["body", "questions"],
                 "too_long": ["body", "questions", "q"], "many_options": ["body", "questions", "c"]}
        for name, method, path, body in WIRE:
            status = GOLDEN[name][1]
            if status == 200:
                continue
            with self.subTest(name=name):
                code, head, text = call(self.srv.port, method, path, body)
                got = json.loads(text)
                self.assertEqual(code, status)
                if name in where:
                    kind = "json_invalid" if name == "bad_json" else "value_error"
                    self.assertEqual(got["detail"], [{"loc": where[name], "msg": got["error"], "type": kind}])
                else:
                    self.assertEqual(got["detail"], got["error"])
        mixed = {"state": "x", "questions": {"fine": {"type": "noul", "instructions": "ok?"},
                                             "it's": {"type": "score", "instructions": "?", "criteria": []}}}
        code, head, body = self.post(mixed)
        self.assertEqual((code, body["detail"][0]["loc"]), (422, ["body", "questions", "it's"]))

    def test_error_location_reads_only_the_opening(self):
        """A reason names its question at the start ("question 'x' ..."). A valid question's key quoted later in the
        reason, here inside another question's bad type, doesn't claim the refusal."""
        req = {"state": "x", "questions": {"good": {"type": "noul", "instructions": "ok?"},
                                           "bad": {"type": "question 'good'", "instructions": "?"}}}
        code, head, body = self.post(req)
        self.assertEqual(code, 422)
        self.assertEqual(body["error"], "unsupported question type \"question 'good'\"")
        self.assertEqual(body["detail"], [{"loc": ["body", "questions", "bad"], "msg": body["error"],
                                           "type": "value_error"}])
        # a reason that does open with a key is located at that key, however the key is spelled
        for key in ("good", "it's", 'say "hi"', "question 'good'"):
            with self.subTest(key=key):
                req = {"state": "x", "questions": {"ok": {"type": "noul", "instructions": "?"}, key: "not an object"}}
                code, head, body = self.post(req)
                self.assertEqual((code, body["error"]), (422, f"question {key!r} must be an object"))
                self.assertEqual(body["detail"][0]["loc"], ["body", "questions", key])

    def test_existing_clients_still_read_everything(self):
        """JevBench's stock typesafe adapter and the Decision Index kit's http engine, as they parse responses."""
        code, head, body = self.post(JEVBENCH, {"Authorization": "Bearer any", "User-Agent": "JevBench/1.2"})
        probs = jevbench_parse("choice", list(JEVBENCH["questions"]["decision"]["criteria"]), body)
        self.assertEqual(set(probs), {"late", "damaged", "wrong"})
        for qtype, q in (("noul", DOC_NOUL["questions"]["is_urgent"]),
                         ("score", DOC_SCORE["questions"]["frustration"])):
            req = {"state": "x", "model": "jev-latest", "questions": {"decision": q}}
            jevbench_parse(qtype, None, self.post(req)[2])
        for req in (DOC_NOUL, DOC_CHOICE, UNICODE):
            dikit_validate(req["questions"], self.post(req)[2])
        for name in ("too_many_options", "eleven_levels"):
            code, head, text = call(self.srv.port, *WIRE_BY_NAME[name])
            self.assertIn(code, (400, 413, 422))
            self.assertTrue(any(m in text for m in DI_CAPACITY_MARKERS), text)

    def test_server_side_only(self):
        """As documented: no CORS headers, so browsers can't call the server; methods the handler doesn't take get
        the HTTP server's own 501, which carries no request id (the handler's responses always do)."""
        origin = {"Origin": "https://app.example"}
        for method, path, body, want in (("POST", "/v1/systemone", J(DOC_NOUL), 200), ("GET", "/v1/models", None, 200),
                                         ("GET", "/healthz", None, 200), ("POST", "/v1/systemone", "{", 400),
                                         ("OPTIONS", "/v1/systemone", None, 501), ("HEAD", "/v1/systemone", None, 501)):
            with self.subTest(method=method, path=path):
                headers = dict(origin, **({"Access-Control-Request-Method": "POST"} if method == "OPTIONS" else {}))
                code, head, text = call(self.srv.port, method, path, body, headers)
                self.assertEqual(code, want)
                self.assertFalse([h for h in head if h.startswith("access-control-")], head)
                self.assertEqual("x-typesafe-request-id" in head, want != 501)

    def test_keep_alive(self):
        """Answers and refusals keep a connection usable; a 404 or 401 sent before reading the body closes it."""
        c = http.client.HTTPConnection("127.0.0.1", self.srv.port, timeout=30)
        try:
            for body, status in ((J(DOC_NOUL), 200), (J({"state": "x", "questions": {}}), 422), ("{", 400),
                                 (J(DOC_SCORE), 200)):
                c.request("POST", "/v1/systemone", body.encode("utf-8"), {"Content-Type": "application/json"})
                r = c.getresponse()
                r.read()
                self.assertEqual((r.status, r.getheader("Connection")), (status, None))
            c.request("POST", "/v1/elsewhere", J(DOC_NOUL).encode("utf-8"), {"Content-Type": "application/json"})
            r = c.getresponse()
            r.read()
            self.assertEqual((r.status, r.getheader("Connection")), (404, "close"))
        finally:
            c.close()
        self.assertEqual(self.post(DOC_NOUL)[0], 200)


@unittest.skipUnless(importlib.util.find_spec("api_doc"), "the Space's api_doc.py isn't beside blink.py")
class TestApiTabExamples(unittest.TestCase):
    """The Space's API tab documents serve.py: its examples, sent to serve.py, get exactly the documented bodies."""

    def test_the_documented_exchanges(self):
        import api_doc as d

        def shape(o):
            return {k: shape(v) for k, v in o.items()} if isinstance(o, dict) else type(o).__name__

        srv = Server(engine=PathFreeEngine)
        for req in (d.SHORT_REQUEST, d.EXAMPLE_REQUEST):
            code, head, text = call(srv.port, "POST", "/v1/systemone", J(req))
            body = json.loads(text)
            self.assertEqual(code, 200)
            TestWire.assert_typesafe(self, req["questions"], body)
            # the documented response differs only in its numbers, which are blink-4b's own
            if req is d.EXAMPLE_REQUEST:
                self.assertEqual(shape({**body, "model": d.SERVED_AS}), shape(d.EXAMPLE_RESPONSE))
        code, head, text = call(srv.port, "POST", "/v1/systemone", J(d.REFUSED_REQUEST))
        self.assertEqual((code, json.loads(text)), (422, d.REFUSED_BODY))
        code, head, text = call(srv.port, "GET", "/v1/models")
        self.assertEqual(json.loads(text.replace(json.dumps(srv.model)[1:-1], d.SERVED_AS)), d.MODELS_BODY)
        keyed = Server("--api-key", "your-key", engine=PathFreeEngine)
        code, head, text = call(keyed.port, "POST", "/v1/systemone", J(d.SHORT_REQUEST))
        self.assertEqual((code, head["www-authenticate"], json.loads(text)), (401, "Bearer", d.UNAUTHORIZED_BODY))
        code, head, text = call(keyed.port, "POST", "/v1/systemone", J(d.SHORT_REQUEST),
                                {"Authorization": "Bearer your-key"})
        self.assertEqual(code, 200)


@unittest.skipUnless(importlib.util.find_spec("typesafe_sdk"), "TypeSafe's Python SDK isn't installed")
class TestOfficialSDK(unittest.TestCase):
    """TypeSafe's own Python client against blink, pointed at it only through its documented environment variables."""

    def test_sdk(self):
        import typesafe_sdk as ts

        srv = Server("--api-key", "blink-key", engine=PathFreeEngine)
        env = {"TYPESAFE_BASE_URL": f"http://127.0.0.1:{srv.port}", "TYPESAFE_API_KEY": "blink-key"}
        with mock.patch.dict(os.environ, env):
            with ts.TypeSafeClient(retry=ts.RetryPolicy(max_retries=0)) as client:
                r = client.system_one(
                    state={"document": "I was charged twice. Please fix this ASAP."},
                    questions={
                        "billing": ts.Noul(instructions="Is this ticket about billing?"),
                        "tone": ts.Choice(instructions="What is the customer's tone?",
                                          criteria={"calm": None, "frustrated": None, "angry": None}),
                        "urgency": ts.Score(instructions="How urgent is this ticket?",
                                            criteria=["can wait", "this week", "today"]),
                    },
                )
                self.assertTrue(0 <= r.nouls["billing"].noul <= 1)
                self.assertIn(r.choices["tone"].choice, {"calm", "frustrated", "angry"})
                self.assertEqual(sorted(r.scores["urgency"].probabilities), [0, 1, 2])
                self.assertTrue(0 <= r.scores["urgency"].confidence <= 1)
                self.assertEqual(r.scores["urgency"].legend[2], "today")
                self.assertEqual((r.model, r.usage.output_tokens), (srv.model, 0))
                self.assertRegex(r.request_id, r"^[0-9a-f]{32}$")
                (m,) = client.models.list().models
                self.assertEqual(m.name, srv.model)
                with self.assertRaises(ts.TypeSafeUnprocessableEntityError) as refused:
                    client.system_one(state="x", questions={"q": {"type": "score", "criteria": ["one level"]}})
                self.assertIn("a score takes 2 to 10 levels", str(refused.exception))
        with ts.TypeSafeClient(api_key="wrong", base_url=env["TYPESAFE_BASE_URL"],
                               retry=ts.RetryPolicy(max_retries=0)) as client:
            with self.assertRaises(ts.TypeSafeAuthenticationError):
                client.system_one(state="x", questions={"q": ts.Noul(instructions="?")})


if __name__ == "__main__":
    unittest.main()
