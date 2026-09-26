"""serve.py over real HTTP with a stand-in engine (no GPU, no weights): every request gets its own status, answers
and usage with batching off (the release path) and on; an overloaded batching queue answers 503; bad batching
settings stop the server before anything loads.

  cd release && BLINK_MOCK=1 python -m unittest test_serve
"""
import concurrent.futures
import contextlib
import http.client
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
for d in (os.path.join(HERE, "..", "space"), HERE):
    if os.path.exists(os.path.join(d, "blink.py")):
        sys.path.insert(0, d)
        break
for d in (HERE, os.path.join(HERE, "..")):
    if os.path.exists(os.path.join(d, "serve.py")):
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
    """serve.main() in a daemon thread, with FakeBatchEngine in place of the torch engine. The newest server's
    engine is blink._ENGINE, which is the one blink.decide() compares against."""

    def __init__(self, *argv: str):
        self.port = free_port()
        args = ["serve.py", "--model", tempfile.mkdtemp(), "--port", str(self.port), *argv]
        with mock.patch.object(blink, "TorchEngine", FakeBatchEngine), mock.patch.object(sys, "argv", args), \
                mock.patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
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
                                           "kernels", "versions", "batching"})
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

    def test_overload_answers_503(self):
        srv = Server("--batch-window-ms", "1", "--max-batch-requests", "1", "--max-queued-requests", "1")
        entered, release = hold_worker(srv.engine)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            first = pool.submit(post, srv.port, *REQS[0])
            self.assertTrue(entered.wait(10))
            second = pool.submit(post, srv.port, *REQS[1])
            wait_for(lambda: health(srv.port)["batching"]["queued"] == 1)
            code, body, headers = post(srv.port, *REQS[2])
            self.assertEqual(code, 503)
            self.assertEqual(headers.get("Retry-After"), "1")
            self.assertIn("error", body)
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
               ["--max-batch-requests", "65"], ["--max-queued-requests", "0"])
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


if __name__ == "__main__":
    unittest.main()
