"""CPU-only HTTP and masked-label contract tests for the opt-in vLLM server."""

import asyncio
import concurrent.futures
import http.client
import importlib.util
import io
import json
import os
import signal
import socket
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_vllm as serving

renderer_path = next(
    (path for path in (HERE / "out/blink-4b/blink.py", HERE.parent / "space/blink.py") if path.is_file()),
    None,
)
if renderer_path is None:
    raise unittest.SkipTest("blink.py is not available in the research or public tree")
spec = importlib.util.spec_from_file_location("blink_serve_vllm_test", renderer_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load renderer: {renderer_path}")
blink = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = blink
spec.loader.exec_module(blink)

QUESTIONS = {
    "route": {"type": "choice", "instructions": "Where?", "criteria": {"first": "One", "second": "Two"}},
    "yes": {"type": "noul", "instructions": "Is it true?"},
    "level": {"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"]},
}


class FakeRenderer:
    def render(self, state, questions):
        if state == "overflow":
            raise blink.BlinkError("question 'route' renders to too many tokens")
        return [{"qkey": key, "keys": [name for name, _ in blink.question_options(question)],
                 "ids": [10, 20, len(str(state)), len(key)],
                 "cand": list(range(65, 65 + len(blink.question_options(question))))}
                for key, question in questions.items()]


class FakeVllm:
    def __init__(self):
        self.seen = []
        self.omit = False
        self.delay = 0
        self.started = threading.Event()

    async def generate(self, prompt, params, request_id):
        self.seen.append((prompt, params, request_id))
        self.started.set()
        if self.delay:
            await asyncio.sleep(self.delay)
        ids = params.allowed_token_ids[:-1] if self.omit else params.allowed_token_ids
        logprobs = {token: SimpleNamespace(logprob=float(token - 65)) for token in ids}
        yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(logprobs=[logprobs])])


class FakeEnqueueVllm:
    def __init__(self, expected):
        self.expected = expected
        self.active = set()
        self.aborts = []
        self.registered = threading.Event()
        self.abort_started = threading.Event()
        self.aborted = threading.Event()
        self.release = asyncio.Event()

    async def generate(self, prompt, params, request_id):
        collector = None
        try:
            self.active.add(request_id)
            if len(self.active) == self.expected:
                self.registered.set()
            await self.release.wait()
            collector = object()
            yield SimpleNamespace(finished=True, outputs=[])
        except asyncio.CancelledError:
            if collector is not None:
                await self.abort(request_id)
            raise

    async def abort(self, request_id):
        self.abort_started.set()
        await asyncio.sleep(0.05)
        self.aborts.append(request_id)
        self.active.remove(request_id)
        if not self.active:
            self.aborted.set()


def fake_scorer(engine=None):
    underlying = engine or FakeVllm()
    scorer = serving.VllmScorer(
        blink, FakeRenderer(), underlying,
        lambda **kwargs: SimpleNamespace(**kwargs),
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    return scorer, underlying


class ScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_e2_masked_logprobs_and_released_answer_shape(self):
        scorer, engine = fake_scorer()
        result = await scorer.decide("evidence", QUESTIONS)
        self.assertEqual(result["meta"]["input_tokens"], 4 * len(QUESTIONS))
        self.assertEqual(result["answers"]["route"]["choice"], "second")
        self.assertEqual(result["answers"]["level"]["choice"], "2")
        self.assertEqual(result["answers"]["yes"]["type"], "noul")
        for _, params, request_id in engine.seen:
            self.assertEqual(params.max_tokens, 1)
            self.assertEqual(params.logprobs, len(params.allowed_token_ids))
            self.assertEqual(params.temperature, 1.0)
            self.assertFalse(params.detokenize)
            self.assertEqual(len(request_id), 32)
        self.assertAlmostEqual(sum(result["answers"]["route"]["probabilities"].values()), 1.0)

    async def test_missing_label_is_an_error_not_a_zero_score(self):
        scorer, engine = fake_scorer()
        engine.omit = True
        with self.assertRaisesRegex(RuntimeError, "omitted an offered-label"):
            await scorer.decide("evidence", QUESTIONS)

    async def test_repeated_cancellation_finishes_external_id_abort(self):
        engine = FakeEnqueueVllm(expected=1)
        scorer, _ = fake_scorer(engine)
        task = asyncio.create_task(scorer.decide("evidence", {"route": QUESTIONS["route"]}))
        self.assertTrue(await asyncio.to_thread(engine.registered.wait, 2))
        task.cancel()
        self.assertTrue(await asyncio.to_thread(engine.abort_started.wait, 2))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(engine.aborts), 1)
        self.assertFalse(engine.active)


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_label_token_counts_against_effective_model_limit(self):
        scorer, engine = fake_scorer()
        question = {"decision": QUESTIONS["route"]}
        for length, limit, accepted in ((32767, 32768, True), (32768, 32768, False),
                                        (40000, 32768, False), (511, 512, True), (512, 512, False)):
            with self.subTest(length=length, limit=limit):
                scorer.max_model_len = limit
                scorer.renderer.render = lambda state, questions, length=length: [
                    {"qkey": "decision", "keys": ["first", "second"],
                     "ids": [1] * length, "cand": [65, 66]}
                ]
                engine.seen.clear()
                if accepted:
                    await scorer.decide("evidence", question)
                    self.assertEqual(len(engine.seen), 1)
                else:
                    with self.assertRaisesRegex(blink.BlinkError, "question 'decision' renders to"):
                        await scorer.decide("evidence", question)
                    self.assertEqual(engine.seen, [])

    async def test_one_overflow_rejects_all_question_prefills(self):
        scorer, engine = fake_scorer()
        scorer.max_model_len = 512
        scorer.renderer.render = lambda state, questions: [
            {"qkey": "route", "keys": ["first", "second"], "ids": [1] * 511, "cand": [65, 66]},
            {"qkey": "yes", "keys": ["yes", "no"], "ids": [1] * 512, "cand": [65, 66]},
        ]
        with self.assertRaisesRegex(blink.BlinkError, "question 'yes' renders to 512 tokens"):
            await scorer.decide("evidence", {"route": QUESTIONS["route"], "yes": QUESTIONS["yes"]})
        self.assertEqual(engine.seen, [])


class HttpFixture(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()

        def run():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.loop_thread = threading.Thread(target=run, daemon=True)
        self.loop_thread.start()
        self.scorer, self.engine = fake_scorer()
        self.open_http()

    def open_http(self, key="secret", concurrency=2, renderer_blink=blink, connections=None):
        if hasattr(self, "server"):
            self.server.shutdown()
            self.server.server_close()
            self.http_thread.join(timeout=5)
        health = {"ok": True, "model": "blink-4b", "warmup": {"repeat_identical": True},
                  "accepts_images": False}
        handler = serving.handler_for(
            renderer_blink, self.scorer, self.loop, "blink-4b", health, key, concurrency
        )
        self.server = serving.BoundedHTTPServer(
            ("127.0.0.1", 0), handler, connections or concurrency * 2 + 16
        )
        self.server.daemon_threads = True
        self.http_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.http_thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.http_thread.join(timeout=5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()

    def request(self, method, path, body=None, token="secret"):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        try:
            conn.request(method, path, body=body, headers=headers)
            result = conn.getresponse()
            content = json.loads(result.read())
            return result.status, dict(result.getheaders()), content
        finally:
            conn.close()


class HttpTests(HttpFixture):
    def test_health_models_and_authentication_contract(self):
        code, headers, body = self.request("GET", "/healthz", token=None)
        self.assertEqual(code, 200)
        self.assertTrue(body["ok"])
        self.assertIs(body["accepts_images"], False)
        self.assertTrue(headers["x-typesafe-request-id"])
        self.assertEqual(self.request("GET", "/v1/models", token=None)[0], 401)
        code, _, body = self.request("GET", "/v1/models")
        self.assertEqual(code, 200)
        self.assertEqual(list(body["models"][0]), ["name", "description", "release_date", "accepts_images"])
        self.assertEqual(body["models"][0]["name"], "blink-4b")
        self.assertIs(body["models"][0]["accepts_images"], False)
        self.assertEqual(self.request("GET", "/not-found")[0], 404)

    def test_choice_noul_score_usage_and_model_ignored(self):
        payload = {"model": "any-name", "state": "evidence", "questions": QUESTIONS}
        self.assertEqual(self.request("POST", "/v1/systemone", payload, token=None)[0], 401)
        code, headers, body = self.request("POST", "/v1/systemone", payload)
        self.assertEqual(code, 200)
        self.assertTrue(headers["x-typesafe-request-id"])
        self.assertEqual(body["model"], "blink-4b")
        self.assertEqual(body["usage"], {"input_tokens": 12, "output_tokens": 0})
        self.assertEqual(set(body["answers"]), set(QUESTIONS))
        self.assertEqual(body["answers"]["route"]["choice"], "second")

    def test_text_requests_keep_the_same_answers_and_usage(self):
        first = self.request("POST", "/v1/systemone", {
            "state": {"note": "data:image/png is only MIME prose"}, "questions": QUESTIONS,
        })
        second = self.request("POST", "/v1/systemone", {
            "state": {"note": "data:image/png is only MIME prose"}, "questions": QUESTIONS,
        })
        self.assertEqual(first[0], 200)
        self.assertEqual(second[0], 200)
        self.assertEqual(first[2], second[2])
        self.assertEqual(len(self.engine.seen), 2 * len(QUESTIONS))

    def test_inline_and_top_level_images_are_refused_before_scoring(self):
        for state, images, loc in (
            ({"screen": [{"content": "See data:image/png;base64,QUFB"}]}, None,
             ["body", "state", "screen", 0, "content"]),
            ("evidence", ["data:image/png;base64,QUFB"], ["body", "images", 0]),
            ("evidence", [], ["body", "images"]),
            ("evidence", None, ["body", "images"]),
        ):
            with self.subTest(loc=loc):
                payload = {"state": state, "questions": QUESTIONS}
                if not isinstance(state, dict) or images is not None:
                    payload["images"] = images
                code, _, body = self.request("POST", "/v1/systemone", payload)
                self.assertEqual(code, 422)
                self.assertEqual(body["error"], "this model reads text only")
                self.assertEqual(body["detail"], [
                    {"loc": loc, "msg": "this model reads text only", "type": "value_error"}
                ])
        self.assertEqual(self.engine.seen, [])

    def test_malformed_image_header_preserves_error_location(self):
        for payload, loc in (
            ({"state": {"screen": "data:image/png; base64,QUFB"}},
             ["body", "state", "screen"]),
            ({"state": "evidence", "images": ["data:image/png; base64,QUFB"]},
             ["body", "images", 0]),
        ):
            with self.subTest(loc=loc):
                code, _, body = self.request("POST", "/v1/systemone",
                                             {**payload, "questions": QUESTIONS})
                self.assertEqual(code, 422)
                self.assertEqual(body["detail"], [{
                    "loc": loc,
                    "msg": "image data URI parameters or whitespace are not supported",
                    "type": "value_error",
                }])
        self.assertEqual(self.engine.seen, [])

    def test_older_renderer_without_image_helpers_is_rejected_before_serving(self):
        legacy = SimpleNamespace(BlinkError=blink.BlinkError, MAX_QUESTIONS=blink.MAX_QUESTIONS,
                                 question_options=blink.question_options, validate=blink.validate)
        with self.assertRaisesRegex(RuntimeError, "v1.3 image scanner"):
            self.open_http(renderer_blink=legacy)
        with self.assertRaisesRegex(RuntimeError, "v1.3 image scanner"):
            serving.require_image_contract(SimpleNamespace())
        self.assertEqual(self.engine.seen, [])

    def test_images_key_refusal_precedes_invalid_questions(self):
        code, _, body = self.request("POST", "/v1/systemone", {
            "state": "evidence", "questions": {"bad": {"type": "other"}}, "images": [],
        })
        self.assertEqual(code, 422)
        self.assertEqual(body["detail"][0], {
            "loc": ["body", "images"], "msg": "this model reads text only", "type": "value_error",
        })
        self.assertEqual(self.engine.seen, [])

    def test_all_255_offered_options_are_returned(self):
        question = {"type": "choice", "instructions": "Select the candidate",
                    "criteria": {f"option_{i:03d}": f"Candidate {i}" for i in range(255)}}
        code, _, body = self.request("POST", "/v1/systemone",
                                     {"state": "synthetic", "questions": {"decision": question}})
        self.assertEqual(code, 200)
        self.assertEqual(len(body["answers"]["decision"]["probabilities"]), 255)
        self.assertEqual(body["usage"]["output_tokens"], 0)
        self.assertAlmostEqual(sum(body["answers"]["decision"]["probabilities"].values()), 1.0)
        self.assertEqual(len(self.engine.seen[-1][1].allowed_token_ids), 255)

    def test_422_and_400_error_detail(self):
        bad = {"state": "evidence", "questions": {"broken": {"type": "mystery"}}}
        code, _, body = self.request("POST", "/v1/systemone", bad)
        self.assertEqual(code, 422)
        self.assertEqual(body["detail"][0]["loc"], ["body", "questions", "broken"])
        self.assertEqual(self.request("POST", "/v1/systemone", b"not JSON")[0], 400)
        self.assertEqual(self.request("POST", "/v1/systemone", b"[]")[0], 400)
        self.assertEqual(self.request("POST", "/different", b"{}")[0], 404)
        self.assertEqual(self.request("POST", "/v1/systemone",
                                      {"state": "overflow", "questions": QUESTIONS})[0], 422)

    def test_context_overflow_is_located_before_any_prefill(self):
        self.scorer.max_model_len = 3
        code, _, body = self.request("POST", "/v1/systemone",
                                     {"state": "evidence", "questions": QUESTIONS})
        self.assertEqual(code, 422)
        self.assertEqual(body["detail"][0]["loc"], ["body", "questions", "route"])
        self.assertEqual(self.engine.seen, [])

    def test_internal_error_is_logged_with_request_id_but_not_sent_to_client(self):
        private = "private checkpoint at /private/model/weights"
        with mock.patch.object(self.scorer, "decide", side_effect=RuntimeError(private)), \
                mock.patch.object(serving.sys, "stderr", new_callable=io.StringIO) as log:
            code, headers, body = self.request("POST", "/v1/systemone",
                                                {"state": "evidence", "questions": QUESTIONS})
        self.assertEqual(code, 500)
        self.assertEqual(body, {"error": "internal error", "detail": "internal error"})
        self.assertNotIn(private, json.dumps(body))
        self.assertIn(private, log.getvalue())
        self.assertIn(headers["x-typesafe-request-id"], log.getvalue())

    def test_disconnected_response_write_does_not_emit_traceback(self):
        self.open_http(key=None)
        self.engine.delay = 0.1
        payload = json.dumps({"state": "evidence", "questions": QUESTIONS}).encode()
        client = socket.create_connection(("127.0.0.1", self.server.server_port), timeout=3)
        client.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n"
                       + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
        self.assertTrue(self.engine.started.wait(2))
        with mock.patch.object(serving.sys, "stderr", new_callable=io.StringIO) as log:
            client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            client.close()
            time.sleep(0.3)
            self.assertNotIn("Traceback", log.getvalue())

    def test_max_concurrency_returns_explicit_529(self):
        self.open_http(concurrency=1)
        self.engine.delay = 0.3
        answer = []
        thread = threading.Thread(
            target=lambda: answer.append(self.request(
                "POST", "/v1/systemone", {"state": "evidence", "questions": QUESTIONS}
            )), daemon=True
        )
        thread.start()
        self.assertTrue(self.engine.started.wait(2))
        status, headers, _ = self.request("POST", "/v1/systemone",
                                           {"state": "evidence", "questions": QUESTIONS})
        self.assertEqual(status, 529)
        self.assertEqual(headers["Retry-After"], "1")
        image_status, _, image_body = self.request("POST", "/v1/systemone", {
            "state": "data:image/png;base64,QUFB", "questions": QUESTIONS,
        })
        self.assertEqual(image_status, 422)
        self.assertEqual(image_body["detail"][0]["loc"], ["body", "state"])
        status, _, empty_body = self.request("POST", "/v1/systemone", {
            "state": "evidence", "questions": QUESTIONS, "images": [],
        })
        self.assertEqual(status, 422)
        self.assertEqual(empty_body["detail"][0]["loc"], ["body", "images"])
        thread.join(timeout=5)
        self.assertEqual(answer[0][0], 200)

    def test_http_timeout_aborts_requests_registered_before_collectors_exist(self):
        self.scorer, self.engine = fake_scorer(FakeEnqueueVllm(expected=len(QUESTIONS)))
        self.open_http()
        submit = asyncio.run_coroutine_threadsafe

        def immediate_timeout(coro, loop):
            actual = submit(coro, loop)

            class TimedOut:
                def result(self, timeout):
                    if not self_registered.wait(2):
                        raise AssertionError("requests did not register before timeout")
                    raise concurrent.futures.TimeoutError

                def cancel(self):
                    return actual.cancel()

            self_registered = self.engine.registered
            return TimedOut()

        with mock.patch.object(serving.asyncio, "run_coroutine_threadsafe", side_effect=immediate_timeout):
            code, _, body = self.request("POST", "/v1/systemone",
                                         {"state": "evidence", "questions": QUESTIONS})
        self.assertEqual(code, 504)
        self.assertEqual(body["error"], "decision timed out")
        self.assertTrue(self.engine.aborted.wait(3))
        self.assertEqual(len(self.engine.aborts), len(QUESTIONS))
        self.assertFalse(self.engine.active)

    def test_open_mode_accepts_any_authorization_header(self):
        self.open_http(key=None)
        self.assertEqual(self.request("GET", "/v1/models", token=None)[0], 200)
        self.assertEqual(self.request("POST", "/v1/systemone",
                                      {"state": "evidence", "questions": QUESTIONS},
                                      token="unrelated")[0], 200)


class FramingTests(HttpFixture):
    def setUp(self):
        super().setUp()
        self.open_http(key=None)

    def raw(self, wire: bytes, *, half_close=False, timeout=5) -> tuple[bytes, dict]:
        with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=timeout) as conn:
            conn.settimeout(timeout)
            conn.sendall(wire)
            if half_close:
                conn.shutdown(socket.SHUT_WR)
            chunks = []
            while chunk := conn.recv(65536):
                chunks.append(chunk)
        response = b"".join(chunks)
        self.assertEqual(response.count(b"HTTP/1.1 "), 1, response[:300])
        header, body = response.split(b"\r\n\r\n", 1)
        return header, json.loads(body)

    def test_get_body_is_closed_without_executing_embedded_request(self):
        embedded = b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"
        wire = (b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
                + str(len(embedded)).encode() + b"\r\n\r\n" + embedded)
        headers, body = self.raw(wire)
        self.assertIn(b" 400 ", headers)
        self.assertIn(b"Connection: close", headers)
        self.assertIn("GET requests must not include a body", body["error"])

    def test_transfer_encoding_duplicate_and_nondecimal_lengths_close(self):
        cases = (
            b"Transfer-Encoding: chunked\r\n",
            b"Content-Length: 2\r\nContent-Length: 2\r\n",
            b"Content-Length: 0_0\r\n",
            b"Content-Length: +2\r\n",
            b"Content-Length: -2\r\n",
            b"Content-Length: 0, 0\r\n",
        )
        for method in (b"GET", b"POST"):
            for fields in cases:
                with self.subTest(method=method, fields=fields):
                    wire = method + b" /healthz HTTP/1.1\r\nHost: localhost\r\n" + fields + (
                        b"\r\nGET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n"
                    )
                    headers, body = self.raw(wire)
                    self.assertIn(b" 400 ", headers)
                    self.assertIn(b"Connection: close", headers)
                    self.assertEqual(body["detail"][0]["loc"], ["body"])

    def test_premature_eof_returns_located_400_and_closes(self):
        headers, body = self.raw(
            b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: 30\r\n\r\n{}",
            half_close=True,
        )
        self.assertIn(b" 400 ", headers)
        self.assertIn(b"Connection: close", headers)
        self.assertIn("incomplete request body", body["error"])
        self.assertEqual(body["detail"][0]["loc"], ["body"])

    def test_exact_20_mib_cap_accepts_and_one_byte_over_rejects(self):
        request = {"state": "evidence", "questions": {"yes": QUESTIONS["yes"]}}
        encoded = json.dumps(request).encode()
        payload = encoded + b" " * (serving.MAX_BODY_BYTES - len(encoded))
        wire = (b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
        headers, body = self.raw(wire, timeout=15)
        self.assertIn(b" 200 ", headers)
        self.assertIn("yes", body["answers"])
        headers, body = self.raw(
            b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
            + str(serving.MAX_BODY_BYTES + 1).encode() + b"\r\n\r\n"
        )
        self.assertIn(b" 413 ", headers)
        self.assertIn(b"Connection: close", headers)
        self.assertEqual(body["error"], "request body exceeds 20 MiB")

    def test_bodyless_persistent_get_remains_usable(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            conn.request("GET", "/healthz")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            response.read()
            conn.request("GET", "/v1/models")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            response.read()
        finally:
            conn.close()

    def test_bounded_connection_admission_and_idle_deadline(self):
        self.open_http(key=None, connections=2)
        with mock.patch.object(self.server.RequestHandlerClass, "timeout", 0.3):
            stalled = [socket.create_connection(("127.0.0.1", self.server.server_port), timeout=3)
                       for _ in range(2)]
            try:
                for conn in stalled:
                    conn.settimeout(3)
                    conn.sendall(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n")
                for _ in range(100):
                    if not self.server.connections.acquire(blocking=False):
                        break
                    self.server.connections.release()
                    time.sleep(0.005)
                else:
                    self.fail("two stalled connections were not admitted")
                with mock.patch.object(serving.sys, "stderr", new_callable=io.StringIO) as log:
                    with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=3) as third:
                        third.settimeout(3)
                        self.assertEqual(third.recv(1), b"")
                    self.assertIn("connection limit", log.getvalue())
                for conn in stalled:
                    self.assertEqual(conn.recv(1), b"")
                self.assertEqual(self.request("GET", "/healthz", token=None)[0], 200)
            finally:
                for conn in stalled:
                    conn.close()


class CheckpointTests(unittest.TestCase):
    def test_loads_local_renderer_with_postponed_dataclass_annotations(self):
        loaded = serving.load_blink(renderer_path.parent)
        self.assertIs(sys.modules["blink_vllm_model"], loaded)
        self.assertTrue(callable(loaded.validate))

    def test_default_v32_engine_settings(self):
        with mock.patch.dict(os.environ, {"BLINK_API_KEY": ""}):
            args = serving.parse_args([])
        self.assertEqual(args.max_concurrency, 32)
        self.assertEqual(args.max_num_seqs, 32)
        self.assertEqual(args.max_num_batched_tokens, 8192)
        self.assertEqual(args.max_model_len, 32768)
        self.assertEqual(args.tensor_parallel_size, 1)
        self.assertEqual(args.gpu_memory_utilization, 0.85)
        self.assertFalse(args.prefix_cache)
        self.assertFalse(args.no_chunked_prefill)
        self.assertEqual(args.quantization, "auto")
        self.assertIsNone(args.api_key)

    def test_detects_quantization_and_rejects_explicit_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            self.assertIsNone(serving.detect_quantization(root, "auto"))
            with self.assertRaisesRegex(ValueError, "requested compressed-tensors"):
                serving.detect_quantization(root, "compressed-tensors")
            (root / "config.json").write_text(json.dumps({
                "quantization_config": {"quant_method": "compressed-tensors"}
            }))
            self.assertEqual(serving.detect_quantization(root, "auto"), "compressed-tensors")
            with self.assertRaisesRegex(ValueError, "requested none"):
                serving.detect_quantization(root, "none")

    def test_multimodal_limit_comes_from_config_not_folder_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}))
            self.assertFalse(serving.has_vision_model(root))
            (root / "config.json").write_text(json.dumps({
                "model_type": "qwen3_5", "vision_config": {"hidden_size": 32},
            }))
            self.assertTrue(serving.has_vision_model(root))

    def test_weight_manifest_checks_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "weights.sha256").write_text(
                f"{serving.sha256(root / 'model.safetensors')}  model.safetensors\n"
            )
            self.assertTrue(serving.verify_weights(root))
            (root / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                serving.verify_weights(root)


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        (Path(self.folder.name) / "config.json").write_text("{}", encoding="utf-8")
        self.engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(head_dtype="torch.float32", max_model_len=32768),
                cache_config=SimpleNamespace(enable_prefix_caching=False),
                scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
            ),
            shutdown=mock.Mock(),
        )
        self.warmup_started = threading.Event()

    def run_startup(self, decide, *, interrupt=None, factory=None, before_start=False):
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
        vllm = types.ModuleType("vllm")
        vllm.__path__ = []
        vllm.AsyncEngineArgs = lambda **kwargs: kwargs
        vllm.SamplingParams = object
        inputs = types.ModuleType("vllm.inputs")
        inputs.TokensPrompt = object
        vllm_v1 = types.ModuleType("vllm.v1")
        vllm_v1.__path__ = []
        vllm_engine = types.ModuleType("vllm.v1.engine")
        vllm_engine.__path__ = []
        async_llm = types.ModuleType("vllm.v1.engine.async_llm")
        async_llm.AsyncLLM = SimpleNamespace(from_engine_args=factory or (lambda _: self.engine))
        modules = {"transformers": transformers, "vllm": vllm, "vllm.inputs": inputs,
                   "vllm.v1": vllm_v1, "vllm.v1.engine": vllm_engine,
                   "vllm.v1.engine.async_llm": async_llm}

        original_submit = serving.submit_initializer
        queued, release = threading.Event(), threading.Event()

        def submit(loop, initialize):
            if before_start:
                def pause():
                    queued.set()
                    if not release.wait(5):
                        raise AssertionError("initializer scheduling was not released")

                loop.call_soon_threadsafe(pause)
            future, scheduled = original_submit(loop, initialize)
            if interrupt is None:
                return future, scheduled

            class InterruptingFuture:
                def result(self, timeout):
                    if not (queued if before_start else self_warmup).wait(5):
                        raise AssertionError("engine did not reach the requested interrupt point")
                    if before_start:
                        threading.Timer(0.05, release.set).start()
                    interrupt()

            self_warmup = self.warmup_started
            return InterruptingFuture(), scheduled

        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(serving, "load_blink", return_value=blink), \
                mock.patch.object(blink.TorchEngine, "_verify_labels", return_value=(["A"], [65])), \
                mock.patch.object(serving, "submit_initializer", side_effect=submit), \
                mock.patch.object(serving.VllmScorer, "decide", decide):
            serving.main(["--model", self.folder.name])

    def test_warmup_failure_shuts_engine_down_once(self):
        async def fail_warmup(*_):
            raise RuntimeError("warmup failed")

        with self.assertRaisesRegex(RuntimeError, "warmup failed"):
            self.run_startup(fail_warmup)
        self.engine.shutdown.assert_called_once_with()

    def test_interrupt_during_warmup_shuts_engine_down_once(self):
        async def wait_during_warmup(*_):
            self.warmup_started.set()
            await asyncio.sleep(30)

        def interrupt():
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_startup(wait_during_warmup, interrupt=interrupt)
        self.engine.shutdown.assert_called_once_with()

    def test_sigterm_during_warmup_shuts_engine_down_once(self):
        async def wait_during_warmup(*_):
            self.warmup_started.set()
            await asyncio.sleep(30)

        previous = signal.getsignal(signal.SIGTERM)

        def interrupt():
            handler = signal.getsignal(signal.SIGTERM)
            self.assertIsNot(handler, previous)
            handler(signal.SIGTERM, None)

        with self.assertRaises(KeyboardInterrupt):
            self.run_startup(wait_during_warmup, interrupt=interrupt)
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)
        self.engine.shutdown.assert_called_once_with()

    def test_interrupt_while_engine_factory_is_running_shuts_down_late_engine(self):
        release_factory = threading.Event()
        started = time.monotonic()

        def factory(_):
            self.warmup_started.set()
            if not release_factory.wait(5):
                raise AssertionError("engine factory did not finish")
            return self.engine

        async def wait_during_warmup(*_):
            await asyncio.sleep(30)

        def interrupt():
            threading.Timer(0.2, release_factory.set).start()
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_startup(wait_during_warmup, factory=factory, interrupt=interrupt)
        self.engine.shutdown.assert_called_once_with()
        self.assertGreater(time.monotonic() - started, 0.15)
        self.assertFalse(any(t.name == "blink-vllm-async" and t.is_alive() for t in threading.enumerate()))

    def test_timeout_while_initializing_shuts_engine_down_once(self):
        async def wait_during_warmup(*_):
            self.warmup_started.set()
            await asyncio.sleep(30)

        def timeout():
            raise concurrent.futures.TimeoutError

        with self.assertRaises(concurrent.futures.TimeoutError):
            self.run_startup(wait_during_warmup, interrupt=timeout)
        self.engine.shutdown.assert_called_once_with()

    def test_interrupt_before_initialize_starts_closes_loop(self):
        factory = mock.Mock(return_value=self.engine)

        async def warmup(*_):
            return {"answers": {}}

        with self.assertRaises(KeyboardInterrupt):
            self.run_startup(warmup, factory=factory, before_start=True,
                             interrupt=lambda: (_ for _ in ()).throw(KeyboardInterrupt))
        self.assertEqual(self.engine.shutdown.call_count, factory.call_count)
        self.assertFalse(any(t.name == "blink-vllm-async" and t.is_alive() for t in threading.enumerate()))

    def test_abandoned_factory_closes_late_engine_once(self):
        owner = serving.EngineOwner()
        owner.abandon()
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            owner.adopt(self.engine)
        owner.close()
        self.engine.shutdown.assert_called_once_with()

    def test_invalid_effective_config_closes_created_engine_once(self):
        self.engine.vllm_config.model_config.head_dtype = "torch.bfloat16"

        async def warmup(*_):
            self.fail("no warmup should run after invalid engine settings")

        with self.assertRaisesRegex(RuntimeError, "FP32"):
            self.run_startup(warmup)
        self.engine.shutdown.assert_called_once_with()

    def test_changed_effective_context_limit_fails_before_warmup(self):
        self.engine.vllm_config.model_config.max_model_len = 16384

        async def warmup(*_):
            self.fail("no warmup should run with an unqualified effective context")

        with self.assertRaisesRegex(RuntimeError, "max-model-len"):
            self.run_startup(warmup)
        self.engine.shutdown.assert_called_once_with()

    def test_main_rejects_legacy_renderer_before_importing_vllm(self):
        legacy = SimpleNamespace(BlinkError=blink.BlinkError, validate=blink.validate)
        with mock.patch.object(serving, "load_blink", return_value=legacy), \
                mock.patch.dict(sys.modules, {"vllm": None}), \
                self.assertRaisesRegex(RuntimeError, "v1.3 image scanner"):
            serving.main(["--model", self.folder.name])
        self.engine.shutdown.assert_not_called()


if __name__ == "__main__":
    unittest.main()
