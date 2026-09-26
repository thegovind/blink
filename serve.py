"""blink server: a Jev-compatible decision endpoint (the TypeSafe API's wire format).

    pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" accelerate safetensors huggingface_hub
    hf download thegovind/blink-4b --revision v1.0 --local-dir blink-4b
    python blink-4b/serve.py --model ./blink-4b --port 8000

POST /v1/systemone  {"state": ..., "model": ..., "questions": {...}}  ->  {"model", "answers", "usage"}
GET  /v1/models     ->  {"models": [{"name", "description", "release_date"}]}
GET  /healthz       ->  {"ok", "model", "revision", "weights_verified", "hub_offline", "warmup", "kernels", "versions",
                         "batching", "api_key_required"}

A client written for the TypeSafe API works unchanged against this server once its base URL points here (for the
official SDKs: TYPESAFE_BASE_URL=http://127.0.0.1:8000). The request's "model" is accepted and not used: this
server answers with the one model it serves and names it in the response. Any Authorization header is accepted and
ignored unless a key is set with --api-key or BLINK_API_KEY; then /v1/systemone and /v1/models need
"Authorization: Bearer <key>" and answer HTTP 401 without it (/healthz stays open).

Requests are served one at a time. A request blink can't answer (a malformed question, or one over a limit: options
per choice, context length, questions per request) gets HTTP 422 with the reason; nothing is truncated. A body that
isn't a JSON object gets HTTP 400. Error bodies are {"error": reason, "detail": ...}, where detail is a list of
{"loc", "msg", "type"} for a 400 or 422 and the reason again otherwise. Every response carries an
x-typesafe-request-id header. With weights.sha256 beside the weights, every listed file is hashed before serving
(weights_verified). Serving a local folder switches the Hugging Face libraries to offline mode before any of them
loads (hub_offline reports the setting the libraries actually use); the server does not otherwise restrict the network.

Opt-in cross-request batching (--batch-window-ms, default 0 = off): requests that arrive within the window are
decided together, up to --max-batch-requests; at most --max-queued-requests wait, and past that a request gets
HTTP 529 (overloaded) with Retry-After. Each request still gets its own answers or its own error.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import socket
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WARMUP = ("Order 4471 arrived with a cracked screen. The customer attached photos and wants a replacement.",
          {"route": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"returns": "Damaged or wrong items", "billing": "Charges and refunds",
                                  "shipping": "Late or lost parcels"}},
           "urgent": {"type": "noul", "instructions": "Does this need a reply today?"}})


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def verify(root: str):
    """True/False against weights.sha256 ("<sha256>  <file>" lines) beside the weights; None without one."""
    manifest = os.path.join(root, "weights.sha256")
    if not os.path.exists(manifest):
        return None, []
    bad = []
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                digest, name = line.split(None, 1)
                name = name.strip()
                path = os.path.join(root, name)
                if not os.path.exists(path) or sha256(path) != digest:
                    bad.append(name)
    return not bad, bad


def versions() -> dict:
    out = {}
    for mod in ("torch", "transformers", "fla"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def window_ms(value: str) -> float:
    """--batch-window-ms: 0 (off) or a finite window of at most 1000 ms."""
    try:
        ms = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    if not 0 <= ms <= 1000:  # also rejects nan and inf
        raise argparse.ArgumentTypeError("must be 0 (off) to 1000 ms")
    return ms


def int_range(lo: int, hi: int):
    def parse(value: str) -> int:
        try:
            n = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
        if not lo <= n <= hi:
            raise argparse.ArgumentTypeError(f"must be {lo}-{hi}")
        return n

    return parse


def api_key(value: str):
    """--api-key / BLINK_API_KEY: empty means open; a key must fit in an Authorization header (printable ASCII, no
    whitespace), as TypeSafe's SDKs require of theirs."""
    key = value.strip()
    if not key:
        return None
    if not (key.isascii() and key.isprintable()) or " " in key:
        raise argparse.ArgumentTypeError("must be printable ASCII without whitespace")
    return key


def bearer(header) -> str | None:
    """The token of an "Authorization: Bearer <token>" header, else None."""
    scheme, _, token = (header or "").strip().partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


def error_loc(blink, req: dict, message: str) -> list:
    """Where a refused request went wrong, as a FastAPI-style location (TypeSafe's 422 body): the question the reason
    opens with (blink's refusals that name one start "question '<key>' "), else the first question blink can't read,
    else the questions map. Only the opening counts: a key quoted later in a reason, say inside a bad type, isn't it."""
    questions = req.get("questions")
    if isinstance(questions, dict) and 0 < len(questions) <= getattr(blink, "MAX_QUESTIONS", 512):
        for key in questions:
            if message.startswith(f"question {key!r} "):
                return ["body", "questions", key]
        for key, q in questions.items():
            try:
                blink.question_options(q)
            except Exception:  # noqa: BLE001 - only locating the refusal, which is already decided
                return ["body", "questions", key]
    return ["body", "questions"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve blink over a Jev-compatible HTTP API.")
    ap.add_argument("--model", default=os.environ.get("BLINK_MODEL", "thegovind/blink-4b"))
    ap.add_argument("--revision", default=os.environ.get("BLINK_REVISION"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    # a string default is parsed only when the flag is absent, so a stray BLINK_BATCH_WINDOW_MS can't block an explicit 0
    ap.add_argument("--batch-window-ms", type=window_ms, default=os.environ.get("BLINK_BATCH_WINDOW_MS", "0"),
                    help="opt-in cross-request batching: requests arriving within this window are decided in one "
                         "GPU call (0 = one request at a time, the evaluated default; at most 1000)")
    ap.add_argument("--max-batch-requests", type=int_range(1, 64), default=16,
                    help="most requests decided together (1-64)")
    ap.add_argument("--max-queued-requests", type=int_range(1, 1024), default=64,
                    help="most requests waiting for a batch; past this a request gets HTTP 529 (1-1024)")
    # a string default goes through api_key() too, so BLINK_API_KEY is checked the same way
    ap.add_argument("--api-key", type=api_key, default=os.environ.get("BLINK_API_KEY", ""),
                    help="require 'Authorization: Bearer <key>' on /v1/systemone and /v1/models (default: open, any "
                         "Authorization header is accepted and ignored); BLINK_API_KEY keeps it out of the process list")
    a = ap.parse_args()

    local = os.path.isdir(a.model)
    if local:
        # the Hub libraries read these once, when they are first imported, so they must be set before that
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
            os.environ.setdefault(var, "1")
    os.environ["BLINK_ENGINE"] = "torch"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import blink

    busy = getattr(blink, "BlinkBusy", ())  # an older blink.py has no batching (and no BlinkBusy)

    if local:
        root = a.model
    else:
        from huggingface_hub import snapshot_download

        root = snapshot_download(a.model, revision=a.revision)
    verified, bad = verify(root)
    if verified is False:
        sys.exit(f"weights.sha256 mismatch: {', '.join(bad)}")
    engine = blink.TorchEngine(root, None, blink.TEMPERATURE)
    blink._ENGINE = engine
    first = blink.decide(*WARMUP)["answers"]
    repeat_identical = blink.decide(*WARMUP)["answers"] == first
    ver = versions()
    kernels = f"flash-linear-attention {ver['fla']}" if ver["fla"] else "reference (much slower; install flash-linear-attention)"
    try:
        from huggingface_hub import constants as hub_constants

        hub_offline = bool(hub_constants.HF_HUB_OFFLINE)
    except Exception:
        hub_offline = None
    health = {"ok": True, "model": a.model, "revision": a.revision, "weights_verified": verified,
              "hub_offline": hub_offline, "warmup": {"repeat_identical": repeat_identical}, "kernels": kernels,
              "versions": ver}
    lock = threading.Lock()
    batcher = (blink.Batcher(a.batch_window_ms / 1000.0, a.max_batch_requests, max_queued=a.max_queued_requests)
               if a.batch_window_ms > 0 else None)
    health["batching"] = ({"window_ms": a.batch_window_ms, "max_requests": a.max_batch_requests,
                           "max_queued": a.max_queued_requests} if batcher else None)
    health["api_key_required"] = a.api_key is not None
    # TypeSafe's model list, for clients that ask which names the model field takes; any name is accepted here
    listing = {"models": [{
        "name": a.model,
        "description": "blink: typed decisions (noul, choice, score) with option probabilities from one forward pass. "
                       "This server serves one model; a request's model field is accepted and not used.",
        "release_date": "",
    }]}

    def current_health() -> dict:
        if batcher is None:
            return health
        alive = batcher.alive()
        return {**health, "ok": health["ok"] and alive,
                "batching": {**health["batching"], "worker_alive": alive, "queued": batcher.q.qsize()}}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet by default
            pass

        def setup(self):
            super().setup()
            # a response is two writes (headers, then body); without TCP_NODELAY the body waits
            # on the client's delayed ACK, a flat ~40 ms on every request of a kept-alive connection
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def _send(self, code: int, obj: dict, headers: dict | None = None, reason: str | None = None) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code, reason)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("x-typesafe-request-id", uuid.uuid4().hex)  # read by TypeSafe's SDKs
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _fail(self, code: int, reason: str, detail=None, headers: dict | None = None, *, unread: bool = False,
                  status_text: str | None = None) -> None:
            """An error body: blink's {"error"} plus TypeSafe's (FastAPI's) {"detail"}. With the request body still
            unread, the connection is closed rather than left holding it."""
            if unread:
                self.close_connection = True
                headers = {**(headers or {}), "Connection": "close"}
            self._send(code, {"error": reason, "detail": reason if detail is None else detail}, headers, status_text)

        def _refused(self, code: int, reason: str, loc: list, kind: str) -> None:
            self._fail(code, reason, [{"loc": loc, "msg": reason, "type": kind}])

        def _authorized(self) -> bool:
            if a.api_key is None:
                return True
            token = bearer(self.headers.get("Authorization"))
            return token is not None and hmac.compare_digest(token.encode("utf-8"), a.api_key.encode("utf-8"))

        def _unauthorized(self, unread: bool) -> None:
            self._fail(401, "missing or invalid API key: send Authorization: Bearer <key>",
                       headers={"WWW-Authenticate": "Bearer"}, unread=unread)

        def do_GET(self):
            path = self.path.rstrip("/")
            if path in ("/healthz", "/health"):
                return self._send(200, current_health())
            if path == "/v1/models":
                if not self._authorized():
                    return self._unauthorized(unread=False)
                return self._send(200, listing)
            return self._fail(404, "not found")

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._fail(404, "not found", unread=True)
            if not self._authorized():
                return self._unauthorized(unread=True)
            try:
                size = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(size) or b"{}")
            except (ValueError, json.JSONDecodeError) as exc:
                return self._refused(400, f"invalid JSON: {exc}", ["body"], "json_invalid")
            if not isinstance(req, dict):
                return self._refused(400, "the body must be a JSON object", ["body"], "value_error")
            try:
                if batcher is not None:
                    out = batcher.submit(req.get("state"), req.get("questions"))
                else:
                    with lock:
                        out = blink.decide(req.get("state"), req.get("questions"))
            except blink.BlinkError as exc:
                return self._refused(422, str(exc), error_loc(blink, req, str(exc)), "value_error")
            except busy as exc:
                return self._fail(529, str(exc), headers={"Retry-After": "1"}, status_text="Overloaded")
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                return self._fail(500, f"{type(exc).__name__}: {exc}")
            return self._send(200, {
                "model": a.model,
                "answers": out["answers"],
                "usage": {"input_tokens": out["meta"]["input_tokens"], "output_tokens": 0},
            })

    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"blink serving {a.model} on http://{a.host}:{a.port} ({kernels}; weights_verified={verified}; "
          f"api key {'required' if a.api_key else 'not required'})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
