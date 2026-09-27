"""Opt-in, text-only vLLM server for the TypeSafe System One API.

Install vLLM 0.30.0 and the model's runtime dependencies. For the qualified
bf16 blink-4b checkpoint, run the V32 configuration:

    python serve_vllm.py --model ./blink-4b --host 127.0.0.1 --port 8000 \\
      --quantization none --max-concurrency 32 --max-num-seqs 32 \\
      --max-num-batched-tokens 8192 --max-model-len 32768 \\
      --tensor-parallel-size 1 --gpu-memory-utilization 0.85

Chunked prefill is on and prefix caching is off. Images receive a located 422.
Each question's prompt plus its one label token must fit --max-model-len
(default 32768); the default serve.py may use a larger context. Client sockets
time out after 60 seconds, with at most 2*max-concurrency+16 admitted connections.
Use a buffering reverse proxy for public ingress.
The released serve.py remains the default. This server uses the model folder's
blink.py to validate, render and assemble every typed decision. It asks vLLM
for processed logprobs over the offered single-token letters, with an FP32
offered-label head, and never returns a generated text completion.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import hmac
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import signal
import socket
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WARMUP = (
    "Order 4471 arrived with a cracked screen. The customer wants a replacement.",
    {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"returns": "Damaged items", "billing": "Charges", "shipping": "Late parcels"},
        },
        "urgent": {"type": "noul", "instructions": "Does this need a reply today?"},
    },
)
MAX_BODY_BYTES = 20 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_weights(root: Path) -> bool | None:
    manifest = root / "weights.sha256"
    if not manifest.is_file():
        return None
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, *names = line.split()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or len(names) != 1:
            raise ValueError("invalid weights.sha256 entry")
        name = Path(names[0])
        if name.is_absolute() or name.name != names[0] or not (root / name).is_file():
            raise ValueError(f"invalid or missing weight manifest file: {name}")
        if sha256(root / name) != digest:
            raise ValueError(f"weight checksum mismatch: {name}")
        checked += 1
    if not checked:
        raise ValueError("empty weights.sha256")
    return True


def model_config(root: Path) -> dict:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("model config must be a JSON object")
    return config


def detect_quantization(root: Path, requested: str) -> str | None:
    config = model_config(root)
    text_config = config.get("text_config") or {}
    if not isinstance(text_config, dict):
        raise TypeError("text model config must be a JSON object")
    settings = config.get("quantization_config") or text_config.get("quantization_config")
    if settings is not None and (not isinstance(settings, dict)
                                 or settings.get("quant_method") != "compressed-tensors"):
        raise ValueError("unsupported checkpoint quantization")
    actual = "compressed-tensors" if settings else "none"
    if requested != "auto" and requested != actual:
        raise ValueError(f"requested {requested} but checkpoint quantization is {actual}")
    return None if actual == "none" else actual


def has_vision_model(root: Path) -> bool:
    config = model_config(root)
    return config.get("model_type") == "qwen3_5" or config.get("vision_config") is not None


def load_blink(root: Path):
    path = root / "blink.py"
    if not path.is_file():
        raise FileNotFoundError(f"model folder has no renderer: {path}")
    spec = importlib.util.spec_from_file_location("blink_vllm_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import renderer: {path}")
    blink = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = blink
    try:
        spec.loader.exec_module(blink)
    except BaseException:
        del sys.modules[spec.name]
        raise
    return blink


def bearer(value: str | None) -> str | None:
    scheme, _, token = (value or "").strip().partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None


def api_key(value: str) -> str | None:
    key = value.strip()
    if not key:
        return None
    if not key.isascii() or not key.isprintable() or any(char.isspace() for char in key):
        raise argparse.ArgumentTypeError("API key must be printable ASCII without whitespace")
    return key


def int_range(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(f"must be {minimum}-{maximum}")
        return number

    return parse


def fraction(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("must be strictly between 0 and 1")
    return number


def error_loc(blink, request: dict, message: str) -> list:
    questions = request.get("questions")
    if isinstance(questions, dict) and 0 < len(questions) <= blink.MAX_QUESTIONS:
        for key in questions:
            if message.startswith(f"question {key!r} "):
                return ["body", "questions", key]
        for key, question in questions.items():
            try:
                blink.question_options(question)
            except blink.BlinkError:
                return ["body", "questions", key]
    return ["body", "questions"]


def require_image_contract(blink) -> None:
    missing = [name for name in ("contains_image_uri", "inspect_images")
               if not callable(getattr(blink, name, None))]
    blink_error = getattr(blink, "BlinkError", None)
    if not isinstance(blink_error, type):
        missing.append("BlinkError")
    image_error = getattr(blink, "ImageError", None)
    if (not isinstance(image_error, type) or not isinstance(blink_error, type)
            or not issubclass(image_error, blink_error)):
        missing.append("ImageError")
    if missing:
        raise RuntimeError(f"blink.py must provide the v1.3 image scanner: {', '.join(missing)}")


class EngineOwner:
    def __init__(self):
        self.lock = threading.Lock()
        self.engine = None
        self.abandoned = False

    def adopt(self, engine) -> None:
        with self.lock:
            if not self.abandoned:
                self.engine = engine
                return
        engine.shutdown()
        raise RuntimeError("vLLM startup was interrupted")

    def abandon(self) -> None:
        with self.lock:
            self.abandoned = True

    def close(self) -> None:
        with self.lock:
            self.abandoned = True
            engine, self.engine = self.engine, None
        if engine is not None:
            engine.shutdown()


def submit_initializer(loop, initialize):
    result = concurrent.futures.Future()
    scheduled = concurrent.futures.Future()

    def start():
        try:
            task = loop.create_task(initialize())
        except Exception as exc:  # noqa: BLE001 - notify both waiters of a failed task creation
            scheduled.set_exception(exc)
            result.set_exception(exc)
            return
        scheduled.set_result(task)

        def finish(done):
            if done.cancelled():
                result.cancel()
            elif (error := done.exception()) is not None:
                result.set_exception(error)
            else:
                result.set_result(done.result())

        task.add_done_callback(finish)

    loop.call_soon_threadsafe(start)
    return result, scheduled


async def await_cleanup(awaitable):
    pending = asyncio.ensure_future(awaitable)
    while not pending.done():
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
    return await pending


class VllmScorer:
    """Use exactly the released renderer and E2's masked-logprob readout."""

    def __init__(self, blink, renderer, engine, sampling_params, tokens_prompt, max_model_len=32768):
        self.blink = blink
        self.renderer = renderer
        self.engine = engine
        self.sampling_params = sampling_params
        self.tokens_prompt = tokens_prompt
        self.max_model_len = max_model_len

    async def _read(self, item: dict) -> list[float]:
        ids = item["cand"]
        params = self.sampling_params(
            max_tokens=1,
            temperature=1.0,
            logprobs=len(ids),
            allowed_token_ids=list(ids),
            detokenize=False,
        )
        request_id = uuid.uuid4().hex
        output = None
        try:
            async for response in self.engine.generate(
                self.tokens_prompt(prompt_token_ids=item["ids"]), params, request_id
            ):
                if response.finished:
                    output = response
        except asyncio.CancelledError:
            await await_cleanup(self.engine.abort(request_id))
            raise
        if output is None or len(output.outputs) != 1 or not output.outputs[0].logprobs:
            raise RuntimeError(f"vLLM returned no complete label scores for {item['qkey']}")
        logprobs = output.outputs[0].logprobs[0]
        if any(token not in logprobs or not math.isfinite(float(logprobs[token].logprob))
               for token in ids):
            raise RuntimeError(f"vLLM omitted an offered-label logprob for {item['qkey']}")
        return [float(logprobs[token].logprob) for token in ids]

    async def decide(self, state, questions: dict) -> dict:
        self.blink.validate(questions)
        work = self.renderer.render(state, questions)
        for item in work:
            length = len(item["ids"])
            if length + 1 > self.max_model_len:
                raise self.blink.BlinkError(
                    f"question {item['qkey']!r} renders to {length} tokens; "
                    f"this server's max-model-len is {self.max_model_len}"
                )
        tasks = [asyncio.create_task(self._read(item)) for item in work]
        try:
            rows = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            results = await await_cleanup(asyncio.gather(*tasks, return_exceptions=True))
            for result in results:
                if isinstance(result, Exception):
                    print(f"vLLM label cleanup failed: {type(result).__name__}: {result}",
                          file=sys.stderr, flush=True)
            raise
        answers = {
            item["qkey"]: self.blink.answer_for(
                questions[item["qkey"]], item["keys"], self.blink.softmax(logits, 1.0)
            )
            for item, logits in zip(work, rows)
        }
        return {"answers": answers, "meta": {"input_tokens": sum(len(item["ids"]) for item in work)}}


class BoundedHTTPServer(ThreadingHTTPServer):
    def __init__(self, address, handler, max_connections: int):
        self.connections = threading.BoundedSemaphore(max_connections)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            print("vLLM HTTP connection limit reached", file=sys.stderr, flush=True)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


def handler_for(blink, scorer: VllmScorer, loop: asyncio.AbstractEventLoop,
                model_id: str, health: dict, key: str | None, max_concurrency: int):
    require_image_contract(blink)
    slots = threading.BoundedSemaphore(max_concurrency)
    listing = {"models": [{
        "name": model_id,
        "description": "blink: typed decisions (noul, choice, score) with option probabilities from one prefill. "
                       "This server serves one model; a request's model field is accepted and not used.",
        "release_date": "",
        "accepts_images": False,
    }]}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 60

        def log_message(self, fmt, *args):
            pass

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def send_json(self, code: int, obj: dict, headers: dict | None = None,
                      status_text: str | None = None, request_id: str | None = None) -> None:
            body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code, status_text)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("x-typesafe-request-id", request_id or uuid.uuid4().hex)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            try:
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def fail(self, code: int, reason: str, detail=None, headers: dict | None = None,
                 *, unread: bool = False, status_text: str | None = None,
                 request_id: str | None = None) -> None:
            if unread:
                self.close_connection = True
                headers = {**(headers or {}), "Connection": "close"}
            self.send_json(code, {"error": reason, "detail": reason if detail is None else detail},
                           headers, status_text, request_id)

        def refused(self, code: int, reason: str, location: list, kind: str, *, unread: bool = False) -> None:
            self.fail(code, reason, [{"loc": location, "msg": reason, "type": kind}], unread=unread)

        def content_length(self) -> int | None:
            if self.headers.get_all("Transfer-Encoding"):
                self.refused(400, "Transfer-Encoding is not supported", ["body"], "value_error", unread=True)
                return None
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) > 1:
                self.refused(400, "duplicate Content-Length", ["body"], "value_error", unread=True)
                return None
            if not lengths:
                return 0
            raw = lengths[0]
            if re.fullmatch(r"[0-9]+", raw) is None:
                self.refused(400, "invalid Content-Length", ["body"], "value_error", unread=True)
                return None
            digits = raw.lstrip("0") or "0"
            if len(digits) > len(str(MAX_BODY_BYTES)):
                return MAX_BODY_BYTES + 1
            return int(digits)

        def authorized(self) -> bool:
            if key is None:
                return True
            token = bearer(self.headers.get("Authorization"))
            return token is not None and hmac.compare_digest(token.encode(), key.encode())

        def do_GET(self):
            size = self.content_length()
            if size is None:
                return
            if size:
                return self.refused(400, "GET requests must not include a body",
                                    ["body"], "value_error", unread=True)
            path = self.path.rstrip("/")
            if path in ("/healthz", "/health"):
                return self.send_json(200 if health["ok"] else 503, health)
            if path == "/v1/models":
                if not self.authorized():
                    return self.fail(401, "missing or invalid API key: send Authorization: Bearer <key>",
                                     headers={"WWW-Authenticate": "Bearer"})
                return self.send_json(200, listing)
            return self.fail(404, "not found")

        def do_POST(self):
            size = self.content_length()
            if size is None:
                return
            if size > MAX_BODY_BYTES:
                return self.fail(413, "request body exceeds 20 MiB", unread=True)
            if self.path.rstrip("/") != "/v1/systemone":
                return self.fail(404, "not found", unread=True)
            if not self.authorized():
                return self.fail(401, "missing or invalid API key: send Authorization: Bearer <key>",
                                 headers={"WWW-Authenticate": "Bearer"}, unread=True)
            try:
                payload = self.rfile.read(size)
            except (OSError, TimeoutError) as exc:
                return self.refused(400, f"incomplete request body: {exc}", ["body"], "value_error", unread=True)
            if len(payload) != size:
                return self.refused(400, "incomplete request body", ["body"], "value_error", unread=True)
            try:
                request = json.loads(payload or b"{}")
            except (ValueError, UnicodeDecodeError) as exc:
                return self.refused(400, f"invalid JSON: {exc}", ["body"], "json_invalid")
            if not isinstance(request, dict):
                return self.refused(400, "the body must be a JSON object", ["body"], "value_error")
            try:
                if "images" in request:
                    if not isinstance(request["images"], list):
                        raise blink.ImageError("this model reads text only", ["body", "images"])
                    submission = blink.inspect_images(request.get("state"), request["images"])
                    location = (submission.loc if submission is not None and request["images"]
                                else ["body", "images"])
                    raise blink.ImageError("this model reads text only", location)
                if blink.contains_image_uri(request.get("state")):
                    submission = blink.inspect_images(request.get("state"))
                    if submission is not None:
                        raise blink.ImageError("this model reads text only", submission.loc)
                blink.validate(request.get("questions"))
            except blink.BlinkError as exc:
                loc = exc.loc if isinstance(exc, getattr(blink, "ImageError", ())) else error_loc(blink, request, str(exc))
                return self.refused(422, str(exc), loc, "value_error")
            if not slots.acquire(blocking=False):
                return self.fail(529, "too many concurrent requests",
                                 headers={"Retry-After": "1"}, status_text="Overloaded")
            future = None
            try:
                future = asyncio.run_coroutine_threadsafe(
                    scorer.decide(request.get("state"), request["questions"]), loop
                )
                output = future.result(timeout=180)
            except concurrent.futures.TimeoutError:
                future.cancel()
                return self.fail(504, "decision timed out")
            except blink.BlinkError as exc:
                loc = exc.loc if isinstance(exc, getattr(blink, "ImageError", ())) else error_loc(blink, request, str(exc))
                return self.refused(422, str(exc), loc, "value_error")
            except Exception as exc:  # noqa: BLE001 - explicit HTTP error for a failed inference
                request_id = uuid.uuid4().hex
                print(f"vLLM inference failed request_id={request_id}: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                return self.fail(500, "internal error", request_id=request_id)
            finally:
                slots.release()
            return self.send_json(200, {
                "model": model_id,
                "answers": output["answers"],
                "usage": {"input_tokens": output["meta"]["input_tokens"], "output_tokens": 0},
            })

    return Handler


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("BLINK_MODEL", "thegovind/blink-4b"))
    parser.add_argument("--revision", default=os.environ.get("BLINK_REVISION"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int_range(1, 65535), default=8000)
    parser.add_argument("--quantization", choices=("auto", "none", "compressed-tensors"), default="auto")
    parser.add_argument("--max-concurrency", type=int_range(1, 1024), default=32)
    parser.add_argument("--max-num-seqs", type=int_range(1, 1024), default=32)
    parser.add_argument("--max-num-batched-tokens", type=int_range(512, 131072), default=8192)
    parser.add_argument("--max-model-len", type=int_range(512, 131072), default=32768)
    parser.add_argument("--tensor-parallel-size", type=int_range(1, 8), default=1)
    parser.add_argument("--gpu-memory-utilization", type=fraction, default=0.85)
    parser.add_argument("--prefix-cache", action="store_true", help="experimental; default off")
    parser.add_argument("--no-chunked-prefill", action="store_true", help="experimental; default on")
    parser.add_argument("--api-key", type=api_key, default=os.environ.get("BLINK_API_KEY", ""))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    local = Path(args.model).is_dir()
    if local:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
            os.environ.setdefault(name, "1")
        root = Path(args.model).resolve()
    else:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(args.model, revision=args.revision))
    verified = verify_weights(root)
    quant = detect_quantization(root, args.quantization)
    if quant and not verified:
        raise ValueError("a quantized checkpoint requires a verified weights.sha256")
    blink = load_blink(root)
    require_image_contract(blink)
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    renderer = blink.TorchEngine.__new__(blink.TorchEngine)
    renderer.tok = AutoTokenizer.from_pretrained(root, local_files_only=local)
    renderer.labels, renderer.label_ids = renderer._verify_labels()
    opts = {
        "model": str(root), "tokenizer": str(root), "dtype": "bfloat16",
        "seed": 0, "max_model_len": args.max_model_len,
        "max_logprobs": 255, "logprobs_mode": "processed_logprobs",
        "hf_overrides": {"head_dtype": "float32"},
        "enable_prefix_caching": args.prefix_cache,
        "mamba_cache_mode": "align" if args.prefix_cache else "none",
        "enable_chunked_prefill": not args.no_chunked_prefill,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    if quant:
        opts["quantization"] = quant
    if has_vision_model(root):
        opts["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    worker = threading.Thread(target=run_loop, name="blink-vllm-async", daemon=True)
    worker.start()
    owner = EngineOwner()
    stopping = threading.Event()

    def terminate(_signum, _frame):
        if stopping.is_set():
            return
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)

    async def initialize():
        try:
            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**opts))
            owner.adopt(engine)
            effective = engine.vllm_config
            if str(effective.model_config.head_dtype) != "torch.float32":
                raise RuntimeError("vLLM did not retain the FP32 offered-label head")
            if effective.model_config.max_model_len != args.max_model_len:
                raise RuntimeError("vLLM did not retain the requested max-model-len")
            if effective.cache_config.enable_prefix_caching != args.prefix_cache:
                raise RuntimeError("vLLM did not retain the requested prefix-cache setting")
            if effective.scheduler_config.enable_chunked_prefill != (not args.no_chunked_prefill):
                raise RuntimeError("vLLM did not retain the requested chunked-prefill setting")
            scorer = VllmScorer(blink, renderer, engine, SamplingParams, TokensPrompt,
                                max_model_len=args.max_model_len)
            first = await scorer.decide(*WARMUP)
            repeat = await scorer.decide(*WARMUP)
            return engine, scorer, first["answers"] == repeat["answers"]
        except BaseException:
            owner.close()
            raise

    scheduled = None
    try:
        startup, scheduled = submit_initializer(loop, initialize)
        _engine, scorer, repeated = startup.result(timeout=600)
        versions = {}
        for package in ("torch", "transformers", "vllm", "compressed-tensors"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        health = {
            "ok": True, "model": args.model, "revision": args.revision,
            "weights_verified": verified, "hub_offline": bool(local),
            "warmup": {"repeat_identical": repeated},
            "kernels": "vLLM processed masked logprobs; FP32 offered-label head",
            "versions": versions, "quantization": quant or "none",
            "accepts_images": False,
            "batching": {
                "max_concurrency": args.max_concurrency,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "chunked_prefill": not args.no_chunked_prefill,
                "prefix_cache": args.prefix_cache,
                "tensor_parallel_size": args.tensor_parallel_size,
                "limit_mm_per_prompt": opts.get("limit_mm_per_prompt"),
            },
            "api_key_required": args.api_key is not None,
        }
        handler = handler_for(blink, scorer, loop, args.model, health, args.api_key, args.max_concurrency)
        server = BoundedHTTPServer((args.host, args.port), handler, args.max_concurrency * 2 + 16)
        server.daemon_threads = True
        try:
            print(f"blink vLLM serving {args.model} on http://{args.host}:{args.port}", flush=True)
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        stopping.set()
        owner.abandon()
        try:
            if scheduled is not None:
                task = scheduled.result()
                if not task.done():
                    loop.call_soon_threadsafe(task.cancel)

                async def settle():
                    return await asyncio.gather(task, return_exceptions=True)

                results = asyncio.run_coroutine_threadsafe(settle(), loop).result()
                if isinstance(results[0], BaseException) and not isinstance(results[0], asyncio.CancelledError):
                    print(f"vLLM initialization ended: {type(results[0]).__name__}: {results[0]}",
                          file=sys.stderr, flush=True)
        finally:
            try:
                try:
                    owner.close()
                finally:
                    loop.call_soon_threadsafe(loop.stop)
                    worker.join()
                    loop.close()
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
