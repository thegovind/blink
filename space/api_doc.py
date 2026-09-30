"""The API tab: using blink with code written for the TypeSafe API (Jev), and calling this Space.

Every user-facing string of the tab is in this module (the final copy); ui.api_blocks() only arranges it. The GitHub
and Docs links beside the steps are the ones every tab shows (ui.PROJECT_LINKS). DRAFT = True shows COPY["draft"] as
a tag beside the heading, for while copy is being reworked.

The examples are exact: test_blink.TestApiTab checks that EXAMPLE_RESPONSE is blink-4b's saved answer to
EXAMPLE_REQUEST (rounded as shown), that every request is one blink accepts, that the error bodies are the ones blink
returns and that every download pins CODE_REVISION; release/test_serve.py sends the same examples to serve.py over HTTP.
"""

from __future__ import annotations

import json

import examples

DRAFT = False

SERVER_URL = "http://127.0.0.1:8000"
SPACE_ID = "thegovind/blink"
SPACE_URL = "https://thegovind-blink.hf.space"
WIRE_FORMAT_URL = "https://thegovind.github.io/blink/api/"  # the docs site's full reference for this tab
SERVED_AS = "./blink-4b"  # what the answers name as the model after `serve.py --model ./blink-4b`
CODE_REVISION = "v1.4"  # the model repos' code revision with this serve.py and blink.py

COPY = {
    "heading": "Use blink with TypeSafe clients",
    "draft": "",
    "lede": "Use TypeSafe's Python or JavaScript SDK from server-side code against a blink server. "
            "Core request and answer fields stay the same.",
    "wire_docs": "API docs",
    "table_label": "TypeSafe API at a glance",
    "legend": (("same", "Same"), ("adds", "Adds fields"), ("differs", "Different"), ("none", "Not offered")),
    "steps_label": "Three steps",
    "steps": (
        ("Run a blink server", "python blink-4b/serve.py --model ./blink-4b --port 8000"),
        ("Set the base URL", "export TYPESAFE_BASE_URL=http://127.0.0.1:8000"),
        ("Call it as usual", "client.system_one(state=..., questions=...)"),
    ),
    "server": "Run a blink server",
    "server_notes": (
        "One server serves one model. The model field does not switch it. Answers name the blink model.",
        "Check health: curl -s http://127.0.0.1:8000/healthz",
    ),
    "key": "API key",
    "key_notes": (
        "The server is open by default. Set --api-key or BLINK_API_KEY to require a key.",
        "Both /v1/systemone and /v1/models then require Authorization: Bearer <key>. A missing or "
        "wrong key gets 401. /healthz stays open.",
    ),
    "batching": "Batching",
    "batching_notes": (
        "Off by default: one request at a time.",
        "Set --batch-window-ms 5 for a 5 ms collection window, with --max-batch-requests capping each "
        "batch. A full queue returns 529 with Retry-After, which TypeSafe's SDKs retry.",
        "Image requests run separately, including multiple questions about the same image.",
    ),
    "client": "Point a TypeSafe client at it",
    "client_notes": (
        "Set TYPESAFE_BASE_URL to your server and TYPESAFE_API_KEY to any value if the server is open. "
        "Use your server key if you set one.",
        "The SDKs default to a 10-second timeout. Long documents may need more time.",
    ),
    "python": "Python",
    "javascript": "JavaScript",
    "http": "HTTP",
    "example": "Request and response",
    "example_note": "Example blink-4b answer. Numbers rounded to three decimals.",
    "fields_label": "Answer fields",
    "fields": (
        ("noul", "noul: probability of yes", "probabilities"),
        ("choice", "choice, probabilities, confidence", ""),
        ("score", "score: expected level, legend, probabilities, confidence", "choice: likeliest level"),
    ),
    "fields_cols": ("Type", "TypeSafe fields", "Also returned"),
    "errors": "Errors and limits",
    "errors_cols": ("Status", "When"),
    "error_rows": (
        ("400", "The body isn't a JSON object."),
        ("401", "Missing or wrong key when one is set."),
        ("404", "Unknown path."),
        ("422", "Unsupported question or an over-limit request. The error names the reason."),
        ("500", "An unexpected server error."),
        ("529", "The batching queue is full. Retry using Retry-After."),
    ),
    "error_note": "No 429: the server has no rate limit. Error bodies repeat the reason in error and "
                  "detail.",
    "limits": "255 choice options · 2 to 10 score levels · 131,072 tokens per question · 512 questions "
              "per request. Images (self-host opt-in): 2 per request, 8 MiB each, at most 2,088,960 resized pixels. "
              "Over a limit gets 422.",
    "models": "Model list",
    "space": "Call this Space",
    "mlx": "Run it on a Mac (MLX)",
    "mlx_notes": (
        "blink_mlx.py keeps the model repo's blink.py for prompts, option labels and answers. MLX runs the "
        "forward pass on Apple silicon.",
        "Text and JSON state only, bf16 weights only. Answers have the same fields as blink.decide.",
        "It was checked on Linux. Run --check once on your Mac: it compares your answers with this Space's "
        "saved runs for blink-4b and blink-mimo-9b.",
        "It is not a server. TypeSafe clients need serve.py.",
    ),
    "mlx_docs": "MLX guide: memory, checks and limits",
    "space_notes": (
        "This Space is not a TypeSafe endpoint. Its Gradio API takes the same request fields and "
        "returns TypeSafe's answer fields plus meta. Use gradio_client or the Gradio API over HTTP.",
        "model defaults to thegovind/blink-4b. Use thegovind/blink-mimo-9b to switch. TypeSafe names "
        "like jev-latest use the default.",
        "Anonymous calls have a lower daily limit. Pass a Hugging Face token to use your own quota.",
        "Plain HTTP data lists all four values in order: state, questions, temperature, model.",
    ),
    "diff": "What's different",
    "diff_notes": (
        "The server ignores model and names its one blink model in the answer.",
        "usage.input_tokens counts the full prompt for every question, including state each time. "
        "output_tokens is always 0.",
        "noul answers add probabilities for yes and no. score answers add choice, the likeliest level.",
        "For score answers, blink computes confidence as (K × p_max − 1) / (K − 1), the formula "
        "TypeSafe's docs use to illustrate confidence.",
        "Score legends are text, including structured level descriptions as JSON text.",
        "A score needs at least two levels; a one-level score gets 422.",
        "The model list has one entry, the served model, with a blank release_date.",
        "The model list and health report accepts_images; it is false unless image support is enabled.",
        "Image requests alone add image_pixels and visual_tokens to usage; these are self-host extensions.",
        "Responses from /v1/systemone and /v1/models carry an x-typesafe-request-id header.",
        "Cross-origin browser calls are unsupported; the server sends no CORS headers.",
    ),
}

# (row, TypeSafe API, (mark, blink server), (mark, this Space)); marks are the legend's keys
TABLE_COLS = ("", "TypeSafe API", "blink server", "This Space")
TABLE_ROWS = (
    (
        "Endpoint",
        "POST /v1/systemone",
        ("same", "POST /v1/systemone"),
        ("differs", "Gradio API /v1_systemone"),
    ),
    (
        "Base URL",
        "api.typesafe.ai",
        ("differs", "Your server's address"),
        ("differs", "thegovind-blink.hf.space"),
    ),
    ("API key", "Required", ("differs", "Optional"), ("differs", "Hugging Face token, optional")),
    (
        "Request",
        "state, model, questions",
        ("same", "state, model, questions"),
        ("same", "state, questions, model"),
    ),
    ("noul", "noul", ("adds", "noul + probabilities"), ("adds", "noul + probabilities")),
    (
        "choice",
        "choice, probabilities, confidence",
        ("same", "choice, probabilities, confidence"),
        ("same", "choice, probabilities, confidence"),
    ),
    (
        "score",
        "score, legend, probabilities, confidence",
        ("adds", "the same + choice"),
        ("adds", "the same + choice"),
    ),
    (
        "usage",
        "input_tokens, output_tokens",
        ("same", "input_tokens, output_tokens"),
        ("adds", "the same + meta"),
    ),
    ("Model list", "GET /v1/models", ("same", "GET /v1/models"), ("none", "—")),
    (
        "Errors",
        "401, 422, 429, 529",
        ("differs", "401, 422, 529"),
        ("differs", "A Gradio error with the reason"),
    ),
    (
        "SDKs",
        "Python, JavaScript",
        ("same", "Set TYPESAFE_BASE_URL and TYPESAFE_API_KEY"),
        ("differs", "gradio_client or Gradio API over HTTP"),
    ),
)

PIP = ('pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" '
       '"accelerate>=1.1.0" safetensors huggingface_hub')
SERVER_RUN = f"""{PIP}
hf download thegovind/blink-4b --revision {CODE_REVISION} --local-dir blink-4b
python blink-4b/serve.py --model ./blink-4b --port 8000"""

DOCKER_RUN = """cd blink-4b
docker build -t blink-4b . && docker run --rm --gpus all -p 127.0.0.1:8000:8000 blink-4b"""

MLX_URL = "https://github.com/thegovind/blink/tree/main/examples/mlx"
MLX_RUN = """pip install "mlx-lm>=0.31.3"
curl -O https://raw.githubusercontent.com/thegovind/blink/main/examples/mlx/blink_mlx.py
python blink_mlx.py --model thegovind/blink-4b
python blink_mlx.py --check   # compare with the Space's saved runs"""

KEY_RUN = """BLINK_API_KEY=your-key python blink-4b/serve.py --model ./blink-4b --port 8000
# Docker: docker run --rm --gpus all -e BLINK_API_KEY=your-key -p 127.0.0.1:8000:8000 blink-4b"""

BATCH_RUN = """python blink-4b/serve.py --model ./blink-4b --port 8000 --batch-window-ms 5
# Docker: docker run --rm --gpus all -e BLINK_BATCH_WINDOW_MS=5 -p 127.0.0.1:8000:8000 blink-4b"""

CLIENT_ENV = f"""export TYPESAFE_BASE_URL={SERVER_URL}
export TYPESAFE_API_KEY=any-value   # or the key the server checks"""

PYTHON_CLIENT = """# pip install typesafe-sdk
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(timeout=60) as client:
    r = client.system_one(
        state="My card was charged twice this month.",
        questions={
            "queue": Choice(
                instructions="Which team should own this ticket?",
                criteria={"billing": "Charges and refunds", "technical": "Errors and outages"},
            ),
            "urgent": Noul(instructions="Does this need an answer today?"),
            "urgency": Score(instructions="How urgent is it?", criteria=["Can wait", "This week", "Today"]),
        },
    )
print(r.choices["queue"].choice, r.nouls["urgent"].noul, r.scores["urgency"].score)"""

JS_CLIENT = """// npm install @typesafe-ai/sdk
import { TypeSafeClient, choice, noul, score } from "@typesafe-ai/sdk";

const client = new TypeSafeClient({ timeout: 60_000 });
const { answers } = await client.systemOne({
  state: "My card was charged twice this month.",
  questions: {
    queue: choice("Which team should own this ticket?", {
      billing: "Charges and refunds",
      technical: "Errors and outages",
    }),
    urgent: noul("Does this need an answer today?"),
    urgency: score("How urgent is it?", ["Can wait", "This week", "Today"]),
  },
});
console.log(answers.queue.choice, answers.urgent.noul, answers.urgency.score);"""

# TypeSafe's request shape around the Playground's invoice example, which has a saved blink-4b answer
EXAMPLE_REQUEST = {
    "model": "jev-latest",
    "state": examples.PLAYGROUND_STATE,
    "questions": json.loads(examples.PLAYGROUND_QUESTIONS),
}

# blink-4b's saved answer to EXAMPLE_REQUEST (checked in test_blink), rounded to three decimals
EXAMPLE_RESPONSE = {
    "model": SERVED_AS,
    "answers": {
        "intent": {
            "type": "choice",
            "choice": "pay_invoice",
            "probabilities": {"pay_invoice": 0.998, "share_credentials": 0.001, "book_meeting": 0.0,
                              "no_action": 0.001},
            "confidence": 0.998,
        },
        "suspicious": {"type": "noul", "noul": 0.992, "probabilities": {"yes": 0.992, "no": 0.008}},
        "urgency": {
            "type": "score",
            "score": 2.206,
            "probabilities": {"0": 0.004, "1": 0.04, "2": 0.745, "3": 0.17, "4": 0.041},
            "legend": {"0": "No action needed", "1": "This week", "2": "Today", "3": "Within the hour",
                       "4": "Immediately"},
            "choice": "2",
            "confidence": 0.682,
        },
    },
    "usage": {"input_tokens": 668, "output_tokens": 0},
}


def pretty(obj, width: int = 96) -> str:
    """JSON indented by two spaces, with any object or list that fits in `width` kept on one line."""

    def dump(v) -> str:
        return json.dumps(v, ensure_ascii=False)

    def fmt(v, indent: int, lead: int) -> str:
        flat = dump(v)
        if not isinstance(v, (dict, list)) or not v or lead + len(flat) < width:
            return flat
        pad = " " * (indent + 2)
        if isinstance(v, dict):
            parts = [pad + dump(k) + ": " + fmt(x, indent + 2, len(pad) + len(dump(k)) + 2) for k, x in v.items()]
            return "{\n" + ",\n".join(parts) + "\n" + " " * indent + "}"
        parts = [pad + fmt(x, indent + 2, len(pad)) for x in v]
        return "[\n" + ",\n".join(parts) + "\n" + " " * indent + "]"

    return fmt(obj, 0, 0)


def curl_request(body: dict) -> str:
    return (f"curl -s {SERVER_URL}/v1/systemone \\\n"
            '  -H "Authorization: Bearer $TYPESAFE_API_KEY" \\\n'
            '  -H "Content-Type: application/json" \\\n'
            f"  -d @- <<'EOF'\n{pretty(body)}\nEOF")


SHORT_REQUEST = {
    "model": "jev-latest",
    "state": "My card was charged twice this month.",
    "questions": {
        "queue": {"type": "choice", "instructions": "Which team should own this ticket?",
                  "criteria": {"billing": "Charges and refunds", "technical": "Errors and outages"}},
        "urgent": {"type": "noul", "instructions": "Does this need an answer today?"},
    },
}

HTTP_CLIENT = curl_request(SHORT_REQUEST)

# A refused request and the body serve.py returns for it (checked in test_blink and test_serve)
REFUSED_REQUEST = {
    "model": "jev-latest",
    "state": "Our records show invoice 88213 is overdue.",
    "questions": {"urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["Today"]}},
}
REFUSED_BODY = {
    "error": "a score takes 2 to 10 levels",
    "detail": [{"loc": ["body", "questions", "urgency"], "msg": "a score takes 2 to 10 levels",
                "type": "value_error"}],
}
UNAUTHORIZED_BODY = {
    "error": "missing or invalid API key: send Authorization: Bearer <key>",
    "detail": "missing or invalid API key: send Authorization: Bearer <key>",
}

MODELS_CALL = f"curl -s {SERVER_URL}/v1/models"
MODELS_BODY = {
    "models": [{
        "name": SERVED_AS,
        "description": "blink: typed decisions (noul, choice, score) with option probabilities from one forward "
                       "pass. This server serves one model; a request's model field is accepted and not used.",
        "release_date": "",
        "accepts_images": False,
    }],
}

SPACE_STATE = "My card was charged twice this month."
SPACE_QUESTIONS = {"urgent": {"type": "noul", "instructions": "Does this need an answer today?"}}

SPACE_PYTHON = f"""# pip install gradio_client
from gradio_client import Client

client = Client("{SPACE_ID}")  # Client("{SPACE_ID}", token="hf_...") uses your own daily limit
r = client.predict(
    state="{SPACE_STATE}",
    questions={json.dumps(SPACE_QUESTIONS)},
    model="thegovind/blink-4b",
    api_name="/v1_systemone",
)
print(r["answers"]["urgent"]["noul"])"""

SPACE_DATA = [SPACE_STATE, SPACE_QUESTIONS, None, "thegovind/blink-4b"]

SPACE_CURL = f"""URL={SPACE_URL}/gradio_api/call/v1_systemone
ID=$(curl -s -X POST $URL -H "Content-Type: application/json" \\
  -d '{json.dumps({"data": SPACE_DATA})}' | cut -d'"' -f4)
curl -s -N $URL/$ID"""

# the live Space's reply to SPACE_DATA (2026-09-26, blink-4b), numbers rounded to three decimals like the tab's
# other examples; the timings are one warm call's and are marked as varying
SPACE_REPLY = """event: complete
data: [{"model": "thegovind/blink-4b", "answers": {"urgent": {"type": "noul", "noul": 0.328, "probabilities": {"yes": 0.328, "no": 0.672}}}, "usage": {"input_tokens": 98, "output_tokens": 0}, "meta": {"model": "thegovind/blink-4b", "engine": "torch", "temperature": 1.0, "input_tokens": 98, "generated_tokens": 0, "latency_ms": 262.7, "model_ms": 84.7, "prefill_tokens": 98}}]
# latency_ms and model_ms vary from call to call"""
