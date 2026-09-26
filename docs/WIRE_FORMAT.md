# TypeSafe API wire format

blink's v1.2 server speaks TypeSafe's API. Its Python and JavaScript SDKs work from server-side code by changing the base URL.

## Endpoints

| Method | Path | Use |
| --- | --- | --- |
| POST | `/v1/systemone` | Answer questions |
| GET | `/v1/models` | List the served model |
| GET | `/healthz` | Report startup checks |

## Run a server

```sh
pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
hf download thegovind/blink-4b --revision v1.2 --local-dir blink-4b
python blink-4b/serve.py --model ./blink-4b --port 8000
```

Set `TYPESAFE_BASE_URL` to the server address. TypeSafe's SDKs also require `TYPESAFE_API_KEY`. Any value works while the server is open; use the server key if you set one. The SDKs default to a 10-second timeout, so long documents may need more time.

### Python

```python
# pip install typesafe-sdk
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
print(r.choices["queue"].choice, r.nouls["urgent"].noul, r.scores["urgency"].score)
```

### JavaScript

```js
// npm install @typesafe-ai/sdk
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
console.log(answers.queue.choice, answers.urgent.noul, answers.urgency.score);
```

### Plain HTTP

```sh
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "model": "jev-latest",
  "state": "My card was charged twice this month.",
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Which team should own this ticket?",
      "criteria": {"billing": "Charges and refunds", "technical": "Errors and outages"}
    },
    "urgent": {"type": "noul", "instructions": "Does this need an answer today?"}
  }
}
EOF
```

## Request

| Field | Use |
| --- | --- |
| `state` | Text or JSON to judge |
| `model` | Accepted but ignored; the server uses its one model |
| `questions` | Typed questions, keyed by name |

| Question type | `criteria` |
| --- | --- |
| `noul` | Optional `{"true": "...", "false": "..."}` for yes/no wording |
| `choice` | A list of option keys or a `{key: description}` map |
| `score` | An ordered list of 2 to 10 level labels |

```json
{
  "model": "jev-latest",
  "state": "From: contact at example dot test\nSubject: Overdue invoice 88213 — final notice\n\nOur records show invoice 88213 for $4,180 is 21 days overdue. Wire the balance to the updated account below today to avoid suspension. Account details changed this quarter; use the new IBAN, not the one on the original invoice.",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "What is the sender trying to get the reader to do?",
      "criteria": {
        "pay_invoice": "Pay money against an invoice or bill",
        "share_credentials": "Hand over a password, code, or login",
        "book_meeting": "Agree to a call or meeting",
        "no_action": "Nothing; the message is informational"
      }
    },
    "suspicious": {
      "type": "noul",
      "instructions": "Does this message show signs of payment fraud?",
      "criteria": {
        "true": "Pressure, deadlines, or changed payment details",
        "false": "Ordinary correspondence with no fraud signals"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How quickly does a human need to look at this?",
      "criteria": ["No action needed", "This week", "Today", "Within the hour", "Immediately"]
    }
  }
}
```

## Response

| Question type | TypeSafe answer fields | blink also returns |
| --- | --- | --- |
| `noul` | `noul`, the probability of yes | `probabilities` |
| `choice` | `choice`, `probabilities`, `confidence` | — |
| `score` | `score`, the expected level; `legend`, `probabilities`, `confidence` | `choice`, the likeliest level |

`usage` contains `input_tokens` and `output_tokens`. Answers name the model that served them.

```json
{
  "model": "./blink-4b",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "pay_invoice",
      "probabilities": {
        "pay_invoice": 0.998,
        "share_credentials": 0.001,
        "book_meeting": 0.0,
        "no_action": 0.001
      },
      "confidence": 0.998
    },
    "suspicious": {"type": "noul", "noul": 0.992, "probabilities": {"yes": 0.992, "no": 0.008}},
    "urgency": {
      "type": "score",
      "score": 2.206,
      "probabilities": {"0": 0.004, "1": 0.04, "2": 0.745, "3": 0.17, "4": 0.041},
      "legend": {
        "0": "No action needed",
        "1": "This week",
        "2": "Today",
        "3": "Within the hour",
        "4": "Immediately"
      },
      "choice": "2",
      "confidence": 0.682
    }
  },
  "usage": {"input_tokens": 668, "output_tokens": 0}
}
```

## Model list

```sh
curl -s http://127.0.0.1:8000/v1/models
```

```json
{
  "models": [
    {
      "name": "./blink-4b",
      "description": "blink: typed decisions (noul, choice, score) with option probabilities from one forward pass. This server serves one model; a request's model field is accepted and not used.",
      "release_date": ""
    }
  ]
}
```

## Auth

The server is open by default. Set `--api-key` or `BLINK_API_KEY` to require `Authorization: Bearer <key>` on `/v1/systemone` and `/v1/models`. Without the right key, these routes return 401. `/healthz` stays open.

```json
{
  "error": "missing or invalid API key: send Authorization: Bearer <key>",
  "detail": "missing or invalid API key: send Authorization: Bearer <key>"
}
```

## Errors

| Status | Reason |
| --- | --- |
| 400 | Body is not a JSON object |
| 401 | Missing or wrong key when a key is set |
| 404 | Unknown path |
| 422 | Unsupported question or over a limit; the reason says which |
| 500 | Unexpected server error |
| 529 | Batching queue full; retry after `Retry-After`. v1.1 used 503 |

Error bodies give the reason in `error` and again in `detail`.

```json
{
  "error": "a score takes 2 to 10 levels",
  "detail": [
    {
      "loc": ["body", "questions", "urgency"],
      "msg": "a score takes 2 to 10 levels",
      "type": "value_error"
    }
  ]
}
```

## Limits

| Limit | Value |
| --- | ---: |
| Choice options | 255 |
| Score levels | 2 to 10 |
| Tokens per question | 131,072 |
| Questions per request | 512 |

Over a limit returns 422. Nothing is cut.

## Batching

| Flag | Default | Use |
| --- | ---: | --- |
| `--batch-window-ms` | 0 | Collection window; set to 5 for 5 ms |
| `--max-batch-requests` | 16 | Requests per batch |
| `--max-queued-requests` | 64 | Requests waiting for a batch |

At 0, requests run one at a time. The collection window starts when the server picks the first waiting request. A full queue returns 529 with `Retry-After`, which TypeSafe's SDKs retry.

## What's different from TypeSafe

- One model is served per server. The `model` field is ignored, and answers name the served blink model.
- `usage.input_tokens` counts each question's full prompt, including `state` each time. `output_tokens` is always 0.
- `noul` answers add `probabilities`. `score` answers add `choice`, the likeliest level.
- blink computes `confidence` as `(K*p_max-1)/(K-1)`. TypeSafe's docs use this formula to illustrate confidence.
- There are no rate limits or 429 responses. One-level scores return 422.
- Cross-origin browser calls are unsupported; the server sends no CORS headers.
- Responses from `/v1/systemone` and `/v1/models` carry an `x-typesafe-request-id` header.
- The model list has one entry with a blank `release_date`.

## This Space

The Space is not a TypeSafe endpoint. Its Gradio API takes the same request fields and returns TypeSafe's answer fields plus `meta`. Use `gradio_client` or the Gradio API over HTTP.

```python
# pip install gradio_client
from gradio_client import Client

client = Client("thegovind/blink")  # Client("thegovind/blink", token="hf_...") uses your own daily limit
r = client.predict(
    state="My card was charged twice this month.",
    questions={"urgent": {"type": "noul", "instructions": "Does this need an answer today?"}},
    model="thegovind/blink-4b",
    api_name="/v1_systemone",
)
print(r["answers"]["urgent"]["noul"])
```

Names beginning with `jev-` use the default model, `thegovind/blink-4b`. You can select `thegovind/blink-mimo-9b` instead. Anonymous calls have a lower daily limit. Pass a Hugging Face token to use your own quota.
