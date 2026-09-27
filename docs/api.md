# API

Point a TypeSafe SDK or plain HTTP client at blink's server. It is wire-compatible with TypeSafe's API.

## Endpoints

| Method | Path | Use |
|---|---|---|
| `POST` | `/v1/systemone` | Answer typed questions |
| `GET` | `/v1/models` | Return the one served model |
| `GET` | `/healthz` | Check server health |

## Point TypeSafe SDKs at blink

Set these variables in server-side Python or JavaScript:

| Variable | Value |
|---|---|
| `TYPESAFE_BASE_URL` | The blink server URL |
| `TYPESAFE_API_KEY` | Any value, or the server's configured key |

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

| Field | Value |
|---|---|
| `state` | Text or JSON |
| `model` | Accepted and ignored |
| `questions` | Questions keyed by name |

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

Read answers by question name.

| Type | Answer fields |
|---|---|
| `noul` | Probability of yes |
| `choice` | Picked option, option probabilities, confidence |
| `score` | Expected level, legend, level probabilities, confidence |

Probabilities are over the offered options only, not certified chances of being right.

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

## Errors and limits

| Status | Meaning |
|---|---|
| `400` | The body isn't a JSON object (or isn't valid JSON) |
| `401` | The server has an API key and the request's key is missing or wrong |
| `404` | Path not found |
| `422` | Unsupported question or limit exceeded; the reason says which |
| `500` | Server error |
| `529` | Batching queue full; check `Retry-After` |

Limits: 255 options per choice, 2-10 score levels, 131,072 tokens per question, and 512 questions per request. Requests over these limits fail; blink does not cut input.

Set an optional key with `--api-key` or `BLINK_API_KEY`. Enable batching with `--batch-window-ms 5`.

## Screenshots (opt-in)

Image input is off by default and works only on a self-hosted server. TypeSafe's hosted Jev is text-only and its API has no image field. Text-only requests keep the byte-identical pre-image prompt and numeric path.

Put a `data:image/png;base64,...` URI (JPEG and WebP work too) in a string in `state`, or send data URIs in a top-level `images` list. Replace `<base64 PNG omitted>` with actual base64 before sending:

```json
{
  "state": "Screenshot: data:image/png;base64,<base64 PNG omitted>",
  "questions": {"save_visible": {"type": "noul", "instructions": "Is a Save button visible?"}}
}
```

Image mode needs `torchvision==0.28.0`; install it before starting the server. For downloaded 4B or 27B folders, cache the matching base model at the pinned revision below before serving; local-folder mode is offline. Download the full v1.3 model repository so `graft_keys.py` sits beside `blink.py`.

Enable screenshots with the flag for your model:
- `blink-mimo-9b`: `--vision` (its own tower).
- `blink-4b`: `--vision-tower Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- `blink-27b`: `--vision-tower Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

`--image-layout first` is the default and places images before the user text; `--image-layout inline` places them at image placeholders in the rendered state. Neither option enables images. For renamed folders or Docker's `/blink`, set `--model-name` to `blink-4b`, `blink-mimo-9b`, or `blink-27b`, matching the checkpoint.

Defaults: 2 images, 8 MiB decoded per image, 20 MP source, 2,088,960 pixels after resizing. Raise these with `--max-images`, `--max-image-bytes`, `--max-image-source-pixels`, and `--max-image-pixels`.

Image URLs are never fetched. In each string value in `state`, `data:image/` (any case) starts a check. The first comma after that prefix must fall within 256 characters (counting `d` through comma); the header before it must end in `;base64` after whitespace is removed.

| Input | Result |
|---|---|
| `data:image/png;base64,`, `data:image/jpeg;base64,` or `data:image/webp;base64,` (any case, no spaces or parameters) with valid data | Image |
| Other recognized headers (such as GIF, parameters or spaces after `data:image/`), or invalid image data | `422` |
| Bare MIME text or unrecognized headers in `state` (changed prefix, missing or late comma, missing `;base64`) | Text |
| Top-level `images` entry | Must be a complete supported data URI; otherwise `422` |

## The Space

For text-only requests, send `state` and `questions` to the demo Space with `gradio_client`. The demo Space does not serve screenshots and is not a TypeSafe endpoint.

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

## Not supported

- Cross-origin browser calls (no CORS).
- Rate limits and `429` responses.
- One-level scores.
