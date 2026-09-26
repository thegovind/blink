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

## The Space

Send the same fields to the Space with `gradio_client`. The Space is not a TypeSafe endpoint.

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
