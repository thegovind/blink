---
name: blink
license: Apache-2.0
description: >
  Use blink for typed decisions (yes/no, pick one, 2-10 level score) over text
  or JSON state. Each question gets probabilities over its offered options from
  one forward pass, with no generated text; use it instead of prompting an LLM
  and parsing its reply. Its self-hosted server is wire-compatible with
  TypeSafe's API.
---

# blink

Use blink for fixed decisions over text or JSON state. Each question gets a probability for every offered option from one forward pass; blink generates no text.

Probabilities are over the offered options only, not certified chances of being right.

| Need | Type | Read |
|---|---|---|
| Yes or no | `noul` | Probability of yes |
| Pick one | `choice` | Pick, probabilities, confidence |
| Rate ordered levels | `score` | Expected level, legend, probabilities, confidence |

Do not use blink for writing, rewriting, summarizing, explaining, or any task that needs generated text.

Code: Apache-2.0. Weights: non-commercial research and evaluation only.

## Call blink

Point a server-side TypeSafe Python or JavaScript SDK at a self-hosted server. Set `TYPESAFE_BASE_URL` to the server URL. Set `TYPESAFE_API_KEY` to any value or the server's key. Or send plain HTTP to `POST /v1/systemone`.

For screenshots on a self-hosted v1.3 or later server with vision enabled, send a PNG, JPEG, or WebP data URI in a `state` string or top-level `images` list; see the [API example](https://thegovind.github.io/blink/api.md#screenshots-servepy-only-opt-in).

For a quick text-only test, call the [Space](https://huggingface.co/spaces/thegovind/blink) with `gradio_client`. Send `state` and `questions` through its Gradio API. The demo Space does not serve screenshots and is not a TypeSafe endpoint.

## Request and answer shapes

Questions are keyed by name. `model` is accepted and ignored.

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

blink-4b's saved answer to this request, rounded to three decimals.

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

| Scope | Fields |
|---|---|
| Response | `model`, `answers`, `usage` |
| `noul` answer | `type`, `noul`, `probabilities` for `yes` and `no` |
| `choice` answer | `type`, `choice`, `probabilities`, `confidence` |
| `score` answer | `type`, `score`, `probabilities`, `legend`, `choice`, `confidence` |

## Write good questions

- Give clear instructions. Ask for one decision at a time.
- Keep choice options and score levels short, distinct, and complete.
- Include only the state that matters to the decision.

## Read answers

Read `noul` as the probability of yes. Read `choice` as the picked option. Use `probabilities` for the full distribution and `confidence` for `choice` and `score`. Read `score` as the expected level.

Default `serve.py` limits: 255 options per choice, 2-10 score levels, 131,072 tokens per question and 512 questions per request; input is never truncated. The 4B vLLM server was tested at a 32,768-token context; questions over that return `422`.

Errors: `400` the body isn't a JSON object (or isn't valid JSON), `401` the server has an API key and the request's key is missing or wrong, `404` unknown path, `422` unsupported question or exceeded limit, `500` server error, and `529` full batching queue with `Retry-After`.

## Links

- [Docs](https://thegovind.github.io/blink/)
- [Agent docs index](https://thegovind.github.io/blink/llms.txt)
- [API reference](https://thegovind.github.io/blink/api.md)
