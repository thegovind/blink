---
name: blink
license: Apache-2.0
description: >
  Use blink for one-pass typed decisions (yes/no, pick one, 1-N score) with
  probabilities over text or JSON state; it is cheaper and faster than prompting
  an LLM and parsing its text; wire-compatible with TypeSafe's API.
---

# blink

Use blink when an agent needs a fixed decision and probabilities, not prose. Send text or JSON state with named questions. blink answers every option in one forward pass and generates no text.

Use:

- `noul` for yes/no. Read the result as the probability of yes.
- `choice` to pick one of up to 255 options. Read the pick, probabilities, and confidence.
- `score` for 2-10 ordered levels. Read the expected level, legend, probabilities, and confidence.

Do not use blink for writing, rewriting, summarizing, explaining, or any task that needs generated text.

## Call blink

For a self-hosted server, use a server-side TypeSafe Python or JavaScript SDK. Set `TYPESAFE_BASE_URL` to the server URL and `TYPESAFE_API_KEY` to any value, or to the server's key. You can also send plain HTTP to `POST /v1/systemone`.

For a quick try, call the [Space](https://huggingface.co/spaces/thegovind/blink) with `gradio_client`. The Space accepts the same fields through its Gradio API; it is not a TypeSafe endpoint.

## Request and answer shapes

Questions are keyed by name. `model` is accepted and ignored.

```json
{
  "state": {"ticket": "Customer cannot sign in and needs access today."},
  "model": "any",
  "questions": {
    "needs_reply": {
      "type": "noul",
      "instructions": "Does this need a reply today?"
    },
    "owner": {
      "type": "choice",
      "instructions": "Who should handle this?",
      "criteria": {"account": "Account team", "support": "Support team"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["low", "medium", "high"]
    }
  }
}
```

All numeric values in this minimal answer example are illustrative.

```json
{
  "answers": {
    "needs_reply": {"noul": 0.82},
    "owner": {
      "choice": "support",
      "probabilities": {"account": 0.25, "support": 0.75},
      "confidence": 0.50
    },
    "urgency": {
      "score": 1.4,
      "legend": {"0": "low", "1": "medium", "2": "high"},
      "probabilities": {"0": 0.10, "1": 0.40, "2": 0.50},
      "confidence": 0.25
    }
  }
}
```

## Write good questions

- Use clear instructions. Ask for one decision at a time.
- Keep choice options and score levels short, distinct, and complete.
- Include only the state that matters to the decision.

## Read answers

Probabilities show the distribution over every option. Confidence is returned for `choice` and `score`. For `score`, `score` is the expected level, not generated text.

Limits: 255 options, 2-10 score levels, 131,072 tokens per question, and 512 questions per request; input is not cut.

Errors: `400` bad JSON object, `401` missing required key, `404` unknown path, `422` unsupported question or exceeded limit, `500` server error, and `529` full batching queue with `Retry-After`.

## Links

- [Docs](https://thegovind.github.io/blink/)
- [Agent docs index](https://thegovind.github.io/blink/llms.txt)
- [API reference](https://thegovind.github.io/blink/api.md)
