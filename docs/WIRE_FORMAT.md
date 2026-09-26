# Wire format

blink serves the TypeSafe/Jev-compatible endpoint:

`POST /v1/systemone`

Request:

```json
{
  "state": "text or JSON-like value",
  "questions": {
    "name": {
      "type": "choice",
      "instructions": "What should be decided?",
      "criteria": {"a": "First option", "b": "Second option"}
    }
  }
}
```

Question types:

- `choice`: `criteria` is an object or list with 1-255 options.
- `noul`: yes/no, with optional `criteria.true` and `criteria.false` labels.
- `score`: `criteria` is a list of 2-10 ordered levels.

Response:

```json
{
  "model": "thegovind/blink-4b",
  "answers": {
    "name": {
      "type": "choice",
      "choice": "a",
      "probabilities": {"a": 0.7, "b": 0.3},
      "confidence": 0.4
    }
  },
  "usage": {"input_tokens": 123, "output_tokens": 0}
}
```

`GET /healthz` reports startup checks, dependency versions, weight verification
status, and warm-up repeatability.
