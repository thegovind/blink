# Changelog

## v1.1.0 - 2026-09-26

- `serve.py` adds opt-in cross-request batching with `--batch-window-ms`, off by default at 0.
  `--max-batch-requests` defaults to 16 and `--max-queued-requests` to 64. Requests beyond the queue
  limit get HTTP 503 with `Retry-After`.
- Serving is unchanged with the flag unset. An error in one request does not affect others in its batch.
- New tests cover batching, including HTTP-level checks.
- `thegovind/blink-4b` and `thegovind/blink-mimo-9b` have this code at revision `v1.1`. Their weights are
  unchanged from `v1.0`.
- The model cards now report local, descriptive Decision Index 0.2 runs, not accepted leaderboard results.
  Known training exposure remains without the leaderboard's penalty, so the scores cannot be ranked against it.

## v1.0.0 - 2026-09-25

- Initial public-quality source package for blink.
- Includes runtime, HTTP server, Space demo, training utilities, evaluation
  helpers, tests, docs, model cards, diagrams, and preregistration records.
- Links to v1.0 model weights on Hugging Face; weights are not included.
