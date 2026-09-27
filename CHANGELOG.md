# Changelog

## v1.4.0 - 2026-09-27

- Opt-in, text-only `serve_vllm.py` for blink-4b only. `serve.py` stays the default.
- [Serving docs](https://thegovind.github.io/blink/models.md#serve-it) cover latency,
  throughput and quality limits. `blink.py`, `serve.py` and the weights are unchanged.
- The MiMo and 27B vLLM paths failed quality checks. All tested INT8 builds failed too;
  none of those paths ships.

Code: Apache-2.0. Weights: non-commercial research and evaluation only; see each model card's license.

## v1.3.0 - 2026-09-27

- Opt-in screenshots for self-hosted `/v1/systemone`; text-only requests keep the byte-identical pre-image
  prompt and numeric path.
- Send PNG, JPEG, or WebP data URIs in a `state` string or top-level `images` list. Image URLs are not
  fetched. Detected but unsupported headers and invalid image submissions return `422`; bare MIME text
  and undetected headers in `state` stay text (see the [API rule](https://thegovind.github.io/blink/api.md#screenshots-opt-in)).
- Use `--vision` for blink-mimo-9b or a matched `--vision-tower` for blink-4b and blink-27b;
  `--image-layout first` is the default. Use `--model-name` for renamed folders.
- Default limits: 2 images, 8 MiB decoded each, 20 MP source, 2,088,960 pixels after resize;
  adjustable with `--max-images`, `--max-image-bytes`, `--max-image-source-pixels`, and `--max-image-pixels`.

Code: Apache-2.0. Weights: non-commercial research and evaluation only; see each model card's license.

## v1.2.0 - 2026-09-26

- `serve.py` supports TypeSafe's Python and JavaScript SDKs from server-side code by changing the base URL. It adds
  `GET /v1/models`, optional API keys, `error` and `detail` in error bodies, and `x-typesafe-request-id` on API responses.
- Score answers now include `confidence`.
- A full batching queue now returns HTTP 529 with `Retry-After`, rather than the 503 returned by `v1.1`.
- `thegovind/blink-27b`, `thegovind/blink-mimo-9b`, and `thegovind/blink-4b` have this code at revision `v1.2`. Weights
  are unchanged from `v1.0`.
- The Space has an API tab, and the docs site is live at `thegovind.github.io/blink`.

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
