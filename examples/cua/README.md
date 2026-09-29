# Computer-use scenarios

Ten original, offline web apps exercise blink as a screenshot-driven agent: shop, mail, settings, calendar, files, travel, careers, phone, desktop (canvas), and game (canvas lane dodge). The [contract](SPEC.md) defines each app's tasks and independent `check()`/`expected()` oracle. The harness serves the apps locally, marks numbered boxes with `space/screens.py`, sends typed questions to a self-hosted blink server, records the action and video, and writes a report. No app needs an external page or account.

## Start a server and run one scenario

Install Python 3.12, Playwright Chromium, Pillow, HTTPX and imageio-ffmpeg. From the repository root:

```sh
uv run --no-project --python 3.12 --with playwright --with pillow --with httpx \
  --with imageio-ffmpeg python -m playwright install chromium
```

Use the self-hosted `serve.py` screenshot path, not the text-only Space API or `serve_vllm.py`. For blink-mimo-9b:

```sh
python serve.py --model thegovind/blink-mimo-9b --vision --port 8000
```

For blink-4b or blink-27b, instead use the matching base vision tower at the pinned revision:

```sh
python serve.py --model thegovind/blink-4b \
  --vision-tower Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --port 8000
python serve.py --model thegovind/blink-27b \
  --vision-tower Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --port 8000
```

Run **one** of those commands, for the chosen model. The [screenshot setup](../../docs/api.md#screenshots-servepy-only-opt-in) covers the image dependency, downloaded folders and model-name override. Verify `/healthz` returns `accepts_images: true` before running the harness.

```sh
uv run --no-project --python 3.12 --with playwright --with pillow --with httpx \
  --with imageio-ffmpeg python -m examples.cua.harness.sweep \
  --scenario shop --task 0 --seeds 1 --model blink-mimo-9b \
  --server blink-mimo-9b=http://127.0.0.1:8000 \
  --runs-dir "$PWD/cua-runs" --run-id one-shop --workers 1
```

Substitute the model and URL for the server actually started. `--server model=URL` accepts an explicit HTTP or HTTPS URL; only send screenshots to a server under your control. The offline app itself is always served locally. For a source folder without `.git`, pass `--harness-sha` with the exact source revision. Runs have unique IDs; reusing an ID with a changed configuration is an error.

## Run a matched sweep

Start one image-capable server per model at the URLs supplied below, then run the same seeds and harness across all ten apps:

```sh
BLINK_CUA_SERVERS="blink-mimo-9b=http://127.0.0.1:8000,blink-4b=http://127.0.0.1:8001,blink-27b=http://127.0.0.1:8002" \
BLINK_CUA_RUNS_DIR="$PWD/cua-runs" \
uv run --no-project --python 3.12 --with playwright --with pillow --with httpx \
  --with imageio-ffmpeg python -m examples.cua.harness.sweep \
  --model blink-mimo-9b --model blink-4b --model blink-27b \
  --seeds 1-10 --workers 3 --run-id matched-1-10
```

`--server` overrides `BLINK_CUA_SERVERS`; `--runs-dir`, `--registry` and `--apps-root` override `BLINK_CUA_RUNS_DIR`, `BLINK_CUA_REGISTRY` and `BLINK_CUA_APPS_ROOT`. Repeat `--server` to distribute a model's jobs across servers. The default registry is `apps/registry.json`; `python examples/cua/apps/build_registry.py --check` verifies it against the ten `scenario.json` files.

Read `cua-runs/matched-1-10/report.md` for model and per-app rows (including failures and Wilson 95% intervals), `report.json` for denominators, and `results.jsonl` for every episode. Each episode folder has `summary.json`, `trace.jsonl`, numbered `steps/` screenshots and, unless disabled, `video.mp4` plus audit media. The `success` field comes from each app's `check()`, including forbidden side effects. `--mock` is an oracle-backed CPU check, **not** a model result; use it to verify setup before pointing at a server.

| Model | Completed tasks | Success |
|---|---:|---:|
| blink-27b | 199 / 230 | 87% (Wilson 95%: 82-90%) |
| blink-4b | 152 / 230 | 66% |
| blink-mimo-9b | 133 / 230 | 58% |

Matched zero-shot sweep over ten apps and seeds 1-10 using the same harness. Tasks done per app:

| App | blink-27b | blink-4b | blink-mimo-9b |
|---|---:|---:|---:|
| Shop | 30/30 | 29/30 | 6/30 |
| Settings | 30/30 | 30/30 | 30/30 |
| Files | 30/30 | 19/30 | 22/30 |
| Calendar | 20/20 | 10/20 | 10/20 |
| Phone | 20/20 | 10/20 | 11/20 |
| Desktop (canvas) | 18/20 | 17/20 | 15/20 |
| Mail | 22/30 | 22/30 | 19/30 |
| Careers | 14/20 | 9/20 | 10/20 |
| Travel | 11/20 | 5/20 | 8/20 |
| Game (canvas, turn-based) | 4/10 | 1/10 | 2/10 |

The game waits for each decision (turn-based). In real time, none of the nine runs cleared all 20 waves; the best lasted 14.

Videos, marked screenshots and per-step records for these runs are in the [thegovind/blink-cua](https://huggingface.co/datasets/thegovind/blink-cua) dataset; [watch them in the Space](https://huggingface.co/spaces/thegovind/blink?tab=computer-use).

## Run in a container

The Dockerfile installs the same browser and Python dependencies without assuming a preexisting machine image. Set the output mount and the URL that will be reachable **from the container**. This example uses host networking on Linux; use a reachable URL and the networking options appropriate to another host:

```sh
docker build -f examples/cua/Dockerfile -t blink-cua .
BLINK_CUA_RUNS_DIR="$PWD/cua-runs"
BLINK_SERVER_URL=http://127.0.0.1:8000
mkdir -p "$BLINK_CUA_RUNS_DIR"
docker run --rm --network host \
  -v "$PWD:/work:ro" -v "$BLINK_CUA_RUNS_DIR:/runs" \
  blink-cua python -m examples.cua.harness.sweep \
  --scenario shop --task 0 --model blink-mimo-9b --seeds 1 \
  --server "blink-mimo-9b=$BLINK_SERVER_URL" --runs-dir /runs --run-id container-shop
```

To run the CPU tests without a server, run each file in a separate pytest
process. The synchronous browser tests keep an event loop active until their
process exits:

```sh
for file in examples/cua/tests/test_*.py; do
  uv run --no-project --python 3.12 --with playwright --with pillow --with httpx \
    --with imageio-ffmpeg --with pytest python -m pytest -q -p no:cacheprovider "$file" || exit 1
done
```

Code: Apache-2.0. Weights: non-commercial research and evaluation only; see each model card's license.
