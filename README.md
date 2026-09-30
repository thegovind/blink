# blink

Send text or JSON state with typed questions. Each question gets a probability for every offered option from one forward pass. No generated text.

Code: Apache-2.0 ([LICENSE](LICENSE)). Weights: non-commercial research and evaluation only; see each model card's license.
[![Models](https://img.shields.io/badge/Hugging%20Face-models-yellow.svg)](https://huggingface.co/thegovind)
[![Space](https://img.shields.io/badge/demo-Space-orange.svg)](https://huggingface.co/spaces/thegovind/blink)

## What it does

- `noul`: probability of yes.
- `choice`: picked option, probabilities, confidence.
- `score`: expected level across 2-10 ordered levels, probabilities, confidence.

Probabilities are over the offered options only, not certified chances of being right.

blink reads each question's prompt in one forward pass and takes the next-token scores of the offered option letters. It never writes text, so it can only pick among the options you list. Offer "none" or "unclear" when those can happen.

## Get started

| You want to | Start here |
|---|---|
| Try it, no install | Open the [Space](https://huggingface.co/spaces/thegovind/blink). |
| Call it from code, no install | Call the Space with `gradio_client`; see [The Space](https://thegovind.github.io/blink/api.md#the-space). Anonymous calls have a daily limit. |
| Serve it for TypeSafe SDKs or HTTP | [Run it](#run-it), then set `TYPESAFE_BASE_URL` to your server. |
| Call it from Python, no server | Load `blink.py` from a model repo; see the [blink-4b quickstart](https://huggingface.co/thegovind/blink-4b#quickstart). |
| Run it on a Mac | [Run it on a Mac](#run-it-on-a-mac) with MLX. |
| Give it to an agent | [Use it from agents](#use-it-from-agents). |

## Try it

Open the [Space](https://huggingface.co/spaces/thegovind/blink).

See [Computer use](https://thegovind.github.io/blink/computer-use/) for browser agents and self-hosted screenshots, or try [Screen click](https://huggingface.co/spaces/thegovind/blink?tab=computer-use).

Run [ten offline computer-use scenarios](examples/cua/README.md) against a self-hosted screenshot-enabled blink server, or use the CPU mock to check the harness.

## Run it

```sh
pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
hf download thegovind/blink-4b --revision v1.4 --local-dir blink-4b
python blink-4b/serve.py --model ./blink-4b --port 8000
```

Check `curl -s http://localhost:8000/healthz`.

For self-hosted latency and throughput, see [serving options](https://thegovind.github.io/blink/models.md#serve-it).

## Run it on a Mac

```sh
pip install "mlx-lm>=0.31.3"
curl -O https://raw.githubusercontent.com/thegovind/blink/main/examples/mlx/blink_mlx.py
python blink_mlx.py --model thegovind/blink-4b
python blink_mlx.py --check   # compare with the Space's saved runs
```

`blink_mlx.py` uses the model repo's `blink.py` for prompts and answers, and MLX for the forward pass. Text and JSON state only, bf16 weights only, no server. It was checked on Linux, so run `--check` once on your Mac. See [examples/mlx](examples/mlx/README.md) for memory, checks and limits.

## Use it from code

Follow the [API docs](https://thegovind.github.io/blink/api.md).

For screenshots on a self-hosted v1.3 or later server, [turn on image input](https://thegovind.github.io/blink/api.md#screenshots-servepy-only-opt-in).

## Use it from agents

See [`skills/blink/SKILL.md`](skills/blink/SKILL.md) and the [agent experience guide](https://thegovind.github.io/blink/agents.md).

## Models

| Model | Base | Weights (bf16) | Live in the Space |
|---|---|---:|---|
| [thegovind/blink-4b](https://huggingface.co/thegovind/blink-4b) | Qwen/Qwen3.5-4B | 8.4 GB | yes |
| [thegovind/blink-mimo-9b](https://huggingface.co/thegovind/blink-mimo-9b) | XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B | 18.8 GB | yes |
| [thegovind/blink-27b](https://huggingface.co/thegovind/blink-27b) | Qwen/Qwen3.8-27B | 53.8 GB | no |

Code: Apache-2.0. Weights: non-commercial research and evaluation only; see each model card's license.

## Links

- [Docs](https://thegovind.github.io/blink/)
- [Agent experience](https://thegovind.github.io/blink/agents.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- Code: Apache-2.0 ([LICENSE](LICENSE)); weights: non-commercial research and evaluation only (see each model card's license).
