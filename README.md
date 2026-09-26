# blink

Send text or JSON state plus typed questions. Get a probability for every option in one forward pass, with no generated text.

[![License](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Models](https://img.shields.io/badge/Hugging%20Face-models-yellow.svg)](https://huggingface.co/thegovind)
[![Space](https://img.shields.io/badge/demo-Space-orange.svg)](https://huggingface.co/spaces/thegovind/blink)

## What it does

- Accepts state plus questions keyed by name.
- Supports yes/no (`noul`), choice, and ordered score questions.
- Returns option probabilities and no generated text.

## Try it

Use the [live Space](https://huggingface.co/spaces/thegovind/blink).

## Run it

```sh
pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
hf download thegovind/blink-4b --revision v1.2 --local-dir blink-4b
python blink-4b/serve.py --model ./blink-4b --port 8000
```

Then check `curl -s http://localhost:8000/healthz`.

## Use it from code

See the [API docs](https://thegovind.github.io/blink/api.md).

## Use it from agents

Give your agent [`skills/blink/SKILL.md`](skills/blink/SKILL.md), then read the [agent experience guide](https://thegovind.github.io/blink/agents.md).

## Models

| Model | Base | Weights (bf16) | Live in the Space |
|---|---|---:|---|
| [thegovind/blink-4b](https://huggingface.co/thegovind/blink-4b) | Qwen/Qwen3.5-4B | 8.4 GB | yes |
| [thegovind/blink-mimo-9b](https://huggingface.co/thegovind/blink-mimo-9b) | XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B | 18.8 GB | yes |
| [thegovind/blink-27b](https://huggingface.co/thegovind/blink-27b) | Qwen/Qwen3.8-27B | 53.8 GB | no |

Use code revision `v1.2`. Weights are unchanged since `v1.0`.

## Links

- [Docs](https://thegovind.github.io/blink/)
- [Agent experience](https://thegovind.github.io/blink/agents.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [License](LICENSE)
