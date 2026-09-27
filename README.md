# blink

Send text or JSON state with typed questions. Each question gets a probability for every offered option from one forward pass. No generated text.

[![License](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Models](https://img.shields.io/badge/Hugging%20Face-models-yellow.svg)](https://huggingface.co/thegovind)
[![Space](https://img.shields.io/badge/demo-Space-orange.svg)](https://huggingface.co/spaces/thegovind/blink)

## What it does

- `noul`: probability of yes.
- `choice`: picked option, probabilities, confidence.
- `score`: expected level across 2-10 ordered levels, probabilities, confidence.

Probabilities are over the offered options only, not certified chances of being right.

## Try it

Open the [Space](https://huggingface.co/spaces/thegovind/blink).

## Run it

```sh
pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
hf download thegovind/blink-4b --revision v1.3 --local-dir blink-4b
python blink-4b/serve.py --model ./blink-4b --port 8000
```

Check `curl -s http://localhost:8000/healthz`.

## Use it from code

Follow the [API docs](https://thegovind.github.io/blink/api.md).

For screenshots on a self-hosted v1.3 server, [turn on image input](https://thegovind.github.io/blink/api.md#screenshots-opt-in).

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
- [License](LICENSE)
