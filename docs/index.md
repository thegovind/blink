# blink

Send text or JSON state with typed questions. Each question gets a probability for every offered option from one forward pass. No generated text.

Probabilities are over the offered options only, not certified chances of being right.

## Try it in the Space

Open the [Space](https://huggingface.co/spaces/thegovind/blink) to try blink or use its API tab.

## Call the API

Follow the [API guide](api.md) to send `noul`, `choice`, and `score` questions.

## Give it to your agent

Copy the blink skill, then follow the [agent experience guide](agents.md).

## Quickstart

```sh
pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
hf download thegovind/blink-4b --revision v1.4 --local-dir blink-4b
python blink-4b/serve.py --model ./blink-4b --port 8000
```

Code: Apache-2.0. Weights: non-commercial research and evaluation only; see each model card's license.
