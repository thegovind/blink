# blink

Send text or JSON state plus typed questions. Get a probability for every option in one forward pass, with no generated text.

## Try it in the Space

Test blink and find the API tab in the [live Space](https://huggingface.co/spaces/thegovind/blink).

## Call the API

Send `noul`, `choice`, and `score` questions with the [API guide](api.md).

## Give it to your agent

Add the blink skill from the repo and use the [agent experience guide](agents.md).

## Quickstart

```sh
pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
hf download thegovind/blink-4b --revision v1.2 --local-dir blink-4b
python blink-4b/serve.py --model ./blink-4b --port 8000
```
