# blink

Send text or JSON state with typed questions. Each question gets a probability for every offered option from one forward pass. No generated text.

Probabilities are over the offered options only, not certified chances of being right.

blink reads each question's prompt in one forward pass and takes the next-token scores of the offered option letters. It never writes text, so it can only pick among the options you list.

## Try it in the Space

Open the [Space](https://huggingface.co/spaces/thegovind/blink) to try blink or use its API tab.

Try [Screen click](https://huggingface.co/spaces/thegovind/blink?tab=computer-use) for screenshots, or open [Computer use](computer-use.md) for browser agents and self-hosted setup.

Run the [offline computer-use scenarios](https://github.com/thegovind/blink/tree/main/examples/cua) with a self-hosted blink server; the [Computer use scenarios section](computer-use.md#scenarios) has the results and videos.

## Call the API

Follow the [API guide](api.md) to send `noul`, `choice`, and `score` questions.

## Train it on your decisions

Follow the [customize guide](customize.md) to fine-tune on your data, keep the probabilities honest, and serve the result with PyTorch, on a Mac with MLX, or on Foundry.

## Give it to your agent

Copy the blink skill, then follow the [agent experience guide](agents.md).

## Quickstart

```sh
pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
hf download thegovind/blink-4b --revision v1.4 --local-dir blink-4b
python blink-4b/serve.py --model ./blink-4b --port 8000
```

## Run it on a Mac

```sh
pip install "mlx-lm>=0.31.3"
curl -O https://raw.githubusercontent.com/thegovind/blink/main/examples/mlx/blink_mlx.py
python blink_mlx.py --model thegovind/blink-4b
python blink_mlx.py --check   # compare with the Space's saved runs
```

Text and JSON state only, bf16 weights only, no server. It was checked on Linux, so run `--check` once on your Mac. The [MLX example](https://github.com/thegovind/blink/tree/main/examples/mlx) covers memory, checks and limits.

Code: Apache-2.0. Weights: non-commercial research and evaluation only; see each model card's license.
