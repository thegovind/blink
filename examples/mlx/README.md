# Run blink on a Mac with MLX

blink reads a decision in one forward pass over the prompt and never generates text. `blink_mlx.py` keeps the
published `blink.py` for prompts, option labels, request checks and answers, and lets [MLX](https://github.com/ml-explore/mlx)
run the forward pass on Apple silicon. It returns the same answers as `blink.decide`.

## Start

```sh
pip install "mlx-lm>=0.31.3"
curl -O https://raw.githubusercontent.com/thegovind/blink/main/examples/mlx/blink_mlx.py
python blink_mlx.py
```

The first run downloads blink-4b at revision `v1.4` from Hugging Face and answers the model-card example: a
customer message with one `choice`, one `noul` and one `score` question.

## Your own request

Put the request in a file with the same fields as the API:

```json
{
  "state": "My card was charged twice this month.",
  "questions": {"urgent": {"type": "noul", "instructions": "Does this need an answer today?"}}
}
```

```sh
python blink_mlx.py --request request.json
python blink_mlx.py --model thegovind/blink-mimo-9b --request request.json
python blink_mlx.py --model ./my-model --request request.json          # a local folder that holds blink.py
python blink_mlx.py --adapter runs/mine --request request.json         # an adapter from train_mlx.py, on its own base
```

From Python, with `blink_mlx.py` next to your code (`load` also takes `adapter=` and `temperature=`):

```python
from blink_mlx import load

blink = load("thegovind/blink-4b")
out = blink.decide("My card was charged twice this month.",
                   {"urgent": {"type": "noul", "instructions": "Does this need an answer today?"}})
print(out["answers"]["urgent"]["noul"])
```

## Fine-tune on your Mac

`train_mlx.py` trains a LoRA adapter with the same objective as the PyTorch trainer: blink's prompts, option order
and letters reshuffled every epoch, cross-entropy over the offered letters against each row's target, rank 16 and
alpha 32 on the attention, Gated DeltaNet and MLP projections, AdamW with warmup and cosine decay. Run it from a
clone of the repository, with rows in the [training format](../train/README.md):

```sh
git clone https://github.com/thegovind/blink && cd blink
pip install "mlx-lm>=0.31.3"
python examples/train/prepare.py rows.jsonl --out-dir data --dev-fraction 0.15
PYTHONPATH=lab python -m jevlab.synth_core_v5 data/general.jsonl 200 general   # about a tenth of your rows
python examples/mlx/anchor_mlx.py --model thegovind/blink-4b --inp data/general.jsonl --out data/anchors.jsonl
python examples/mlx/train_mlx.py --model thegovind/blink-4b --data data/train.jsonl,data/anchors.jsonl \
  --dev data/dev.jsonl --out runs/mine --lr 4e-5
python examples/mlx/blink_mlx.py --model thegovind/blink-4b --adapter runs/mine --request request.json
```

- `anchor_mlx.py` labels general decisions with the model's own probabilities, so training on your task does not
  erase what it already does. Aim for about one anchor row per ten training rows.
- `train_mlx.py` reads the dev rows before and after training and writes `eval-init.json` and `eval-final.json`:
  accuracy, log loss, Brier score and calibration error, per `src`. `--eval-only` reads them without training.
- If the probabilities drifted, `PYTHONPATH=lab python -m jevlab.fit_temp runs/mine/eval-final.json` fits one
  temperature T; use it with `blink_mlx.py --temperature T`, and confirm on rows you did not fit it on.
- Gradient checkpointing is on by default. Training needs more memory than answering: start with blink-4b and
  `--max-len 2048` if memory is tight.
- Training runs the Gated DeltaNet layers one prompt token at a time (mlx-lm's differentiable path; answering uses
  a fused kernel), so training time grows with prompt length. Try a few hundred short rows first.

This trainer was checked against the PyTorch trainer on a small random model with blink-4b's architecture and
tokenizer, on Linux, in float32: the same prompt tokens as serving, per-row loss within 1.2e-5, the same number of
LoRA parameters, and the LoRA B gradients at initialisation within 5e-5 relative. It has not been timed on a Mac. The adapter runs with
`blink_mlx.py`; to serve it with `serve.py`, train with the [PyTorch trainer](../train/README.md) instead.

## Memory

| Model | Weights (bf16) | Suggested Mac memory |
|---|---:|---:|
| [thegovind/blink-4b](https://huggingface.co/thegovind/blink-4b) | 8.4 GB | 16 GB |
| [thegovind/blink-mimo-9b](https://huggingface.co/thegovind/blink-mimo-9b) | 18.8 GB | 32 GB |
| [thegovind/blink-27b](https://huggingface.co/thegovind/blink-27b) | 53.8 GB | 96 GB |

macOS gives MLX only part of unified memory by default, and the prompt needs room beside the weights. These
suggestions leave that room; they were not measured on each Mac. blink-mimo-9b downloads its full checkpoint and
loads only its text side.

## Check your Mac's answers

```sh
python blink_mlx.py --check
python blink_mlx.py --model thegovind/blink-mimo-9b --check
```

`--check` compares every bundled Space example that has a saved PyTorch run (22 unique requests) with that run:
the prompt's token count, the top answer and the largest probability gap. It reads the saved runs at a fixed Space
commit, so later Space updates do not change the comparison. The Space has saved runs for blink-4b and
blink-mimo-9b. For blink-27b, record your own on a machine that runs PyTorch with `space/record_replay.py`, then
pass it: `python blink_mlx.py --model thegovind/blink-27b --check --reference your-run.json`.

Checked on Linux with MLX 0.32.3 and mlx-lm 0.31.3, over 22 requests and 81 questions per model:

| Model | Reference | Same token count | Same top answer | Largest probability gap |
|---|---|---:|---:|---:|
| blink-4b | Space's saved runs | 22/22 | 80/81 | 0.040 |
| blink-mimo-9b | Space's saved runs | 22/22 | 79/81 | 0.022 |
| blink-27b | PyTorch on the same machine | 22/22 | 81/81 | 0.029 |

The three questions that changed answer were near-ties in the saved runs: their top two options were within 0.024.
This check did not run on a Mac. On Apple silicon, mlx-lm runs the Gated DeltaNet layers with a different kernel,
so run `--check` once on yours.

## Limits

- Text and JSON state only. Screenshots need `serve.py` with image input; see the [API guide](https://thegovind.github.io/blink/api.md#screenshots-servepy-only-opt-in).
- bf16 weights only; the runner refuses quantized conversions. Quantized MLX builds have not been checked, and the
  INT8 builds tried on the server path changed too many answers.
- One forward pass per question, one after another, with no batching or prefix cache. Each question costs about
  one prompt pass, and the state is read again for every question.
- Not a server. For HTTP and TypeSafe SDKs, run `serve.py`; see [Run it](../../README.md#run-it).

Code: Apache-2.0. Weights: non-commercial research and evaluation only; see each model card's license.
