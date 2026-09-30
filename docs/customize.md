# Customize

Teach blink your own decisions. This page is the map: pick a path, build the data, train, check the result and serve it. The runnable pieces live in [examples/train](https://github.com/thegovind/blink/tree/main/examples/train), [examples/mlx](https://github.com/thegovind/blink/tree/main/examples/mlx) and [examples/rl](https://github.com/thegovind/blink/tree/main/examples/rl).

## Pick a path

| Your situation | Path | Effort |
|---|---|---|
| The released model already gets most of your questions right | Improve the questions; no training | Minutes |
| You have labeled examples, or can make them | Fine-tune a LoRA adapter | Hours |
| You find out later what actually happened after each decision | Learn from outcomes: the same fine-tune, with the outcome as the label | Hours |
| An agent makes several decisions and passes or fails at the end | RL in an environment (a recipe and results; the trainer is not published) | Days |

Start at the top. Measure the released model on your own questions before you train anything.

## How blink learns

One question about one state becomes one prompt. The model reads it once and scores the offered option letters; the answer is the softmax over those scores. Training moves those scores and nothing else:

- The loss is cross-entropy over the offered letters only, against a target distribution: one-hot for a single right answer, or soft when the answer is uncertain.
- Choice and yes/no options get a new order and new letters every epoch, so the model learns the content, not the position. Score levels keep their order.
- A LoRA adapter (rank 16, alpha 32) trains the attention, Gated DeltaNet and MLP projections. Embeddings, norms and the letters' output rows stay frozen, so the readout keeps working.

Because the loss is a proper scoring rule, the model is pushed toward honest probabilities, not just right answers.

## 1. Build your rows

Collect real cases with the right answer, including 100 to 500 you will use only to measure. Each JSONL line is one question about one state, with a label:

```json
{"id": "t1", "src": "tickets", "group": "customer-17", "state": {"ticket": "I was charged twice this month."}, "question": {"type": "choice", "instructions": "Which queue should take this ticket?", "criteria": {"billing": "Billing", "tech": "Technical support", "sales": "Sales"}}, "gold": "billing"}
```

Where labels come from:

| Source | Field | Notes |
|---|---|---|
| One person's label | `"gold": "billing"` | The usual case. |
| Several people's labels | `"votes": {"billing": 3, "tech": 1}` | Becomes a soft target: honest about close calls. |
| What happened later | `"outcome": "billing"` | Learning from outcomes; see [RL](#7-rl-and-learning-from-outcomes). |
| A stronger model | `"target": {...}` | `jevlab.teacher` samples a reasoning model several times; the shares become the target. |
| A program that knows the answer | `"gold"` or an exact `"target"` | The `jevlab.synth_*` generators show the pattern. |

Then check them and hold out a dev set:

```sh
python examples/train/prepare.py rows.jsonl --out-dir data --dev-fraction 0.15 --model thegovind/blink-4b
```

`prepare.py` checks every row with the trainer's own rules, turns votes and outcomes into labels, keeps each `group` (or each identical state) on one side of the split, refuses a dev file that repeats a training question, and counts prompt tokens. The [training example](https://github.com/thegovind/blink/tree/main/examples/train) documents every field.

## 2. Measure the model you start from

Read the dev rows with the released model before you train anything:

```sh
# a GPU
PYTHONPATH=lab python -m jevlab.dev_run --model thegovind/blink-4b --dev data/dev.jsonl --out runs/base-dev.json
# a Mac
python examples/mlx/train_mlx.py --model thegovind/blink-4b --dev data/dev.jsonl --out runs/base --eval-only
```

`dev_run` prints accuracy, log loss, Brier score and calibration error, overall and per `src`, and saves them with every answer. `train_mlx.py --eval-only` prints accuracy and calibration error and writes the full table to `runs/base/eval-init.json`; its `--out` must be a new folder each time. If the numbers are close to what you need, work on the questions first: clear criteria, an explicit "none" or "unclear" option when that can happen, and score levels that each say what they mean. The [agent skill](https://github.com/thegovind/blink/blob/main/skills/blink/SKILL.md) has the rules.

## 3. Keep what it already knows

Training on one narrow task makes a model forget others. In research runs, 200 updates on narrow data with no replay cost blink-4b 25 to 32 of the 111 hard public JevBench items. The fix is anchor rows: general decisions labeled with the probabilities of the model you start from. Training against them pulls toward your task without moving the rest.

```sh
PYTHONPATH=lab python -m jevlab.synth_core_v5 data/general.jsonl 200 general   # about a tenth of your training rows
# a GPU
PYTHONPATH=lab python -m jevlab.anchor --model thegovind/blink-4b --inp data/general.jsonl --out data/anchors.jsonl
# a Mac
python examples/mlx/anchor_mlx.py --model thegovind/blink-4b --inp data/general.jsonl --out data/anchors.jsonl
```

Aim for about one anchor row for every ten training rows; the released blink-4b training runs had about 8%. `synth_core_v5` needs no network and draws decisions from many families; set its count from your own row count. Add other tasks you care about the same way.

## 4. Train

| Where | How | Notes |
|---|---|---|
| A Mac with Apple silicon | [`examples/mlx/train_mlx.py`](https://github.com/thegovind/blink/tree/main/examples/mlx) | Same prompts, loss, LoRA and schedule as the reference trainer; checked against it on a small model. Not yet timed on a Mac; long prompts train slowly. |
| A GPU, local or a cloud VM | [`examples/train`](https://github.com/thegovind/blink/tree/main/examples/train): `jevlab.train` (PyTorch and PEFT) | The reference trainer; it trained the released models. |
| Azure AI Foundry | [`examples/train/azureml/train-job.yml`](https://github.com/thegovind/blink/blob/main/examples/train/azureml/train-job.yml) | An Azure Machine Learning command job on a GPU cluster that writes a servable folder. A template: schema-checked, not run by the maintainers. |
| Unsloth | Unsloth's model loading with blink's loss | Unsloth's [Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) lists about 10 GB for 16-bit LoRA on the 4B size and advises against 4-bit training for this family. Its standard fine-tuning loss is not blink's: keep blink's prompts and the loss over the offered letters (`forward_batch` in `lab/jevlab/train.py`). Not tested by the maintainers. |

On a GPU:

```sh
python -m pip install -e '.[train]'
PYTHONPATH=lab python -m torch.distributed.run --standalone --nproc_per_node=1 --module jevlab.train \
  --model thegovind/blink-4b --data data/train.jsonl,data/anchors.jsonl --dev data/dev.jsonl --out runs/mine \
  --lr 4e-5 --epochs 1 --budget 8192 --max-len 8192 --warmup 10
```

On a Mac:

```sh
pip install "mlx-lm>=0.31.3"
python examples/mlx/train_mlx.py --model thegovind/blink-4b --data data/train.jsonl,data/anchors.jsonl \
  --dev data/dev.jsonl --out runs/mine-mlx --lr 4e-5
```

On Azure AI Foundry, from the repository root:

```sh
az ml job create -f examples/train/azureml/train-job.yml --set compute=azureml:<your-gpu-cluster> \
  --resource-group <group> --workspace-name <workspace>
```

The job uploads the repository's `data/` folder (train, dev and anchor rows) and the code, without `.git`, your weights
or other working folders (`.amlignore`).

Settings that shipped: LoRA rank 16 and alpha 32, one epoch, warmup of 10 to 20 steps, and a learning rate of 3e-5 to 5e-5 (blink-4b averages two runs from the base, 3e-5 for 96 steps and 4e-5 for 472; blink-mimo-9b used 5e-5 for 615). With a few thousand rows, start at 4e-5 and compare 2e-5 and 5e-5 on dev. If memory runs out, lower `--budget` and `--max-len`; prompts longer than `--max-len` are skipped, never cut.

Use LoRA, not full fine-tuning. In research runs, full fine-tunes fit the training rows better (dev log loss 0.227 against 0.355) but lost 7 to 9 hard JevBench items and grew less calibrated.

## 5. Check before you ship

- **Compare with the model you started from** on the same dev set: accuracy, log loss, Brier score and calibration error. On a GPU:

  ```sh
  PYTHONPATH=lab python -m jevlab.dev_run --model thegovind/blink-4b --adapter runs/mine/final \
    --dev data/dev.jsonl --out runs/mine-dev.json
  ```

  On a Mac, `train_mlx.py` writes `runs/mine-mlx/eval-init.json` (before) and `eval-final.json` (after).
- **Check the rest still works**: your other tasks, and general rows you kept out of the anchors.
- **Fix calibration with one number** if it drifted: `PYTHONPATH=lab python -m jevlab.fit_temp runs/mine-dev.json` (or `runs/mine-mlx/eval-final.json`) fits a temperature T on dev. Serve with `BLINK_TEMPERATURE=T` or `blink_mlx.py --temperature T`, and confirm on held-out data, since T was fitted on dev.
- **Keep a final held-out set** you read once, at the end. Accept the new model only if it beats the one you started from there too.

## 6. Merge, average and serve

```sh
PYTHONPATH=lab python -m jevlab.merge --base thegovind/blink-4b --adapter runs/mine/final --out merged
cp serve.py space/blink.py space/graft_keys.py merged/
python merged/serve.py --model merged --port 8000
```

- **Average with the release** when the fine-tune lost calibration or general skill: download it with `hf download thegovind/blink-4b --revision v1.4 --local-dir blink-4b`, run `PYTHONPATH=lab python -m jevlab.soup soup merged blink-4b`, and copy the three serving files into `soup`. In research runs, averaging a full fine-tune with the release cut its calibration error from 0.151 to 0.103.
- **blink-mimo-9b** trains its text side. `jevlab.merge` writes that side alone; to keep screenshots, `PYTHONPATH=lab python -m jevlab.graft_vlm --parent thegovind/blink-mimo-9b --text merged --out grafted` puts the trained weights back into the full checkpoint with its vision tower unchanged.
- **Docker**: `cp docker/Dockerfile.model merged/Dockerfile`, then `docker build` in `merged`.
- **Azure AI Foundry**: push that image to a registry, then create [an endpoint](https://github.com/thegovind/blink/blob/main/examples/train/azureml/endpoint.yml) and [a deployment](https://github.com/thegovind/blink/blob/main/examples/train/azureml/deployment.yml) that runs `serve.py` with `/healthz` probes. Templates, schema-checked.
- **A Mac**: `python examples/mlx/blink_mlx.py --adapter runs/mine-mlx` for an MLX adapter, or `--model merged` for a merged folder with `blink.py` in it.
- **TypeSafe SDKs**: point `TYPESAFE_BASE_URL` at `serve.py`, as in the [API guide](api.md).

## 7. RL and learning from outcomes

blink never writes text, but its choices are actions and its probabilities over the offered options are a policy, so RL applies.

- **One decision at a time**: if you learn what happened after each decision, train on the outcome with the log score. That is the fine-tune above with `"outcome"` as the label. Do not reward right answers: an accuracy reward makes the model more and more certain without making its top pick come true more often. `python examples/rl/reward_demo.py` shows it in seconds.
- **Several decisions, one result**: an agent that clicks, types and stops needs RL in an environment. Sample choices from blink's probabilities, score whole episodes, and push up the choices from episodes that beat their group (GRPO or GSPO). In research runs on the offline computer-use apps, this lifted blink-mimo-9b from 132 to 194 of the 220 non-game tasks and paused every risky click, but it made the shared model's text probabilities less calibrated, so it was not released.

[examples/rl](https://github.com/thegovind/blink/tree/main/examples/rl) has the recipe, the numbers and a sampling policy for the computer-use harness.

## Screenshots

The trainers here train text and JSON states. For computer use, describe the screen as text, such as an accessibility tree or the visible controls, and train on that. Screenshot training is not published yet: an early screenshot fine-tune finished more tasks but paused fewer risky clicks, and that has to be fixed first.

## License

Code: Apache-2.0. Weights: non-commercial research and evaluation only; see each model card's license. A fine-tune or merge of blink weights stays under those terms. Starting from an Apache-2.0 base checkpoint instead is a separate choice; check the base model's and your data's terms before you share a model.
