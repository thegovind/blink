# Train a decision adapter

Fine-tune a compatible text checkpoint for a set of typed decisions, then merge
the adapter into a standalone folder for `serve.py`. The example data is
invented and demonstrates the file format and commands, **not** a useful
trained model or an accuracy result. Supply independently collected training
and disjoint development data for a real task.

From the repository root, install the project's `train` dependencies
(`python -m pip install -e '.[train]'`), then run:

```sh
MODEL=thegovind/blink-4b OUT_DIR=/tmp/blink-example bash examples/train/train.sh
```

For a small **CPU-only** smoke instead, the full `train` extra is not
needed. With `uv` installed and a compatible small text checkpoint
already saved locally:

```sh
uv venv --python 3.11 /tmp/blink-train-venv
uv pip install --python /tmp/blink-train-venv/bin/python --torch-backend cpu \
  'torch==2.13.0' 'transformers==5.17.0' 'peft>=0.12' 'accelerate>=1.0' 'safetensors>=0.4'
MODEL=/path/to/small-text-model DEVICE=cpu TARGETS=attn,mlp \
  OUT_DIR=/tmp/blink-small-example PATH="/tmp/blink-train-venv/bin:$PATH" \
  bash examples/train/train.sh
```

The script takes 16 training rows and four held-out development rows from
`sample.jsonl`. It runs `torch.distributed.run` with one process, trains a
rank-16, alpha-32 LoRA adapter, merges it into the same base model, starts
`serve.py`, posts one mixed-type request with `curl`, and stops the server. The
adapter is in `$OUT_DIR/run/final`; the loadable checkpoint, including config,
weights, tokenizer and serving scripts, is in `$OUT_DIR/merged`. Run it again
with `python "$OUT_DIR/merged/serve.py" --model "$OUT_DIR/merged"` (set
`--port` if needed). Use a **fresh** output folder.
To use real data, set both `TRAIN_DATA=/path/to/train.jsonl` and
`DEV_DATA=/path/to/dev.jsonl` as well. `MODEL` may instead point to a
compatible local text checkpoint; set `TARGETS=attn,mlp` for a model without
Gated DeltaNet projections. `DEVICE=cpu` selects the small-model smoke path;
the training default remains `cuda`. A full-size checkpoint requires
sufficient resources; no result from these 20 examples predicts its quality.

## One JSON object per question

Each line has `state` (text, object or array), one `question` object and a
supervision field. Do **not** wrap questions in a serving-style `questions`
map. `id` and `src` are useful identifiers; the loss uses `weight` (default
1), `gold` or `target`, and the rendered `state` and `question`.

| Question | `question.criteria` | Supervision keys |
| --- | --- | --- |
| `choice` | Object mapping option keys to descriptions, or a list of keys | `gold` is an original option key; `target` maps those keys to weights |
| `noul` | Optional `{"true": "...", "false": "..."}` (or `yes`/`no`) descriptions | `gold` is `"yes"` or `"no"`; `target` uses `"yes"` / `"no"` |
| `score` | Ordered list of 2-10 level descriptions | `gold` is a **string** index (`"0"`, `"1"`, ...); `target` uses those string indices |

For example (each object belongs on its own JSONL line):

```json
{"id":"c1","state":"Fictional parcel arrived damaged.","question":{"type":"choice","instructions":"Which remedy?","criteria":{"refund":"Return payment","replace":"Send a replacement"}},"gold":"refund"}
{"id":"n1","state":{"deadline":"soon"},"question":{"type":"noul","instructions":"Is a reply needed soon?","criteria":{"true":"Soon","false":"Later"}},"target":{"yes":0.8,"no":0.2}}
{"id":"s1","state":"All fictional test accounts are blocked.","question":{"type":"score","instructions":"Rate the impact.","criteria":["None","Partial","Complete"]},"gold":"2"}
```

`target` takes precedence over `gold` when present: nonnegative values with
a positive sum are normalised over **offered** keys; omitted offered keys
receive zero weight. With no `target`, `gold` becomes one-hot. Use held-out
rows in `--dev` with the same format. Development reports mean restricted-label cross-entropy and
accuracy against each target's largest probability (ignoring row weights),
not an external benchmark.
Long rows beyond `--max-len` are dropped; check the printed encoded count.
Invalid labels or distributions and empty training or development sets
fail rather than producing an untrained adapter.

The `semif` renderer puts `{"evidence": state, "criterion": instructions,
"options": [{"letter": "A", "description": "..."}, ...]}` in the user
message. Non-generic choice keys appear in descriptions (for example,
`refund: Return payment`); generic keys display their descriptions directly.
Yes/no becomes `Yes` / `No` with optional true/false descriptions;
score becomes `Level 0: ...`, `Level 1: ...`.
Choice and yes/no option order and letter assignment are reshuffled
each epoch; score levels stay ordered. The model predicts the next
single-token option letter, with cross-entropy restricted to offered
letters (A-Z, then two-letter labels). The training tokenizer must
verify all 255 labels at the assistant boundary. Only LoRA projections
selected by `--targets` are trained (attention, Gated DeltaNet and MLP
by default); embeddings, norms and
the option-letter head stay frozen. `--init-adapter` can continue an
adapter trained on the **same base**.

`merge.py` saves tokenizer files along with the merged text weights.
`serve.py --model "$OUT_DIR/merged"` loads that folder without a
`weights.sha256` file; without that optional manifest `/healthz` reports
`"weights_verified": null`, not a passed checksum.

**Computer use:** text descriptions of UI controls, accessibility trees and
DOM state can already be used as `state` (see rows `demo-11` through `demo-15`).
Screenshot **pixels** are not trained by this text path: placing an image data
URI in `state` would train on its encoded text. Screenshot LoRA training is
coming next, subject to a separately tested image-aware trainer and merge
path. Serving screenshots with `--vision` or `--vision-tower` does not make
this training script image-aware.

**Licensing:** code is Apache-2.0. Released blink weights are licensed for
non-commercial research and evaluation only; an adapter or merged fine-tune
based on those weights remains subject to those terms. Starting instead
from a compatible Apache-2.0 base checkpoint is a separate choice; it does
not use or relicense blink weights. Check the chosen base and training-data
rights before distributing a fine-tune.
