---
license: other
license_name: blink-research
license_link: LICENSE.md
base_model: Qwen/Qwen3.5-4B
library_name: transformers
pipeline_tag: text-classification
inference: false
tags:
  - decision-model
  - typed-decisions
  - one-pass
  - option-probabilities
language:
  - en
---

# blink-4b

Send a text or JSON `state` and your questions: `choice` picks from up to 255 options, `noul` is yes/no,
and `score` takes 2–10 ordered levels. Each question gets probabilities over its offered options from
one forward pass, with no generated text. Long or large multi-question requests may use several batches.

**Try it:** [Space demo](https://huggingface.co/spaces/thegovind/blink) — blink-4b and blink-mimo-9b run live ·
[blink-4b](https://huggingface.co/thegovind/blink-4b) · [blink-27b](https://huggingface.co/thegovind/blink-27b) ·
[blink-mimo-9b](https://huggingface.co/thegovind/blink-mimo-9b).

*Personal research release by thegovind, not an official product of any company. No affiliation with TypeSafe AI,
Xiaomi, Alibaba Cloud or the Qwen team. Weights are for non-commercial research; see [Licence](#licence).*

![blink-4b one-pass readout diagram: a state and typed choice, noul and score questions are rendered as evidence, criterion and lettered options; questions are batched and each batch is one forward pass, giving next-token logits at the answer position, and an FP32 softmax over the offered letters gives one probability per option. No text is generated.](../assets/blink-4b/blink-4b-readout.png)

## Results

### JevBench: public-item development proxies

| Model | Public-items proxy | Public hard (111) | Hard ECE | Probability TVD | Official JevBench v1.4 |
|---|---:|---:|---:|---:|---:|
| **blink-4b** | 76.5 | 80/111 | 0.067 | 0.226 | not submitted |
| JevK5 v0.2.0 | 76.1 | 79/111 (own runtime: 82/111) | 0.068 | 0.220 | 62.0 |
| Jev 1.13.0 | — | — | — | — | 63.3 |

Both proxies use the same harness on public items. They aren't official JevBench scores or predictions: official scoring needs held-out, judge and sealed items. We claim no official score, rank or parity with Jev or JevK5.

<details><summary>How to read the public-item numbers</summary>

- On the public hard items, blink-4b got 80/111 and JevK5 got 79/111 in our runtime; JevK5's own runtime reports 82/111. We don't claim a hard-accuracy advantage. blink-4b's 95% Wilson interval is 0.631–0.796, before selection effects.
- Speed comes from each row's own serial run. The proxy speed axis already applies JevBench's self-hosted adjustment.
- Cost uses JevBench's 4B tariff ($0.03 per million input tokens) times measured tokens per decision; not a production bill.

</details>

### Decision Index 0.2 (local run)

We ran the full Decision Index 0.2 suite ourselves with the official scoring kit at commit 19ad28e on 2026-09-25. This is a descriptive run, not a leaderboard submission or accepted result. The kit's scorer does not apply the leaderboard's penalty for rows an entrant trained on, so known training exposure stays in these scores and they cannot be ranked against the leaderboard.

| Balanced skill | Balanced raw | Breadth skill | Without MMLU-Pro |
|---:|---:|---:|---:|
| 37.85 | 53.33 | 36.78 | 37.41 |

- Training included 281 MMLU-Pro test-partition questions, so the 0.2 MMLU-Pro score is contaminated. “Without MMLU-Pro” drops that benchmark but does not remove other training effects. Public train splits used in training are listed under “Training overlap” below.

<details><summary>Extra tables and method</summary>

| Area | Number of benchmarks | Skill | Raw |
|---|---:|---:|---:|
| Knowledge & Reasoning | 10 | 26.4 | 43.1 |
| Language Understanding | 10 | 47.4 | 62.7 |
| Retrieval & Classification | 7 | 36.8 | 54.7 |
| Tools & Automation | 6 | 51.6 | 60.0 |
| Arts & Human Taste | 7 | 27.2 | 46.1 |

The seven benchmarks added in 0.2.

| Benchmark | Metric | Requests | Answered | Raw | Skill |
|---|---|---:|---:|---:|---:|
| PhishNChips phishing decisions | accuracy | 2,000 | 2,000 | 63.6 | 27.3 |
| MMLU-Pro | accuracy | 12,032 | 12,032 | 52.1 | 46.1 |
| BBH fixed-option tasks | accuracy | 5,507 | 5,507 | 63.8 | 47.5 |
| RAGTruth response-level hallucination | F1 on hallucinated class | 2,700 | 2,700 | 66.5 | 43.1 |
| HoVer claim verification | accuracy | 4,000 | 4,000 | 63.1 | 26.2 |
| When2Call MCQ | accuracy | 3,652 | 3,652 | 62.8 | 50.3 |
| New Yorker caption matching | accuracy | 528 | 528 | 58.9 | 48.6 |

- All 151,034 of 151,034 scoreable requests scored. Of 44 scored benchmarks, 40 count toward the index across five equal areas.
- Requests shared with 0.1 reuse the model's 0.1 predictions. We ran the 30,419 added requests with the same frozen evaluation setup as 0.1, the evaluated soup, which is the published weights, at temperature 1.0.
- Balanced skill is the headline index. “Without MMLU-Pro” drops MMLU-Pro, averages the other nine Knowledge benchmarks, and keeps five equal areas. It is a sensitivity check, not a score free of training effects.
- These are point estimates, with no significance, calibration, or latency claims. Do not compare them with 0.1 numbers because the editions differ.

Training-row text matches in the added requests.

| Training stage | Rows in the stage | Rows matching added-request text | From MMLU-Pro | From SuperGPQA | Other |
|---|---:|---:|---:|---:|---:|
| T3 | 23,156 | 138 | 131 | 6 | 1 |
| T4 | 42,360 | 156 | 148 | 7 | 1 |

We screened for exact normalised strings of at least 30 characters shared by training rows and added requests, ignoring strings found in 20 or more requests as templates. Counts are training rows by stage and source, not unique test questions. A matching option or passage need not be the same question, and a clean screen cannot rule out semantic or pretraining overlap.

We did not produce the planned calibration read or a score without the DI-S selection sample.

</details>

### Decision Index 0.1 (archived edition)

**Full suite: blink-4b 52.12 vs Jev 1.13.0 59.51.**

| Model | Size class | Decision Index 0.1 | Skill | Breadth |
|---|---|---:|---:|---:|
| **blink-4b** (this model) | 4B | 52.12 | 36.04 | 34.18 |
| Jev 1.13.0 | closed | 59.51 | 46.26 | 44.79 |
| Jevfire | 27B | 55.74 | 40.86 | 39.45 |
| JoshuaSP diffusiongemma (open-jev) | 26B-A4B | 55.56 | 40.84 | 39.19 |
| Decider 35B-A3B | 35B-A3B | 54.34 | 39.37 | 37.99 |
| Kev 9B | 9B | 50.48 | 32.96 | 30.54 |
| Kev 4B | 4B | 47.43 | 28.86 | 25.67 |

We ran the complete archived 0.1 suite: 132,422 requests across 37 benchmarks. The headline index averages 19 panel benchmarks. Comparison rows use the 2026-09-22 leaderboard snapshot. We ran the official kit's scorer locally; these aren't leaderboard submissions. The live [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) moved to 0.2 on 2026-09-24. Our local 0.2 run is in the section above.

On the archived 0.1 board, the best open entry with 3.5–5B served parameters was Kev 4B at 47.43; this model scored 52.12.

| Area | blink-4b | Jev 1.13.0 |
|---|---:|---:|
| Knowledge & Reasoning | 49.1 | 68.8 |
| Language Understanding | 61.0 | 62.3 |
| Retrieval & Classification | 30.1 | 37.0 |
| Tools & Automation | 70.2 | 73.6 |
| Arts & Human Judgment | 50.2 | 56.2 |

<details><summary>Per benchmark (19 panel benchmarks, 0.1)</summary>

| Area | Benchmark | This model | Jev 1.13.0 |
|---|---|---:|---:|
| Knowledge | MMLU | 0.749 | 0.917 |
| Knowledge | GPQA Diamond | 0.372 | 0.783 |
| Knowledge | GSM8K | 0.579 | 0.799 |
| Knowledge | CRUXEval | 0.472 | 0.730 |
| Knowledge | CLadder | 0.637 | 0.726 |
| Knowledge | ChessBench | 0.135 | 0.172 |
| Language | ContractNLI | 0.761 | 0.717 |
| Language | iSarcasmEval | 0.452 | 0.505 |
| Language | VAST | 0.617 | 0.646 |
| Retrieval | BRIGHT | 0.172 | 0.187 |
| Retrieval | Amazon ESCI | 0.431 | 0.552 |
| Tools | BFCL | 0.903 | 0.958 |
| Tools | ToolRet | 0.412 | 0.450 |
| Tools | RouterBench | 0.790 | 0.799 |
| Arts | BPoMP | 0.847 | 0.906 |
| Arts | Humicroedit | 0.605 | 0.619 |
| Arts | POP909-CL | 0.034 | 0.181 |
| Arts | cfcolor | 0.581 | 0.647 |
| Arts | Habermas Machine | 0.443 | 0.459 |

</details>

## What we changed in the network

![blink-4b network diagram: 32 decoder layers repeating 3 Gated DeltaNet layers then one full-attention layer (24 and 8 in total, hidden 2560), with full attention at 0-based layers 3, 7, … 31 as in the tensor names; LoRA rank 16 on every attention, Gated DeltaNet and MLP projection (32.5M parameters, merged after training); token embeddings, norms and lm_head frozen, with lm_head tied to the token embeddings, one matrix; the vision encoder and multi-token-prediction head removed; and the answer read from the offered option-letter rows of lm_head.](../assets/blink-4b/blink-4b-network.png)

| | What ships |
|---|---|
| Backbone | Qwen3.5-4B text model; 32 decoder layers (24 Gated DeltaNet, 8 full-attention), hidden 2560; 4,205,751,296 shipped text parameters |
| Tuned | 32.5M LoRA parameters, merged before averaging |
| Final weights | Uniform weight average ("soup") of T3, T4 step 300 and T4 final |

T3/T4 also used KL anchors to the base model: its distributions on prompts whose teacher answers failed verification.

**LoRA targets (rank 16, alpha 32, every language-model layer):** full-attention `q_proj`, `k_proj`, `v_proj`, `o_proj`;
Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`; and every MLP's
`gate_proj`, `up_proj`, `down_proj`. Token embeddings, all norms and `lm_head` stayed frozen.
The trained adapters were merged into the text weights.

Qwen3.5-4B is a vision-language model; this checkpoint ships only its text model. The vision encoder and multi-token-prediction (MTP) head were cut: 0 vision tensors, 0 MTP tensors.

**The real cut is at readout:** no text generation. One prompt pass; next-token logits from only the
offered option-label rows of `lm_head` (verified single tokens A–Z, then two-letter labels), computed in
FP32 and softmaxed over those letters. The rest of the vocabulary is ignored.

**Objective:** "calibration-oriented decision post-training" is plain supervised fine-tuning.
Cross-entropy uses each row's target distribution: code-computed exact probabilities, probability
targets in teacher-written questions kept after a blind re-solve by that same teacher agreed, and
one-hot labels otherwise. For blink-4b, the base model's distributions on anchor rows are also targets. Choice and yes/no options and letter assignments are
reshuffled each epoch; score levels keep their order. Jev's RLCD recipe isn't public; we didn't
use or reproduce it. No RL or preference optimisation.

## The climb

![blink-4b post-training diagram: T3 (23,156 rows) and T4 (42,360 rows) broken down by data category feed supervised fine-tuning with cross-entropy over the offered option letters; both runs start from the base, and T3, T4 step 300 and T4 final are merged and averaged into release v1.0. Decision Index 0.1 full suite 52.12.](../assets/blink-4b/blink-4b-post-training.png)

DI-S is the 3,000-request sample. JevBench hard and ECE here are public-item numbers, not official scores.

| Step | DI 0.1 | Public hard | Hard ECE | Why |
|---|---:|---:|---:|---|
| Qwen3.5-4B, zero-shot | 44.45 DI-S | 0.595 | 0.131 | Baseline before decision training. |
| T1 (106.7k rows; lr 1e-4; 355 steps) | 53.15 DI-S | 0.559 | — | Public train splits plus exact-probability items lifted DI-S but hurt hard items; NLI/classification didn't transfer to long documents. |
| T3 (23,156 question rows; lr 3e-5; 96 steps) | — | 0.649 | 0.089 | Restarted from base with worlds (including "can't tell"), exact probabilities, teacher-written docs, ~10% public replay and 7.6% base anchors. |
| T4 (42,360 question rows; lr 4e-5; 472 steps) | — | 0.712 (step 300); 0.676 (final) | — | Added judge-style items; the earlier checkpoint did better on hard cases. |
| **Soup (T3 + T4 step 300 + T4 final)** | 52.12 full | 0.721 | 0.067 | Averaged three checkpoints for hard accuracy and calibration; shipped. |

**Tried, didn't keep:**
- Distilling 27B answers into 4B didn't help on hard items.
- A DI-focused 4B gained just +0.4 on DI-S.
- Mixing JevK5 weights into the soup didn't help.
- Qwen3.5-9B with the T1 recipe scored 54.5 DI-S, below our pre-set bar.

## Use

```python
# pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
import os, sys
from huggingface_hub import hf_hub_download

os.environ["BLINK_MODEL"] = "thegovind/blink-4b"
os.environ["BLINK_REVISION"] = "v1.2"
sys.path.insert(0, os.path.dirname(hf_hub_download("thegovind/blink-4b", "blink.py", revision="v1.2")))
import blink

out = blink.decide(
    "Order #4411 arrived with a cracked screen. I want my money back, not another one.",
    {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want?",
            "criteria": {"refund": "Money back", "replacement": "A new unit", "info": "Information only"},
        },
        "urgent": {"type": "noul", "instructions": "Does this need a reply today?"},
        "anger": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]},
    },
)
print(out["answers"]["intent"]["probabilities"])
```

## Run it as a server

`serve.py` handles TypeSafe's request and answer fields at `POST /v1/systemone` and lists its model at
`GET /v1/models`. From server-side code, point TypeSafe's Python or JavaScript SDK at the server with
`TYPESAFE_BASE_URL`;
JevBench's stock `typesafe` adapter and the Decision Index kit's `http` engine still work unchanged.
`GET /healthz` reports startup checks. `v1.2` changes code only; its weights are identical to `v1.0`.

```sh
pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
hf download thegovind/blink-4b --revision v1.2 --local-dir blink-4b
python blink-4b/serve.py --model ./blink-4b --port 8000
# TypeSafe SDKs: export TYPESAFE_BASE_URL=http://127.0.0.1:8000 TYPESAFE_API_KEY=any
```

In another terminal: `curl -s http://127.0.0.1:8000/healthz`.

**Or use Docker** from the downloaded folder:

```sh
cd blink-4b
docker build -t blink-4b . && docker run --rm --gpus all -p 127.0.0.1:8000:8000 blink-4b
```

<details><summary>Health, limits and weights</summary>

- `/healthz` reports `weights_verified` (weight, config and tokenizer files listed in `weights.sha256`
  are hashed before serving; a mismatch stops startup), `warmup.repeat_identical` (two matching warm-up
  answers), `kernels` (fast path or slower fallback without flash-linear-attention), `versions` and `hub_offline`.
- Limits: 255 options per choice, 2–10 score levels, 131,072 input tokens per question and 512 questions
  per request. Over-limit requests get HTTP 422 with the reason; nothing is truncated.
- `GET /v1/models` lists the one served model with a blank `release_date`. Every request uses that model
  regardless of its `model` field.
- The server is open by default. Set `--api-key` or `BLINK_API_KEY` to require `Authorization: Bearer <key>` on both API
  routes. Missing or wrong keys get 401; `/healthz` stays open.
- Error bodies put the reason in `error` and `detail`. Over-limit requests return 422, with nothing cut.
- Requests run one at a time. Questions are batched; each batch takes one forward pass (large requests
  can take more than one). Serving the downloaded folder or Docker image enables Hugging Face offline
  mode before model loading (`hub_offline: true`). The server doesn't otherwise restrict network access.
- Set `--batch-window-ms 5` to turn on cross-request batching with a 5 ms collection window, up to
  `--max-batch-requests` requests at a time, which defaults to 16. The window defaults to 0, so requests still
  run one at a time. `--max-queued-requests` lets up to 64 requests wait for a batch by default. Excess requests
  get HTTP 529 with `Retry-After`, so clients should retry. `v1.1` returned HTTP 503 for a full queue. On a
  1,000-request Decision Index sample over HTTP, throughput rose about 20% with 4 concurrent clients and 24%
  with 16. One client saw no gain. Offline runs on long documents showed no meaningful gain. On the Decision
  Index sample, a set of long workflow documents, and the public TypeSafe cases, batched answers passed the same
  numerical-parity checks against an FP32 reference as one-at-a-time answers, covering argmax agreement and
  probability differences. TypeSafe documents were sent as JSON objects. A few near-tied answers can still flip.
  Batching arrived in `v1.1`. The current code revision is `v1.2`, with weights identical to `v1.0`. Update the
  two code files in an existing `v1.0` download, then restart with the flag:

  ```sh
  hf download thegovind/blink-4b serve.py blink.py --revision v1.2 --local-dir blink-4b
  python blink-4b/serve.py --model ./blink-4b --port 8000 --batch-window-ms 5
  ```
- blink-4b weights are 8.4 GB in bf16. Long prompts need more memory.

</details>

<details><summary>Model and probability readout</summary>

| | |
|---|---|
| Base | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0), text weights only |
| Adaptation | LoRA (r16, α32) on attention, Gated DeltaNet and MLP projections; uniform average of three merged checkpoints from two runs |
| Readout | Verified single-token labels (A–Z, then two-letter labels); FP32 softmax of next-token logits / T over the offered labels |
| Temperature | 1.0 (not fitted; see evaluation notes) |
| Input limit | 131,072 tokens per question; longest evaluated prompt: 37,906 tokens; longer inputs are refused, never truncated |
| Runtime | `blink.py` builds prompts and reads out probabilities for text decisions |

These are option-conditional model probabilities, not certified chances of being right. Calibration can shift
across tasks, domains and option sets. For `choice`, `confidence = (p_max − 1/K)/(1 − 1/K)` measures
concentration, not correctness. For `score`, `score` is the expected 0-based level and `choice` is the
most likely level.

</details>

<details><summary>Training and data</summary>

### How it was trained

| Stage | Question rows | Mix |
|---|---:|---|
| T3 | 23,156 | 11,352 decision worlds · 3,741 teacher-written question rows · 2,579 exact-probability worlds · 2,221 public-source (~10%) · 1,763 base-model anchors (7.6%) · 1,500 program-generated reasoning |
| T4 | 42,360 | 12,000 decision worlds · 7,860 teacher-written question rows · 7,000 judge-style · 6,220 exact-probability worlds · 3,500 base-model anchors (8.3%) · 3,000 program-generated reasoning · 2,780 public-source |

T4 judge-style: 3,000 GSM8K-train solution checks, 2,500 Dolly-15k routing, 1,500 program-answer checks.

The released weights average three checkpoints fine-tuned from the base: T3 (lr 3e-5, 96 steps), T4 at step 300 and T4 at its final step 472 (lr 4e-5). The base-model anchor targets come from the base's own distributions on authored prompts whose teacher answers failed verification. Qwen3.8-27B wrote the teacher documents and their typed questions.

### Data sources and licences

| Source | Licence |
|---|---|
| MMLU auxiliary train, MMLU-Pro, CommonsenseQA, GSM8K | MIT |
| AQuA-RAT, Amazon ESCI | Apache-2.0 |
| MedMCQA | Apache-2.0 (dataset card) |
| SuperGPQA | ODC-BY |
| WANLI, ContractNLI, BANKING77, GPQA | CC BY 4.0 |
| ARC | CC BY-SA 4.0 |
| BoolQ, Dolly-15k | CC BY-SA 3.0 |
| ANLI | CC BY-NC 4.0 |
| SciQ | CC BY-NC 3.0 |
| iSarcasmEval | MIT (upstream repository licence) |
| VAST, Humicroedit, OpenBookQA | None stated by source |
| Our code-generated worlds and teacher-written documents (Qwen3.8-27B) | See LICENSE.md |

These are source-repository licences; they don't settle rights in every underlying text.

</details>

<details><summary>Evaluation notes and limits</summary>

### Evaluation notes

- **Scorer parity.** `blink.py` and the evaluation scorer matched prompts, labels and argmaxes on 296 requests (max |Δp| < 1e-7). From a fresh pinned install and a fresh Hub download, JevBench's stock `typesafe` adapter matched the development run on decision answers, probabilities and token usage across all 231 public items (easy 48/48, standard 71/72, hard 80/111); only the reported model identifier differed, so raw responses weren't byte-identical. The Decision Index kit's `http` engine scored DI-S 50.16 versus 50.08 in-process; its per-request timer on 1,000 random suite requests served serially measured a 66.4 ms median over HTTP versus 65.2 ms in-process.
- **Selection.** We reused DI-S, the official kit's 3,000-request sample of the 0.1 suite, to pick the prompt format and candidate checkpoints. We fixed blink-4b using JevBench development proxies before its full-suite run. It scored 52.29 on the 129,422 requests outside DI-S.
- **Training overlap.** Public train splits also used by the 0.1 index: ContractNLI, iSarcasmEval, VAST, Amazon ESCI, Humicroedit, GSM8K (train split; solution-checking items). We also used ANLI and BANKING77 train splits; they're in the 0.1 suite but outside its index, and both are in the 0.2 panel. The audit below reports what was checked and any shared passages.
- **Partitions.** Public-source data included training and development partitions, plus 281 MMLU-Pro test-partition questions and 2 GPQA extended-set questions outside GPQA Diamond; MMLU-Pro is not in the 0.1 suite but is in the 0.2 panel, so the 0.2 MMLU-Pro score is contaminated by those training questions and the 0.2 results are descriptive.
- **Final-mixture audit.** Rechecked every question row (including teacher-written rows, plus base-model anchors) against the complete 0.1 suite (132,422 requests) and JevBench's 231 public items. The checks looked for exact matches of normalised strings of at least 30 characters in any field and shared 13-word passages in each row's question text (instructions, state.question, state.code). Strings or passages seen in 20 or more suite requests were treated as prompt templates and ignored. No public JevBench item matched under these checks; a separate position check found no shared chess positions.
- **Suite overlap.** No content match with the 0.1 suite under these checks; all 36 flags were the fixed BANKING77 prompt template.
- **Audit limits.** The 13-word passage check didn't search long-document bodies or option text. Semantic or pretraining overlap can't be ruled out, and private JevBench items weren't available to check.
- **Generated reasoning.** Our programs computed the labels for CRUXEval-style code and CLadder-style causal questions; no items from those benchmarks were used. We didn't reuse the suite's GSM8K distractors.
- **Teacher documents.** We kept Qwen3.8-27B's documents only if a fresh blind solve by that same teacher agreed with the answer. That's an agreement filter, not independent verification.
- The repo ships no benchmark items, GPQA text, JevBench items or teacher traces.
- **JevBench selection.** We scored all 231 public items on each candidate checkpoint and used JevK5's 65 hand-written hard items (Apache-2.0) as a second selection set. None went into training; these are development results.
- **Temperature.** A split held out from T4 suggested T = 0.82, with negligible gain. But 166 of its 401 items were in T3 training, so it isn't held out from the released average. We kept T = 1.0 without fitting it.
- **One-time lockbox.** We read held-out authored items from domains unseen in any of the three checkpoints' training data (same generator families, not JevBench's sealed set) once: accuracy 0.861 and ECE 0.026 over 396 items.

### Limits

- English-centric. Training included Arabic iSarcasmEval rows; on the 0.1 suite, Arabic task A scored 0.313 and task C pairs 0.645. Broader multilingual performance hasn't been established.
- Doesn't chat or explain answers.
- Text in the state can sway the answer.
- Date arithmetic and long policies are its weakest cases.

</details>

## Licence

Qwen/Qwen3.5-4B is Apache-2.0 (`LICENSE-Qwen`). The blink weights are for **non-commercial research and evaluation
only** (`LICENSE.md`); commercial use isn't licensed. Training used non-commercial, share-alike and unlicensed
sources (see the table above). It's unsettled whether their terms reach the weights, so check upstream terms too.
`blink.py`, `serve.py` and the Dockerfile are Apache-2.0.

<details><summary>Credits</summary>

The Qwen team (base models). SemIf (MIT) for the evidence/criterion/options prompt layout. The Decision Index kit (MIT) and JevBench (MIT) for evaluation. JevK5 (Apache-2.0) for its hand-written hard items, used for evaluation only.

</details>
