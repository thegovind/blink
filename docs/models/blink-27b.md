---
license: other
license_name: blink-research
license_link: LICENSE.md
base_model: Qwen/Qwen3.8-27B
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

# blink-27b

Send a text or JSON `state` and your questions: `choice` picks from up to 255 options, `noul` is yes/no,
and `score` takes 2–10 ordered levels. Each question gets probabilities over its offered options from
one forward pass, with no generated text. Long or large multi-question requests may use several batches.

**Try it:** [Space demo](https://huggingface.co/spaces/thegovind/blink) — blink-4b and blink-mimo-9b run live; blink-27b isn't served ·
[blink-4b](https://huggingface.co/thegovind/blink-4b) · [blink-27b](https://huggingface.co/thegovind/blink-27b) ·
[blink-mimo-9b](https://huggingface.co/thegovind/blink-mimo-9b).

*Personal research release by thegovind, not an official product of any company. No affiliation with TypeSafe AI,
Xiaomi, Alibaba Cloud or the Qwen team. Weights are for non-commercial research; see [Licence](#licence).*

![blink-27b one-pass readout diagram: a state and typed choice, noul and score questions are rendered as evidence, criterion and lettered options; questions are batched and each batch is one forward pass, giving next-token logits at the answer position, and an FP32 softmax over the offered letters gives one probability per option. No text is generated.](../assets/blink-27b/blink-27b-readout.png)

## Results

### Decision Index 0.2 (local run)

We ran the full Decision Index 0.2 suite ourselves with the official scoring kit at commit 19ad28e on 2026-09-25. This is a descriptive run, not a leaderboard submission or accepted result. The kit's scorer does not apply the leaderboard's penalty for rows an entrant trained on, so known training exposure stays in these scores and they cannot be ranked against the leaderboard.

| Balanced skill | Balanced raw | Breadth skill | Without MMLU-Pro |
|---:|---:|---:|---:|
| 52.72 | 63.81 | 51.93 | 52.16 |

- Training included about 3.2k MMLU-Pro test-partition questions, so the 0.2 MMLU-Pro score is contaminated. “Without MMLU-Pro” drops that benchmark but does not remove other training effects. Public train splits used in training are listed under “Training overlap” below.

<details><summary>Extra tables and method</summary>

| Area | Number of benchmarks | Skill | Raw |
|---|---:|---:|---:|
| Knowledge & Reasoning | 10 | 45.0 | 56.7 |
| Language Understanding | 10 | 65.5 | 74.4 |
| Retrieval & Classification | 7 | 45.7 | 57.6 |
| Tools & Automation | 6 | 64.7 | 72.1 |
| Arts & Human Taste | 7 | 42.6 | 58.3 |

The seven benchmarks added in 0.2.

| Benchmark | Metric | Requests | Answered | Raw | Skill |
|---|---|---:|---:|---:|---:|
| PhishNChips phishing decisions | accuracy | 2,000 | 2,000 | 68.5 | 37.1 |
| MMLU-Pro | accuracy | 12,032 | 12,032 | 73.3 | 70.0 |
| BBH fixed-option tasks | accuracy | 5,507 | 5,507 | 78.1 | 68.3 |
| RAGTruth response-level hallucination | F1 on hallucinated class | 2,700 | 2,700 | 81.1 | 67.9 |
| HoVer claim verification | accuracy | 4,000 | 4,000 | 73.8 | 47.6 |
| When2Call MCQ | accuracy | 3,652 | 3,652 | 69.8 | 59.8 |
| New Yorker caption matching | accuracy | 528 | 528 | 71.8 | 64.7 |

- All 151,034 of 151,034 scoreable requests scored. Of 44 scored benchmarks, 40 count toward the index across five equal areas.
- Requests shared with 0.1 reuse the model's 0.1 predictions. We ran the 30,419 added requests with the same frozen evaluation setup as 0.1, the evaluated T4 adapter on the base model, which the published merged weights matched on JevBench's 231 public items (see Evaluation notes), at temperature 1.0.
- Balanced skill is the headline index. “Without MMLU-Pro” drops MMLU-Pro, averages the other nine Knowledge benchmarks, and keeps five equal areas. It is a sensitivity check, not a score free of training effects.
- These are point estimates, with no significance, calibration, or latency claims. Do not compare them with 0.1 numbers because the editions differ.

Training-row text matches in the added requests.

| Training stage | Rows in the stage | Rows matching added-request text | From MMLU-Pro | From SuperGPQA | Other |
|---|---:|---:|---:|---:|---:|
| T2 | 72,700 | 2,125 | 2,009 | 113 | 3 |
| T4 | 74,754 | 1,281 | 1,217 | 63 | 1 |

We screened for exact normalised strings of at least 30 characters shared by training rows and added requests, ignoring strings found in 20 or more requests as templates. Counts are training rows by stage and source, not unique test questions. A matching option or passage need not be the same question, and a clean screen cannot rule out semantic or pretraining overlap.

We did not produce the planned calibration read or a score without the DI-S selection sample.

</details>

### Decision Index 0.1 (archived edition)

**Full suite: blink-27b 63.44 vs Jev 1.13.0 59.51.**

**Important context:** Training used the public train splits of ContractNLI, iSarcasmEval and VAST (the whole Language area), plus Amazon ESCI and Humicroedit. Set Language to Jev's score and this model's index would be 60.44 against Jev's 59.51: 3.00 of the 3.93-point lead is in that area. That's arithmetic, not an ablation or a like-for-like comparison with systems trained only on synthetic data.

**Selection:** The full suite picked T4 over T2, so both finalists' full-suite and minus-DI-S scores are post-selection. [Both rows are in the climb](#the-climb).

| Model | Size class | Decision Index 0.1 | Skill | Breadth |
|---|---|---:|---:|---:|
| **blink-27b** (this model) | 27B | 63.44 | 51.47 | 49.91 |
| Jev 1.13.0 | closed | 59.51 | 46.26 | 44.79 |
| Jevfire | 27B | 55.74 | 40.86 | 39.45 |
| JoshuaSP diffusiongemma (open-jev) | 26B-A4B | 55.56 | 40.84 | 39.19 |
| Decider 35B-A3B | 35B-A3B | 54.34 | 39.37 | 37.99 |
| Kev 9B | 9B | 50.48 | 32.96 | 30.54 |
| Kev 4B | 4B | 47.43 | 28.86 | 25.67 |

We ran the complete archived 0.1 suite: 132,422 requests across 37 benchmarks. The headline index averages 19 panel benchmarks. Comparison rows use the 2026-09-22 leaderboard snapshot. We ran the official kit's scorer locally; these aren't leaderboard submissions. The live [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) moved to 0.2 on 2026-09-24. Our local 0.2 run is in the section above.

| Area | blink-27b | Jev 1.13.0 |
|---|---:|---:|
| Knowledge & Reasoning | 65.3 | 68.8 |
| Language Understanding | 77.3 | 62.3 |
| Retrieval & Classification | 38.9 | 37.0 |
| Tools & Automation | 74.3 | 73.6 |
| Arts & Human Judgment | 61.5 | 56.2 |

<details><summary>Per benchmark (19 panel benchmarks, 0.1)</summary>

| Area | Benchmark | This model | Jev 1.13.0 |
|---|---|---:|---:|
| Knowledge | MMLU | 0.861 | 0.917 |
| Knowledge | GPQA Diamond | 0.531 | 0.783 |
| Knowledge | GSM8K | 0.790 | 0.799 |
| Knowledge | CRUXEval | 0.730 | 0.730 |
| Knowledge | CLadder | 0.752 | 0.726 |
| Knowledge | ChessBench | 0.252 | 0.172 |
| Language | ContractNLI | 0.845 | 0.717 |
| Language | iSarcasmEval | 0.678 | 0.505 |
| Language | VAST | 0.795 | 0.646 |
| Retrieval | BRIGHT | 0.199 | 0.187 |
| Retrieval | Amazon ESCI | 0.578 | 0.552 |
| Tools | BFCL | 0.968 | 0.958 |
| Tools | ToolRet | 0.463 | 0.450 |
| Tools | RouterBench | 0.797 | 0.799 |
| Arts | BPoMP | 0.935 | 0.906 |
| Arts | Humicroedit | 0.635 | 0.619 |
| Arts | POP909-CL | 0.393 | 0.181 |
| Arts | cfcolor | 0.657 | 0.647 |
| Arts | Habermas Machine | 0.456 | 0.459 |

</details>

## What we changed in the network

![blink-27b network diagram: 64 decoder layers repeating 3 Gated DeltaNet layers then one full-attention layer (48 and 16 in total, hidden 5120), with full attention at 0-based layers 3, 7, … 63 as in the tensor names; LoRA rank 16 on every attention, Gated DeltaNet and MLP projection (116.7M parameters, merged after training); token embeddings, norms and lm_head frozen, with lm_head untied, a separate matrix; the vision encoder and multi-token-prediction head removed; and the answer read from the offered option-letter rows of lm_head.](../assets/blink-27b/blink-27b-network.png)

| | What ships |
|---|---|
| Backbone | Qwen3.8-27B text model; 64 decoder layers (48 Gated DeltaNet, 16 full-attention), hidden 5120; 26,895,998,464 shipped text parameters |
| Tuned | 116.7M LoRA parameters, merged into the weights |
| Final weights | One run in two stages: T2, then T4 |

**LoRA targets (rank 16, alpha 32, every language-model layer):** full-attention `q_proj`, `k_proj`, `v_proj`, `o_proj`;
Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`; and every MLP's
`gate_proj`, `up_proj`, `down_proj`. Token embeddings, all norms and `lm_head` stayed frozen.
The trained adapters were merged into the text weights.

Qwen3.8-27B is a vision-language model; this checkpoint ships only its text model. The vision encoder and multi-token-prediction (MTP) head were cut: 0 vision tensors, 0 MTP tensors.

**The real cut is at readout:** no text generation. One prompt pass; next-token logits from only the
offered option-label rows of `lm_head` (verified single tokens A–Z, then two-letter labels), computed in
FP32 and softmaxed over those letters. The rest of the vocabulary is ignored.

**Objective:** "calibration-oriented decision post-training" is plain supervised fine-tuning.
Cross-entropy uses each row's target distribution: code-computed exact probabilities, probability
targets in teacher-written questions kept after a blind re-solve by that same teacher agreed, and
one-hot labels otherwise. Choice and yes/no options and letter assignments are
reshuffled each epoch; score levels keep their order. Jev's RLCD recipe isn't public; we didn't
use or reproduce it. No RL or preference optimisation.

## The climb

![blink-27b post-training diagram: T2 (72,700 rows) and T4 (74,754 rows) broken down by data category feed supervised fine-tuning with cross-entropy over the offered option letters; T2 starts from the base and T4 continues from it, then the adapters are merged into release v1.0. Decision Index 0.1 full suite 63.44.](../assets/blink-27b/blink-27b-post-training.png)

DI-S is the 3,000-request selection sample; full and outside-DI-S scores use the archived 0.1 suite.

| Step | DI-S | Full 0.1 | Outside DI-S | Why |
|---|---:|---:|---:|---|
| Qwen3.8-27B, zero-shot | 56.68 | — | — | JSON evidence/criterion/options beat a markdown layout (52.53); kept JSON for every run. |
| T2 (72,700 question rows; lr 5e-5; 230 steps) | 61.42 | 62.51 | 62.63 | Public train splits, code-computed reasoning and decision worlds got the first full-suite result. |
| **T4** (74,754 question rows; lr 3e-5; 856 more steps) | 64.18 | 63.44 | 63.46 | More program-labelled reasoning, chess moves, teacher-written docs and exact-probability worlds; shipped. |

## Use

```python
# pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
import os, sys
from huggingface_hub import hf_hub_download

os.environ["BLINK_MODEL"] = "thegovind/blink-27b"
os.environ["BLINK_REVISION"] = "v1.2"
sys.path.insert(0, os.path.dirname(hf_hub_download("thegovind/blink-27b", "blink.py", revision="v1.2")))
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
hf download thegovind/blink-27b --revision v1.2 --local-dir blink-27b
python blink-27b/serve.py --model ./blink-27b --port 8000
# TypeSafe SDKs: export TYPESAFE_BASE_URL=http://127.0.0.1:8000 TYPESAFE_API_KEY=any
```

In another terminal: `curl -s http://127.0.0.1:8000/healthz`.

**Or use Docker** from the downloaded folder:

```sh
cd blink-27b
docker build -t blink-27b . && docker run --rm --gpus all -p 127.0.0.1:8000:8000 blink-27b
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
- `--batch-window-ms` is available in the `blink-27b` `v1.2` code. Its speed and numerical parity have not been
  measured for this model.
- blink-27b weights are 53.8 GB in bf16. Long prompts need more memory.

</details>

<details><summary>Model and probability readout</summary>

| | |
|---|---|
| Base | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0), text weights only |
| Adaptation | LoRA (r16, α32) on attention, Gated DeltaNet and MLP projections; T2 then T4, merged into the base weights |
| Readout | Verified single-token labels (A–Z, then two-letter labels); FP32 softmax of next-token logits / T over the offered labels |
| Temperature | 1.0 (not fitted) |
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
| T2 | 72,700 | 41,401 public-source · 16,000 program-generated reasoning · 12,000 decision worlds · 3,299 exact-probability worlds |
| T4 | 74,754 | 23,894 program-generated reasoning · 23,252 public-source · 12,000 decision worlds · 7,860 teacher-written question rows · 6,000 chess move choices · 1,748 exact-probability worlds |

T2 program reasoning: 4,075 code execution, 4,030 logic grids, 3,977 word problems, 3,918 causal.

T4 program reasoning: 13,894 code execution, 6,000 causal, 2,500 word problems, 1,500 logic grids.

T2 started from the base (lr 5e-5, 230 steps); T4 continued it (lr 3e-5, 856 more steps). About 2k early T2 decision-world rows came from a generator with ambiguous counting answers or surface cues in "cannot determine" cases. Those rows stayed in training; T4 used corrected generators.

### Data sources and licences

| Source | Licence |
|---|---|
| MMLU auxiliary train, MMLU-Pro, CommonsenseQA | MIT |
| AQuA-RAT, Amazon ESCI | Apache-2.0 |
| searchless_chess | data CC BY 4.0 (Lichess-derived portions CC0); code Apache-2.0 |
| MedMCQA | Apache-2.0 (dataset card) |
| SuperGPQA | ODC-BY |
| WANLI, ContractNLI, BANKING77, GPQA | CC BY 4.0 |
| ARC | CC BY-SA 4.0 |
| BoolQ | CC BY-SA 3.0 |
| ANLI | CC BY-NC 4.0 |
| SciQ | CC BY-NC 3.0 |
| iSarcasmEval | MIT (upstream repository licence) |
| VAST, Humicroedit, OpenBookQA | None stated by source |
| Our code-generated worlds and teacher-written documents (Qwen3.8-27B) | See LICENSE.md |

These are source-repository licences; they don't settle rights in every underlying text.

</details>

<details><summary>Evaluation notes and limits</summary>

### Evaluation notes

- **Scorer parity.** `blink.py` and the evaluation scorer matched prompts, labels and argmaxes on 296 requests (max |Δp| < 1e-7). With matched batching, the published merged weights matched the evaluated adapter on JevBench's 231 public items (0 argmax changes, max |Δp| < 1e-4). The Decision Index kit's per-request timer on 1,000 random suite requests served serially measured a 192.2 ms median over HTTP versus 190.0 ms in-process. From a fresh Hub download and install at the pinned revision, `serve.py` with JevBench's stock adapter returned easy 48/48, standard 71/72 and hard 89/111 versus 90/111 hard in the padded-batch evaluation: one near-tie flipped (p = 0.508 batched vs 0.500 served), 1/231 answers changed, max |Δp| 0.022; serial serving changes bf16 numerics, so this is not exact reproduction.
- **Selection.** We reused DI-S, the official kit's 3,000-request sample of the 0.1 suite, to pick the prompt format and candidate checkpoints. On the 129,422 requests outside DI-S, T4 scored 63.46.
- **Training overlap.** Public train splits also used by the 0.1 index: ContractNLI, iSarcasmEval, VAST, Amazon ESCI, Humicroedit, ChessBench (searchless_chess training positions; none of the 5,000 test positions). We also used ANLI and BANKING77 train splits; they're in the 0.1 suite but outside its index, and both are in the 0.2 panel. The audit below reports what was checked and any shared passages.
- **Partitions.** Public-source data included training and development partitions, plus about 3.2k MMLU-Pro test-partition questions and 19 GPQA extended-set questions outside GPQA Diamond; MMLU-Pro is not in the 0.1 suite but is in the 0.2 panel, so the 0.2 MMLU-Pro score is contaminated by those training questions and the 0.2 results are descriptive.
- **Final-mixture audit.** Rechecked every question row (including teacher-written rows) against the complete 0.1 suite (132,422 requests) and JevBench's 231 public items. The checks looked for exact matches of normalised strings of at least 30 characters in any field and shared 13-word passages in each row's question text (instructions, state.question, state.code). Strings or passages seen in 20 or more suite requests were treated as prompt templates and ignored. No public JevBench item matched under these checks; a separate position check found no shared chess positions.
- **Suite overlap.** 16 BANKING77/VAST training rows share a 13-word passage with 31 suite requests: 23 of VAST's 3,006 and 8 of BANKING77's 3,080. Two VAST training posts are near-duplicates of a test post, but these rows had no exact normalised-text match under the audit. Dropping those requests leaves the index at 63.44 (VAST 0.7955 → 0.7954); BANKING77 is outside the 0.1 index.
- **Audit limits.** The 13-word passage check didn't search long-document bodies or option text. Semantic or pretraining overlap can't be ruled out, and private JevBench items weren't available to check.
- **Generated reasoning.** Our programs computed the labels for CRUXEval-style code and CLadder-style causal questions; no items from those benchmarks were used. We didn't reuse the suite's GSM8K distractors.
- **Teacher documents.** We kept Qwen3.8-27B's documents only if a fresh blind solve by that same teacher agreed with the answer. That's an agreement filter, not independent verification.
- The repo ships no benchmark items, GPQA text, JevBench items or teacher traces.

### Limits

- English-centric. Training included Arabic iSarcasmEval rows; on the 0.1 suite, Arabic task A scored 0.425 and task C pairs 0.85. Broader multilingual performance hasn't been established.
- Doesn't chat or explain answers.
- Text in the state can sway the answer.
- It trails Jev on MMLU and GPQA Diamond (see the benchmark table).
- Date and number reasoning is its weakest type on JevBench's public hard items (temporal/numeric: 0.40).
- Too costly for JevBench's cost axis; use blink-4b there.

</details>

## Licence

Qwen/Qwen3.8-27B is Apache-2.0 (`LICENSE-Qwen`). The blink weights are for **non-commercial research and evaluation
only** (`LICENSE.md`); commercial use isn't licensed. Training used non-commercial, share-alike and unlicensed
sources (see the table above). It's unsettled whether their terms reach the weights, so check upstream terms too.
`blink.py`, `serve.py` and the Dockerfile are Apache-2.0.

<details><summary>Credits</summary>

The Qwen team (base models). SemIf (MIT) for the evidence/criterion/options prompt layout. The Decision Index kit (MIT) and JevBench (MIT) for evaluation. JevK5 (Apache-2.0) for its hand-written hard items, used for evaluation only.

</details>
