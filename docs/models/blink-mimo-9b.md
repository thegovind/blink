---
license: other
license_name: blink-research
license_link: LICENSE.md
base_model: XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
library_name: transformers
pipeline_tag: image-text-to-text
inference: false
tags:
  - decision-model
  - typed-decisions
  - one-pass
  - option-probabilities
language:
  - en
---

# blink-mimo-9b

Send a text or JSON `state` and your questions: `choice` picks from up to 255 options, `noul` is yes/no,
and `score` takes 2–10 ordered levels. Each question gets probabilities over its offered options from
one forward pass, with no generated text. Long or large multi-question requests may use several batches.

**Try it:** [Space demo](https://huggingface.co/spaces/thegovind/blink) — blink-4b and blink-mimo-9b run live ·
[blink-4b](https://huggingface.co/thegovind/blink-4b) · [blink-27b](https://huggingface.co/thegovind/blink-27b) ·
[blink-mimo-9b](https://huggingface.co/thegovind/blink-mimo-9b).

*Personal research release by thegovind, not an official product of any company. No affiliation with TypeSafe AI,
Xiaomi, Alibaba Cloud or the Qwen team. Weights are for non-commercial research; see [Licence](#licence).*

![blink-mimo-9b one-pass readout diagram: a state and typed choice, noul and score questions are rendered as evidence, criterion and lettered options; questions are batched and each batch is one forward pass, giving next-token logits at the answer position, and an FP32 softmax over the offered letters gives one probability per option. No text is generated.](assets/blink-mimo-9b-readout.png)

## Results

### Decision Index 0.2 (local run)

We ran the full Decision Index 0.2 suite ourselves with the official scoring kit at commit 19ad28e on 2026-09-25. This is a descriptive run, not a leaderboard submission or accepted result. The kit's scorer does not apply the leaderboard's penalty for rows an entrant trained on, so known training exposure stays in these scores and they cannot be ranked against the leaderboard.

| Balanced skill | Balanced raw | Breadth skill | Without MMLU-Pro |
|---:|---:|---:|---:|
| 43.36 | 57.24 | 42.38 | 42.84 |

- Our Blink fine-tune did not use MMLU-Pro or GPQA as direct data sources. Screening found 176 SuperGPQA training rows matching added-request text. SuperGPQA is a training source, not a benchmark in this suite. Public train splits used in training are listed under “Training overlap” below.

<details><summary>Extra tables and method</summary>

| Area | Number of benchmarks | Skill | Raw |
|---|---:|---:|---:|
| Knowledge & Reasoning | 10 | 33.2 | 48.2 |
| Language Understanding | 10 | 54.1 | 67.2 |
| Retrieval & Classification | 7 | 40.5 | 55.4 |
| Tools & Automation | 6 | 57.0 | 65.2 |
| Arts & Human Taste | 7 | 32.0 | 50.2 |

The seven benchmarks added in 0.2.

| Benchmark | Metric | Requests | Answered | Raw | Skill |
|---|---|---:|---:|---:|---:|
| PhishNChips phishing decisions | accuracy | 2,000 | 2,000 | 66.5 | 33.1 |
| MMLU-Pro | accuracy | 12,032 | 12,032 | 61.4 | 56.5 |
| BBH fixed-option tasks | accuracy | 5,507 | 5,507 | 67.8 | 53.4 |
| RAGTruth response-level hallucination | F1 on hallucinated class | 2,700 | 2,700 | 63.4 | 37.8 |
| HoVer claim verification | accuracy | 4,000 | 4,000 | 65.6 | 31.3 |
| When2Call MCQ | accuracy | 3,652 | 3,652 | 64.5 | 52.7 |
| New Yorker caption matching | accuracy | 528 | 528 | 62.5 | 53.1 |

- All 151,034 of 151,034 scoreable requests scored. Of 44 scored benchmarks, 40 count toward the index across five equal areas.
- Requests shared with 0.1 reuse the model's 0.1 predictions. We ran the 30,419 added requests with the same frozen evaluation setup as 0.1, the evaluated adapter loaded on the MiMo base, which the published graft matched on JevBench's 231 public items (see Evaluation notes), at temperature 1.0.
- Balanced skill is the headline index. “Without MMLU-Pro” drops MMLU-Pro, averages the other nine Knowledge benchmarks, and keeps five equal areas. It is a sensitivity check, not a score free of training effects.
- These are point estimates, with no significance, calibration, or latency claims. Do not compare them with 0.1 numbers because the editions differ.

Training-row text matches in the added requests.

| Training stage | Rows in the stage | Rows matching added-request text | From MMLU-Pro | From SuperGPQA | Other |
|---|---:|---:|---:|---:|---:|
| MiMo | 123,195 | 180 | 0 | 176 | 4 |

We screened for exact normalised strings of at least 30 characters shared by training rows and added requests, ignoring strings found in 20 or more requests as templates. Counts are training rows by stage and source, not unique test questions. A matching option or passage need not be the same question, and a clean screen cannot rule out semantic or pretraining overlap.

We did not produce the planned calibration read or a score without the DI-S selection sample.

</details>

### Decision Index 0.1 (archived edition)

**Full suite: blink-mimo-9b 56.53 vs Jev 1.13.0 59.51.**

**Training overlap:** Public train splits included ContractNLI, iSarcasmEval and VAST (the whole Language area), plus Amazon ESCI and Humicroedit. With Language set to Jev's score, this model's index would be 54.96 vs Jev's 59.51. That's arithmetic, not an ablation or a like-for-like comparison with models trained only on synthetic data.

| Model | Size class | Decision Index 0.1 | Skill | Breadth |
|---|---|---:|---:|---:|
| **blink-mimo-9b** (this model) | 9B | 56.53 | 42.17 | 40.37 |
| Jev 1.13.0 | closed | 59.51 | 46.26 | 44.79 |
| Jevfire | 27B | 55.74 | 40.86 | 39.45 |
| JoshuaSP diffusiongemma (open-jev) | 26B-A4B | 55.56 | 40.84 | 39.19 |
| Decider 35B-A3B | 35B-A3B | 54.34 | 39.37 | 37.99 |
| Kev 9B | 9B | 50.48 | 32.96 | 30.54 |
| Kev 4B | 4B | 47.43 | 28.86 | 25.67 |

We ran the complete archived 0.1 suite: 132,422 requests across 37 benchmarks. The headline index averages 19 panel benchmarks. Comparison rows use the 2026-09-22 leaderboard snapshot. We ran the official kit's scorer locally; these aren't leaderboard submissions. The live [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) moved to 0.2 on 2026-09-24. Our local 0.2 run is in the section above.

Third in our local archived 0.1 comparison, behind blink-27b and Jev and ahead of every open entry in the September 22 snapshot (best: Jevfire).

| Area | blink-mimo-9b | Jev 1.13.0 |
|---|---:|---:|
| Knowledge & Reasoning | 55.1 | 68.8 |
| Language Understanding | 70.1 | 62.3 |
| Retrieval & Classification | 34.8 | 37.0 |
| Tools & Automation | 70.5 | 73.6 |
| Arts & Human Judgment | 52.1 | 56.2 |

<details><summary>Per benchmark (19 panel benchmarks, 0.1)</summary>

| Area | Benchmark | This model | Jev 1.13.0 |
|---|---|---:|---:|
| Knowledge | MMLU | 0.802 | 0.917 |
| Knowledge | GPQA Diamond | 0.408 | 0.783 |
| Knowledge | GSM8K | 0.658 | 0.799 |
| Knowledge | CRUXEval | 0.547 | 0.730 |
| Knowledge | CLadder | 0.661 | 0.726 |
| Knowledge | ChessBench | 0.229 | 0.172 |
| Language | ContractNLI | 0.817 | 0.717 |
| Language | iSarcasmEval | 0.506 | 0.505 |
| Language | VAST | 0.780 | 0.646 |
| Retrieval | BRIGHT | 0.177 | 0.187 |
| Retrieval | Amazon ESCI | 0.520 | 0.552 |
| Tools | BFCL | 0.893 | 0.958 |
| Tools | ToolRet | 0.422 | 0.450 |
| Tools | RouterBench | 0.799 | 0.799 |
| Arts | BPoMP | 0.841 | 0.906 |
| Arts | Humicroedit | 0.638 | 0.619 |
| Arts | POP909-CL | 0.076 | 0.181 |
| Arts | cfcolor | 0.597 | 0.647 |
| Arts | Habermas Machine | 0.455 | 0.459 |

</details>

### JevBench: public items only

| Model | Easy | Standard | Hard | Hard ECE | Official score |
|---|---:|---:|---:|---:|---|
| **blink-mimo-9b** | 48/48 | 70/72 | 77/111 (0.694) | 0.136 | not submitted |

Same public-item harness as the other models. These are development results, not official scores or predictions; no official rank or parity is claimed. blink-4b is the JevBench entry. MiMo's hard ECE is 0.136 vs blink-4b's 0.067; it's less calibrated here. Its weakest public hard cases are date/number reasoning, trade-offs and long policies.

### Picking the next click (our probe)

| Input | blink-mimo-9b | MiMo base | blink-4b |
|---|---:|---:|---:|
| Page text | 53.8% | 49.0% | 55.4% |
| Screenshot | 48.4% | 43.0% | — |

Our harness and sampling on 500 Multimodal-Mind2Web test steps with five offered elements (chance 20%), not the official evaluation. Five choices are easier than ranking a whole page; blink never trained on web actions. Training lifted MiMo by about five points in both modes, but text still beats screenshots; blink-4b is as good with page text.

## What we changed in the network

![blink-mimo-9b network diagram: 32 decoder layers repeating 3 Gated DeltaNet layers then one full-attention layer (24 and 8 in total, hidden 4096), with full attention at 0-based layers 3, 7, … 31 as in the tensor names; LoRA rank 16 on every attention, Gated DeltaNet and MLP projection (43.3M parameters, merged after training); token embeddings, norms and lm_head frozen, with lm_head untied, a separate matrix; the 27-block vision tower kept byte-for-byte; and the answer read from the offered option-letter rows of lm_head.](assets/blink-mimo-9b-network.png)

| | What ships |
|---|---|
| Backbone | MiMo-V2.6-Distill-Qwen-9B (upstream revision 2367e86), a Qwen3.5-9B fine-tune; 32 decoder layers (24 Gated DeltaNet, 8 full-attention), hidden 4096, untied embeddings |
| Vision | 27 encoder blocks, hidden 1152 projected to 4096; all 333 vision tensors unchanged |
| Tuned | 43.3M LoRA parameters; all 248 targeted tensors changed, all 179 other language-model tensors bit-identical to MiMo |
| Final weights | Merged and grafted into MiMo; vision-language reload with no missing or unexpected keys |
| Serving | 9.41B parameters in the full checkpoint (0.46B vision); 8.95B text parameters serving decisions (17.9 GB in bf16); 18.8 GB total |

**LoRA targets (rank 16, alpha 32, every language-model layer):** full-attention `q_proj`, `k_proj`, `v_proj`, `o_proj`;
Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`; and every MLP's
`gate_proj`, `up_proj`, `down_proj`. Token embeddings, all norms and `lm_head` stayed frozen.
The trained adapters were merged into the text weights.

MiMo's vision tower stays byte-for-byte unchanged. The full checkpoint loads with MiMo's processor and transformers' image-text-to-text classes; `blink.py` and `serve.py` use its text side only. Screenshot results above come from our separate probe harness, not `blink.py`.

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

![blink-mimo-9b post-training diagram: one run (123,195 rows) broken down by data category feed supervised fine-tuning with cross-entropy over the offered option letters; a single pre-registered epoch is merged and grafted back into the full vision-language checkpoint, vision tower unchanged, as release v1.0. Decision Index 0.1 full suite 56.53.](assets/blink-mimo-9b-post-training.png)

One pre-registered run, one epoch; no other MiMo variant was trained or picked. The gate was DI-S ≥ 55 before training: pass it, then read the full suite and release. DI-S intervals are a few points wide, so this is a narrow gate pass, not a claim of superiority.

| Step | DI-S | Full 0.1 | Outside DI-S | Why |
|---|---:|---:|---:|---|
| MiMo base, zero-shot | 47.09 | — | — | Baseline before decision training. |
| **blink-mimo-9b** (123,195 question rows; lr 5e-5; 615 steps) | 55.3 | 56.53 | 56.60 | Program-labelled reasoning, teacher questions and judge data cleared the gate; shipped. |

DI-S areas: Knowledge 55.6, Language 66.4, Retrieval 33.0, Tools 70.7, Arts 50.8.

## Use

```python
# pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
import os, sys
from huggingface_hub import hf_hub_download

os.environ["BLINK_MODEL"] = "thegovind/blink-mimo-9b"
os.environ["BLINK_REVISION"] = "v1.0"
sys.path.insert(0, os.path.dirname(hf_hub_download("thegovind/blink-mimo-9b", "blink.py", revision="v1.0")))
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

`serve.py` accepts Jev-compatible `POST /v1/systemone` (`{state, questions}` → `{answers, usage}`);
JevBench's stock `typesafe` adapter and the Decision Index kit's `http` engine use this format.
`GET /healthz` reports startup checks.

```sh
pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
hf download thegovind/blink-mimo-9b --revision v1.0 --local-dir blink-mimo-9b
python blink-mimo-9b/serve.py --model ./blink-mimo-9b --port 8000
```

In another terminal: `curl -s http://127.0.0.1:8000/healthz`.

**Or use Docker** from the downloaded folder:

```sh
cd blink-mimo-9b
docker build -t blink-mimo-9b . && docker run --rm --gpus all -p 127.0.0.1:8000:8000 blink-mimo-9b
```

<details><summary>Health, limits and weights</summary>

- `/healthz` reports `weights_verified` (weight, config and tokenizer files listed in `weights.sha256`
  are hashed before serving; a mismatch stops startup), `warmup.repeat_identical` (two matching warm-up
  answers), `kernels` (fast path or slower fallback without flash-linear-attention), `versions` and `hub_offline`.
- Limits: 255 options per choice, 2–10 score levels, 131,072 input tokens per question and 512 questions
  per request. Over-limit requests get HTTP 422 with the reason; nothing is truncated.
- Requests run one at a time. Questions are batched; each batch takes one forward pass (large requests
  can take more than one). Serving the downloaded folder or Docker image enables Hugging Face offline
  mode before model loading (`hub_offline: true`). The server doesn't otherwise restrict network access.
- Set `--batch-window-ms 5` to turn on cross-request batching with a 5 ms collection window, up to
  `--max-batch-requests` requests at a time, which defaults to 16. The window defaults to 0, so requests still
  run one at a time. `--max-queued-requests` lets up to 64 requests wait for a batch by default. Excess requests
  get HTTP 503 with `Retry-After`, so clients should retry. On a 1,000-request Decision Index sample over HTTP,
  throughput rose about 15% with 4 concurrent clients and 16% with 16. One client saw no gain. Offline runs on
  long documents showed no meaningful gain. On the Decision Index sample, a set of long workflow documents, and
  the public TypeSafe cases covered by the FP32 reference, batched answers passed the same numerical-parity
  checks against an FP32 reference as one-at-a-time answers, covering argmax agreement and probability
  differences. TypeSafe documents were sent as JSON objects. The FP32 reference could not run the five longest
  TypeSafe documents, so parity covered 170 of 354 TypeSafe questions. Batched and one-at-a-time answers chose
  the same option every time on the other 184. A few near-tied answers can still flip. `v1.1` changes code only,
  leaving `v1.0` weights unchanged. Update the two code files in an existing `v1.0` download, then restart with
  the flag:

  ```sh
  hf download thegovind/blink-mimo-9b serve.py blink.py --revision v1.1 --local-dir blink-mimo-9b
  python blink-mimo-9b/serve.py --model ./blink-mimo-9b --port 8000 --batch-window-ms 5
  ```
- blink-mimo-9b weights are 18.8 GB in bf16. Long prompts need more memory.

</details>

<details><summary>Model and probability readout</summary>

| | |
|---|---|
| Base | [XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) (MIT declared in the model card; see LICENSE-MiMo.md), full vision-language weights |
| Adaptation | LoRA r16 / alpha 32 across the language model's attention, Gated DeltaNet and MLP; merged into the full MiMo checkpoint |
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
| MiMo | 123,195 | 61,394 public-source · 23,894 program-generated reasoning · 12,000 decision worlds · 7,860 teacher-written question rows · 7,000 judge-style · 6,000 chess move choices · 5,047 exact-probability worlds |

One pre-registered supervised run, one epoch (lr 5e-5, 615 steps). The mix joins the 27B's stage-1 public and exact-probability rows with its stage-2 reasoning, chess and decision worlds, plus judge-style data. Qwen3.8-27B wrote the teacher documents and typed questions. Public train splits include ContractNLI, iSarcasmEval, VAST, Amazon ESCI, Humicroedit, ANLI, WANLI, BoolQ, BANKING77, MedMCQA, AQuA, MMLU auxiliary train, SciQ and SuperGPQA.

### Data sources and licences

| Source | Licence |
|---|---|
| MMLU auxiliary train, CommonsenseQA, GSM8K | MIT |
| AQuA-RAT, Amazon ESCI | Apache-2.0 |
| searchless_chess | data CC BY 4.0 (Lichess-derived portions CC0); code Apache-2.0 |
| MedMCQA | Apache-2.0 (dataset card) |
| SuperGPQA | ODC-BY |
| WANLI, ContractNLI, BANKING77 | CC BY 4.0 |
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

- **Scorer parity.** `blink.py` and the lab scorer agree within 1e-7 on this checkpoint (p99 |Δp| 1.4e-8); the published graft matches the evaluated adapter on JevBench's 231 public items (0 argmax changes, max |Δp| 0.0). The kit's per-request timer on 1,000 random suite requests served one at a time measured 68.3 ms median via HTTP vs 66.9 ms in-process. From a fresh Hub download and install at the pinned revision, `serve.py` with JevBench's stock adapter returned easy 48/48, standard 70/72 and hard 77/111, matching the evaluation's answers (0 argmax changes; max |Δp| about 0.03).
- **Selection.** The prompt format came from earlier DI-S reads; this MiMo run used the sample as a pre-registered gate. The full suite followed that gate; it scored 56.60 on the 129,422 requests outside DI-S, which were not used for selection.
- **Training overlap.** Public train splits also used by the 0.1 index: ContractNLI, iSarcasmEval, VAST, Amazon ESCI, Humicroedit, ChessBench (searchless_chess training positions; none of the 5,000 test positions), GSM8K (train split; solution-checking items). We also used ANLI and BANKING77 train splits; they're in the 0.1 suite but outside its index, and both are in the 0.2 panel. The audit below reports what was checked and any shared passages.
- **Partitions.** Public-source data included training and development partitions.
- **Final-mixture audit.** Rechecked every question row (including teacher-written rows) against the complete 0.1 suite (132,422 requests) and JevBench's 231 public items. The checks looked for exact matches of normalised strings of at least 30 characters in any field and shared 13-word passages in each row's question text (instructions, state.question, state.code). Strings or passages seen in 20 or more suite requests were treated as prompt templates and ignored. No public JevBench item matched under these checks; a separate position check found no shared chess positions.
- **Suite overlap.** 16 BANKING77/VAST training rows share a 13-word passage with 31 suite requests: 23 of VAST's 3,006 and 8 of BANKING77's 3,080. Two VAST training posts are near-duplicates of a test post, but these rows had no exact normalised-text match under the audit. Dropping those requests leaves the index at 56.53 (VAST 0.7805 → 0.7803); BANKING77 is outside the 0.1 index.
- **Audit limits.** The 13-word passage check didn't search long-document bodies or option text. Semantic or pretraining overlap can't be ruled out, and private JevBench items weren't available to check.
- **Generated reasoning.** Our programs computed the labels for CRUXEval-style code and CLadder-style causal questions; no items from those benchmarks were used. We didn't reuse the suite's GSM8K distractors.
- **Teacher documents.** We kept Qwen3.8-27B's documents only if a fresh blind solve by that same teacher agreed with the answer. That's an agreement filter, not independent verification.
- The repo ships no benchmark items, GPQA text, JevBench items or teacher traces.
- **No MMLU-Pro or GPQA.** No rows from either were used in this fine-tune; SuperGPQA is a separate source.

### Limits

- English-centric. Training included Arabic iSarcasmEval rows; on the 0.1 suite, Arabic task A scored 0.321 and task C pairs 0.79. Broader multilingual performance hasn't been established.
- Doesn't chat or explain answers.
- Text in the state can sway the answer.
- Its weakest public hard cases are date/number reasoning, trade-offs and long policies.

</details>

## Licence

The MiMo model card declares MIT without a separate upstream licence file or copyright line (`LICENSE-MiMo.md`); its Qwen/Qwen3.5-9B base is Apache-2.0 (`LICENSE-Qwen`). The blink weights are for **non-commercial research and evaluation
only** (`LICENSE.md`); commercial use isn't licensed. Training used non-commercial, share-alike and unlicensed
sources (see the table above). It's unsettled whether their terms reach the weights, so check upstream terms too.
`blink.py`, `serve.py` and the Dockerfile are Apache-2.0.

<details><summary>Credits</summary>

Xiaomi MiMo (MiMo base) and the Qwen team (Qwen base). SemIf (MIT) for the evidence/criterion/options prompt layout. The Decision Index kit (MIT) and JevBench (MIT) for evaluation.

</details>
