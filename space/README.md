---
title: blink
emoji: "\u26A1"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: One-pass typed decisions with option probabilities
tags:
  - decision-model
  - classification
  - calibration
  - system-1
models:
  - thegovind/blink-4b
  - thegovind/blink-mimo-9b
  - Qwen/Qwen3.5-4B
  - thegovind/blink-27b
---

# blink

Send a state and typed questions. Get a probability for every offered option; no text is generated.

**Live models:** [blink-4b](https://huggingface.co/thegovind/blink-4b) (Qwen3.5-4B text) and [blink-mimo-9b](https://huggingface.co/thegovind/blink-mimo-9b) (MiMo-V2.6-Distill-Qwen-9B). [blink-27b](https://huggingface.co/thegovind/blink-27b) is in Results.

**Ask tab:** [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) drafts questions and options. blink scores options without generating text.

**API:** See the API tab to use TypeSafe clients with a local blink server or call this Space via Gradio.

**Code:** [GitHub](https://github.com/thegovind/blink) · **Docs:** [thegovind.github.io/blink](https://thegovind.github.io/blink/)

## Results

| Decision Index 0.1 (archived) | Index |
|---|---:|
| blink-27b | 63.44 |
| Jev 1.13.0 | 59.51 |
| blink-mimo-9b | 56.53 |
| blink-4b | 52.12 |

These are local runs with the official kit's scorer, not leaderboard submissions. The full suite picked the 27B finalist, so its full-suite score is post-selection. The 4B was fixed before its full run. blink-27b trained on the public train splits of ContractNLI, iSarcasmEval and VAST (all of Language), plus Amazon ESCI and Humicroedit. Set Language to Jev's score and its index would be 60.44 vs Jev's 59.51. That's arithmetic, not an ablation.

| JevBench public-item development proxy | Score |
|---|---:|
| blink-4b | 76.5 |
| JevK5 v0.2.0 | 76.1 |

JevBench figures are public-item development proxies, not official scores or predictions. Held-out, judge and sealed items need the maintainer's run; no official score, rank or parity is claimed.

| Held-out tasks (equal-task mean accuracy) | Accuracy | 95% CI |
|---|---:|---:|
| blink-27b | 78.3% | 76.6–80.0 |
| blink-mimo-9b | 73.5% | 71.7–75.2 |
| blink-4b | 68.5% | 66.7–70.4 |

Eight public tasks, 250 items each, never used for training or model selection.

<details><summary>How to read these results</summary>

- **Decision Index 0.1:** Local runs on the archived suite (132,422 requests, 37 benchmarks; 19 panel benchmarks averaged). Comparison rows use the September 22, 2026 snapshot. No 0.2 score claimed.
- **Selection:** The 3,000-request DI-S sample was reused to pick the prompt and checkpoints.

</details>

## Question types

| Type | Input | Answer |
|---|---|---|
| `choice` | Up to 255 options | A probability per option and the top choice |
| `noul` | Yes/no question | Probability of yes |
| `score` | 2–10 ordered levels | Probabilities and expected level |

## How it works

Supervised fine-tuning with LoRA r16 on attention, DeltaNet, and MLP projections. Embeddings, norms, and heads stay frozen.

- **blink-4b** and **blink-27b**: text-only weights.
- **blink-mimo-9b**: fine-tune of MiMo-V2.6-Distill-Qwen-9B (based on Qwen3.5-9B). Retains the vision tower; omits MTP tensors. Screen click uses that tower for screenshots. Everything else uses the text side.

The model reads option letters directly. Details: [blink-4b](https://huggingface.co/thegovind/blink-4b) and [blink-27b](https://huggingface.co/thegovind/blink-27b) model cards.

<details><summary>Probabilities and limits</summary>

`blink.py` uses the evaluation prompt and option-letter readout.

The probabilities are conditional on the offered options, not certified chances of being right. Temperature is 1.0, not fitted.

Inputs are capped at 131,072 tokens per question (longest evaluated prompt: 37,906). Longer inputs are refused, not truncated. Models generate no text, explanations, or tool calls. Text in a state can still sway an answer.

</details>

---

Weights are for non-commercial research and evaluation; the app code is Apache-2.0.
