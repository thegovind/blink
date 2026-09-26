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

**Live models:** [blink-4b](https://huggingface.co/thegovind/blink-4b) (Qwen3.5-4B text) and
[blink-mimo-9b](https://huggingface.co/thegovind/blink-mimo-9b) (MiMo-V2.6-Distill-Qwen-9B).
Switch between them in Playground, Ask or Use cases; every click runs the selected model.
[blink-27b](https://huggingface.co/thegovind/blink-27b) is in Results, not live in this app.

**Ask:** type a question in the Ask tab. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) drafts the question and its options, blink answers it in one pass, and Open in Playground lets you edit the draft and run it again. Drafting is the only step that writes text; blink's answer generates none.

## Results

| Decision Index 0.1 (archived) | Index |
|---|---:|
| blink-27b | 63.44 |
| Jev 1.13.0 | 59.51 |
| blink-mimo-9b | 56.53 |
| blink-4b | 52.12 |

| JevBench public-item development proxy | Score |
|---|---:|
| blink-4b | 76.5 |
| JevK5 v0.2.0 | 76.1 |

| Held-out tasks (equal-task mean accuracy) | Accuracy | 95% CI |
|---|---:|---:|
| blink-27b | 78.3% | 76.6–80.0 |
| blink-mimo-9b | 73.5% | 71.7–75.2 |
| blink-4b | 68.5% | 66.7–70.4 |

Eight public tasks, 250 items each, never used for training or for choosing a model.

<details><summary>How to read these results</summary>

Decision Index figures use the complete archived 0.1 suite: 132,422 requests across 37 benchmarks.
The headline index averages 19 panel benchmarks; comparison rows use the September 22, 2026
leaderboard snapshot. These are local runs with the official kit's scorer, not
leaderboard submissions. The live board is 0.2; the public kit can't build it yet. No 0.2 result is claimed.

We reused the 3,000-request DI-S sample to pick the prompt and checkpoints. The full suite picked the
27B finalist, so its full-suite score is post-selection. The 4B was fixed before its full run.

blink-27b trained on the public train splits of ContractNLI, iSarcasmEval and VAST (all of Language),
plus Amazon ESCI and Humicroedit. Set Language to Jev's score and its index would be 60.44 vs Jev's
59.51. That's arithmetic, not an ablation.

JevBench figures are public-item development proxies, not official scores or predictions. Held-out,
judge and sealed items need the maintainer's run; no official score, rank or parity is claimed.

The speed chart uses each system's own kit-timer measurement. Jev is a hosted API over the network;
blink uses our HTTP server, one request at a time over 1,000 random suite requests. The next-click
figure uses our own harness and sampling on 500 Multimodal-Mind2Web test steps with five options,
not the official evaluation. Five choices are easier than ranking a whole page; blink never trained
on web actions. With page text, blink-4b is about 10 points ahead of its base; a screenshot adds nothing
after training.

</details>

## Question types

| Type | Input | Answer |
|---|---|---|
| `choice` | Up to 255 options | A probability per option and the top choice |
| `noul` | Yes/no question | Probability of yes |
| `score` | 2–10 ordered levels | Probabilities and expected level |

## How it works

LoRA r16 updates each layer's attention, DeltaNet and MLP projections; embeddings, norms and the output
head stay frozen. blink-4b and blink-27b ship text-only weights. blink-mimo-9b keeps MiMo's
unchanged vision tower, but its checkpoint has no MTP tensors. The app uses its text side only.
The model reads option-letter scores; it doesn't generate text.

Jev's RLCD recipe isn't public, so we didn't copy it; this is supervised fine-tuning on decision data.
The [blink-4b](https://huggingface.co/thegovind/blink-4b) and
[blink-27b](https://huggingface.co/thegovind/blink-27b) model cards have the full story.
MiMo-V2.6-Distill-Qwen-9B is itself a fine-tune of Qwen3.5-9B.

<details><summary>Probabilities and limits</summary>

`blink.py` uses the same prompt and option-letter readout as the evaluation scorer.
`choice` returns the top option and a concentration score; `noul` returns the probability of yes;
`score` returns the expected level. The probabilities are conditional on the offered options, not
certified chances of being right. Temperature is 1.0, not fitted.

Inputs are capped at 131,072 tokens per question; the longest evaluated prompt was 37,906.
Longer inputs are refused, not truncated. The model can't write, explain an answer or call a tool.
Text in a state can still sway its answer.

</details>

---

Personal research release by thegovind. Not an official product of any company, and not affiliated with TypeSafe AI, Xiaomi, Alibaba Cloud or the Qwen team. Weights are for non-commercial research and evaluation; the app code is Apache-2.0.
