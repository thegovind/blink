# blink — verified fact pack for the first-principles guide

Every project-specific number the guide uses must come from this file or from a source it cites.
General ML background may be written from knowledge, but must be correct.

## Hard rules for the guide
- The method is **plain supervised fine-tuning (SFT)**, which the cards call "calibration-oriented decision
  post-training". Explain RLCD only as Jev's own claimed recipe: it isn't public, and we didn't use or
  reproduce it. No RL, no preference optimisation (DPO etc.), no quantisation.
- Don't mention employers, internal infrastructure, machine or host names, file-system paths from the
  training machines, or the specific GPU models used for training. Generic terms like "GPU memory" are fine,
  and so are the ZeroGPU facts below (public HF docs).
- The Decision Index numbers are **our local runs of the official kit** on the archived 0.1 edition, not
  leaderboard submissions. The live board moved to 0.2 on 2026-09-24, and we have **no 0.2 score**.
- JevBench numbers are **public-item development proxies from our own runs with JevBench's code**, not
  official scores. blink-4b is submitted (fstandhartinger/jevbench#81) and waits in their measurement queue.

## 1. What blink is
- A typed-decision model. Input: `state`, the text or JSON being judged, and `questions`. Output: a
  probability for every offered option. One prefill forward pass per batch of questions. No generated text.
- Question types (`space/blink.py`, `question_options`):
  - `choice`: `criteria` is either `{key: description}` or a list of keys; 1–255 options.
  - `noul`: yes/no, with optional `criteria {"true": …, "false": …}`. The answer includes `noul` = p(yes).
  - `score`: `criteria` is a list of 2–10 ordered level labels. `score` is the expected 0-based level;
    `choice` is the most likely level.
- `confidence` for choice = (p_max − 1/K)/(1 − 1/K). This measures concentration, not correctness.
- Limits: 255 options per choice, 512 questions per request, 131,072 input tokens per question; over-limit
  requests are refused, never truncated. Questions are packed into forwards of up to 32,768 padded tokens
  (`TOKEN_BUDGET`).
- Prompt ("semif" rendering, from the open SemIf project; fixed for all training and evaluation):
  - system: "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option.
    Respond with only its uppercase letter, with no explanation or reasoning."
  - user: JSON `{"evidence": state, "criterion": instructions, "options": [{"letter", "description"}]}`;
  - chat template with thinking turned off.
- Labels: A–Z, then two-letter AA…ZZ. Each is verified at load to be a single token in the vocabulary.
- **Readout:** take the hidden state at the last prompt position and multiply it by only the `lm_head` rows
  of the offered letters, in FP32. Softmax over those letters (temperature 1.0, not fitted). The rest of the
  vocabulary is ignored. This is a "label-token" or "verbalizer" readout: no new classifier head is added.

## 2. The three models (configs and safetensors headers at tag v1.0; see docs/facts/*.json)
| | blink-4b | blink-27b | blink-mimo-9b |
|---|---|---|---|
| Base | Qwen/Qwen3.5-4B (Apache-2.0) | Qwen/Qwen3.8-27B (Apache-2.0) | XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B (MIT declared; a Qwen3.5-9B fine-tune) |
| Role | JevBench entry | Decision Index model | multimodal-base variant |
| Layers | 32 (24 Gated DeltaNet + 8 full attention) | 64 (48 + 16) | 32 (24 + 8) |
| Full-attention layers | 3, 7, …, 31 (every 4th; `full_attention_interval` 4) | 3, 7, …, 63 | 3, 7, …, 31 |
| hidden_size | 2560 | 5120 | 4096 |
| attention heads (q) / KV heads / head_dim | 16 / 4 / 256 | 24 / 4 / 256 | 16 / 4 / 256 |
| DeltaNet key heads / value heads / head dims | 16 / 32 / 128, 128 | 16 / 48 / 128, 128 | 16 / 32 / 128, 128 |
| MLP intermediate_size | 9216 | 17408 | 12288 |
| vocab_size | 248,320 | 248,320 | 248,320 |
| tie_word_embeddings | **True** (lm_head = embedding matrix) | False | False ("untied embeddings") |
| Shipped parameters | 4,205,751,296 (text only) | 26,895,998,464 (text only) | 9,409,813,744 total = 8,953,803,264 text + 456,010,480 vision |
| Tensors in checkpoint | 426 | 851 | 760 (333 of them vision) |
| bf16 size on disk | 8.41 GB in 2 shards (4.97 + 3.44) | 53.79 GB in 12 shards | 18.82 GB in 4 shards |
| LoRA trainable parameters | 32,464,896 (≈32.5M) | 116,727,808 (≈116.7M) | 43,278,336 (≈43.3M) |

Other config facts, identical for all three:
- `hidden_act` silu; `rms_norm_eps` 1e-6; `max_position_embeddings` 262,144;
- RoPE theta 1e7 with `partial_rotary_factor` 0.25 (RoPE on 64 of 256 head dims);
- multimodal RoPE sections [11, 11, 10];
- `attn_output_gate` present (full-attention output gate);
- linear-attention conv kernel 4.

### Exact tensor shapes, blink-4b (header of the safetensors files; `LM.` = `model.language_model.`)
Shapes are [out, in]. A Linear layer computes y = W x with W of shape [out, in].
- `LM.embed_tokens.weight` [248320, 2560]. With tied embeddings there is no separate `lm_head.weight`; the
  readout reuses these rows. That's 635,699,200 parameters, about 15% of the model.
- Gated DeltaNet layer (layer 0):
  - `linear_attn.in_proj_qkv` [8192, 2560]. That's q 16×128 = 2048, k 2048 and v 32×128 = 4096.
  - `in_proj_z` [4096, 2560]: the output gate, 32×128.
  - `in_proj_a` [32, 2560]: per value head, feeds the decay.
  - `in_proj_b` [32, 2560]: per value head, the write strength β.
  - `A_log` [32] and `dt_bias` [32]: learned decay parameters.
  - `conv1d` [8192, 1, 4]: a depthwise causal conv over the q, k and v channels, kernel 4.
  - `norm` [128]: a gated RMSNorm per head dim.
  - `out_proj` [2560, 4096].
- Full-attention layer (layer 3):
  - `self_attn.q_proj` [8192, 2560]. That's 16 heads × 256 × 2, because the query projection also produces
    the output gate.
  - `k_proj` [1024, 2560] and `v_proj` [1024, 2560]: 4 KV heads × 256 (grouped-query attention, 4 query
    heads share each KV head).
  - `q_norm` [256] and `k_norm` [256]: per-head RMSNorm.
  - `o_proj` [2560, 4096].
- MLP, every layer: `gate_proj` [9216, 2560], `up_proj` [9216, 2560], `down_proj` [2560, 9216]. This is
  SwiGLU: down(silu(gate(x)) ⊙ up(x)).
- Norms: `input_layernorm` [2560], `post_attention_layernorm` [2560], final `LM.norm` [2560].
- Tensor names read as: `model.language_model.layers.{i}.{linear_attn|self_attn|mlp}.{name}.weight`.

### MiMo vision tower (blink-mimo-9b; unchanged, byte for byte)
- `model.visual.patch_embed.proj` [1152, 3, 2, 16, 16]. A Conv3d over 3 colour channels × 2 frames
  (temporal patch 2) × 16×16 pixels; each patch becomes a 1152-dim vector.
- `pos_embed` [2304, 1152]: learned position embeddings.
- 27 transformer blocks with hidden 1152 and 16 heads. Each block has `attn.qkv` [3456, 1152], `attn.proj`,
  `mlp.linear_fc1` [4304, 1152], `mlp.linear_fc2` [1152, 4304], and LayerNorms `norm1`/`norm2` with bias.
- `merger`:
  - `norm` [1152];
  - `linear_fc1` [4608, 4608], where 4608 = 1152 × 4: `spatial_merge_size` 2 merges 2×2 neighbouring patches;
  - `linear_fc2` [4096, 4608], which projects to the language model's hidden size of 4096.
- The merged image tokens enter the language model's sequence between vision start/end tokens.
- `blink.py`, `serve.py` and the Space use the **text side only**. The decision training was text-only.
  Screenshot results come from a separate probe harness.

## 3. LoRA
- Config: rank r = 16, alpha α = 32, so the scale is α/r = 2; dropout 0. Applied through PEFT to every
  language-model layer:
  - full attention: q_proj, k_proj, v_proj, o_proj;
  - Gated DeltaNet: in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj;
  - MLP: gate_proj, up_proj, down_proj.
- Frozen: embeddings, all norms, lm_head, conv1d, A_log and dt_bias (`lab/jevlab/train.py` TARGETS;
  `graft_vlm.py` doc).
- Per adapted matrix W [out, in]: A [r, in] and B [out, r], so r·(in+out) parameters. Forward:
  y = W x + (α/r)·B(A x). Standard init: B = 0, so training starts exactly at the base model.
- Worked count for blink-4b, which reproduces the card's 32.5M exactly:

  | Layer group | Per-matrix terms | Per layer | Layers | Total |
  |---|---|---:|---:|---:|
  | DeltaNet | qkv 16·(2560+8192) = 172,032; z 16·(2560+4096) = 106,496; a 16·(2560+32) = 41,472; b 41,472; out 16·(4096+2560) = 106,496 | 467,968 | 24 | 11,231,232 |
  | Attention | q 172,032; k 16·(2560+1024) = 57,344; v 57,344; o 106,496 | 393,216 | 8 | 3,145,728 |
  | MLP | 3 × 16·(2560+9216) = 3 × 188,416 | 565,248 | 32 | 18,087,936 |
  | **Sum** | | | | **32,464,896** |

  The same method gives 116,727,808 for the 27B and 43,278,336 for MiMo.
- Merge: `PeftModel.merge_and_unload()` folds W ← W + (α/r)·B·A into the base weights and saves a
  standalone checkpoint (`lab/jevlab/merge.py`, `max_shard_size="5GB"`).

## 4. Training (lab/jevlab/train.py; run scripts lab/t3_run.sh, t4_4b_run.sh)
- Row format: `{state, question, gold key | None, target {key: p} | None, weight, src}`.
- Loss per row:
  1. Take the log-softmax over the offered label rows at the last position, in FP32.
  2. Loss = −Σ_k t_k · log p_k (`forward_batch`), where t is the soft target when given, else one-hot gold.
  3. Weight by the row weight.
  This equals KL(t ‖ p) + H(t), and H(t) is a constant.
- Option order and letter assignment are re-drawn every epoch. Score levels keep their order; yes/no order
  is randomised.
- Optimiser: AdamW, betas (0.9, 0.99), weight decay 0.
- Learning-rate schedule: linear warmup, then cosine decay to 0:
  `lr = base · min(1, (step+1)/warmup) · 0.5 · (1 + cos(π · step/total))`.
- Gradient clipping: norm 1.0.
- Precision: bf16 autocast for the body; FP32 for the readout projection and softmax, the same as at serving.
- Data parallel: one full model replica per GPU. LoRA gradients are all-reduced (averaged) by hand each step,
  and the model isn't DDP-wrapped. Batches are packed by a padded-token budget per GPU per micro-batch.
- Base-model anchors (4B only; `lab/jevlab/anchor.py`):
  - The frozen base model scores authored prompts whose teacher answers failed verification.
  - Its restricted-label distribution becomes the soft target, so CE against it = KL(p_base ‖ p_student) +
    const.
  - This keeps the student close to the base on those inputs.
- Teacher: Qwen/Qwen3.8-27B wrote the teacher documents and their typed questions. Its probability targets
  were kept only when a blind re-solve by the same teacher agreed.
- Runs:

  | Model | Stage | Rows | Learning rate | Steps / details |
  |---|---|---:|---|---|
  | blink-4b | T3 | 23,156 | 3e-5 | 96 steps; budget 16,384 tokens; warmup 10; 1 epoch |
  | blink-4b | T4 | 42,360 | 4e-5 | 472 steps; budget 8,192; warmup 20; 1 epoch |
  | blink-27b | T2 | 72,700 | 5e-5 | 230 steps |
  | blink-27b | T4 (continues from T2) | 74,754 | 3e-5 | 856 more steps |
  | blink-mimo-9b | one run | 123,195 | 5e-5 | 615 steps; 1 epoch |

  - Both 4B stages trained from the base.
  - The 4B release is a **soup**: a uniform average of T3 + T4 step 300 + T4 final.
  - The MiMo run had a pre-registered gate of DI-S ≥ 55.
- Data mix (exact; `runs/mix-categories.json`):

  | Category | 4B T3 | 4B T4 | 27B T2 | 27B T4 | MiMo |
  |---|---:|---:|---:|---:|---:|
  | decision worlds (program-generated) | 11,352 | 12,000 | 12,000 | 12,000 | 12,000 |
  | teacher-written question rows | 3,741 | 7,860 | — | 7,860 | 7,860 |
  | exact-probability worlds | 2,579 | 6,220 | 3,299 | 1,748 | 5,047 |
  | public-source rows (public train splits) | 2,221 | 2,780 | 41,401 | 23,252 | 61,394 |
  | base-model anchors | 1,763 | 3,500 | — | — | — |
  | program-generated reasoning | 1,500 | 3,000 | 16,000 | 23,894 | 23,894 |
  | judge-style | — | 7,000 | — | — | 7,000 |
  | chess moves | — | — | — | 6,000 | 6,000 |
  | **total** | **23,156** | **42,360** | **72,700** | **74,754** | **123,195** |

  - Program-generated reasoning covers code execution, causal questions, word problems and logic grids.
  - Judge-style rows: 3,000 GSM8K-train solution checks, 2,500 Dolly-15k routing and 1,500 program-answer
    checks.
  - Chess: searchless_chess training positions.
  - The per-model public datasets are in each card's "Data sources and licences" table.
- Tried and not kept (4B card):
  - distilling 27B answers into the 4B (no help on hard items);
  - a DI-focused 4B (+0.4 DI-S only);
  - mixing JevK5 weights into the soup;
  - Qwen3.5-9B with the T1 recipe (54.5 DI-S, below the bar).
  - Earlier, T1 (106.7k rows, lr 1e-4, 355 steps) lifted DI-S but hurt hard items.

## 5. Merging and grafting
- **Soup** (`lab/jevlab/soup.py`): load each merged checkpoint in FP32, sum the state dicts, divide by N, and
  save in bf16 with `max_shard_size="5GB"`. All ingredients were fine-tuned from the same base, which is why
  averaging works (they sit in one low-loss region).
- **Graft** (`lab/jevlab/graft_vlm.py`), for MiMo:
  - The text-only merged checkpoint is written back into the full vision-language parent. The parent's
    vision tower, config and processor files are kept byte for byte.
  - LoRA-targeted language-model tensors come from the merged text checkpoint.
  - Every tensor LoRA never touched (embeddings, norms, convs, decay parameters, lm_head) must equal the
    parent's and is written from the parent's own bytes.
  - Every targeted tensor must differ from the parent's.
  - Key sets and shapes must be identical, and the result must reload as a vision-language model with no
    missing or unexpected keys.

## 6. Files, formats, serving
- **safetensors:** a small JSON header (tensor name → dtype, shape, byte offsets) followed by raw tensor
  bytes. It can be memory-mapped and has no pickle, so no code runs on load.
- **Sharding:** split at about 5 GB (`max_shard_size`). `model.safetensors.index.json` maps each tensor name
  to its shard. Reasons: single-file size limits, parallel and resumable downloads, and loading shard by shard.
- **Repo files:** config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja,
  generation_config.json, weights.sha256, blink.py, serve.py, Dockerfile, eval/*.json, LICENSE*.
  - weights.sha256 is checked by serve.py before serving; a mismatch stops startup.
  - Tag `v1.0` pins the release; later commits on main changed only cards and licences, not weights.
- **serve.py:**
  - `POST /v1/systemone` takes `{state, questions}` and returns `{answers, usage}`, the Jev-compatible wire
    format. JevBench's stock `typesafe` adapter and the Decision Index kit's `http` engine use it.
  - `GET /healthz` reports weights_verified, warmup.repeat_identical, kernels, versions and hub_offline.
  - Requests run one at a time; the questions in a request are batched.
  - A fix worth teaching: Nagle's algorithm plus delayed ACK added about 42 ms per request until
    `TCP_NODELAY` was set.
- **Latency:** the kit's per-request timer, on 1,000 random suite requests served one at a time, gave median
  66.4 ms (4B), 192.2 ms (27B) and 68.3 ms (MiMo).
- **Kernels:** fast path via flash-linear-attention (Triton). Without it transformers falls back to
  reference PyTorch, which is correct but very slow; on CPU a single 2B draft didn't finish in 6 minutes.
- **Space** (huggingface.co/spaces/thegovind/blink):
  - Gradio 6.28 on ZeroGPU. ZeroGPU lends a shared GPU per decorated call (`@spaces.GPU`). Models are placed
    on `cuda` at import under CUDA emulation.
  - Daily quotas: 2 min anonymous, 5 min free account, 40 min PRO.
  - Timings: first call on a fresh container ~25–35 s (kernel compile); warm 55–110 ms model time.
  - The first render shows a saved run, because no GPU can be attached while the app starts.
  - Gradio's server-side rendering injected CSS unscoped, which broke the styles; fixed with
    `ssr_mode=False`.

## 7. Evaluation
- **Decision Index:**
  - The benchmark for typed decision engines; the HF Space is multimodalart/jev-decision-index.
  - The **kit** (`sources/repos/decision-index/README.md`) builds request files, runs an engine and scores.
  - Unanswered counts as wrong (full frozen denominator).
  - skill = clip((raw − random)/(1 − random)).
  - Index = `balanced_raw` = 100 · mean over the 5 areas of raw.
  - `breadth_skill` = 100 · (∏_area (0.1 + 0.9·skill)^(1/5) − 0.1)/0.9.
  - Areas: Knowledge & Reasoning, Language Understanding, Retrieval & Classification, Tools & Automation,
    Arts & Human Judgment.
  - 0.1 archived edition: 132,422 requests across 37 benchmarks; the headline index averages 19 panel
    benchmarks.
  - Panel: MMLU, GPQA Diamond, GSM8K, CRUXEval, CLadder, ChessBench, ContractNLI, iSarcasmEval, VAST, BRIGHT,
    Amazon ESCI, BFCL, ToolRet, RouterBench, BPoMP, Humicroedit, POP909-CL, cfcolor, Habermas Machine.
  - 0.2 (live since 2026-09-24): 121,057 requests, 44 benchmarks, different metric transformations.
- **DI-S:** a fixed 3,000-request sample of the 0.1 suite, used for checkpoint selection. It also gives
  "outside DI-S" (the suite minus the sample), a cleaner read.
- **Results (0.1, full suite, our runs):**

  | Model | Index |
  |---|---:|
  | blink-27b | 63.44 |
  | Jev 1.13.0 (leaderboard) | 59.51 |
  | blink-mimo-9b | 56.53 (outside DI-S 56.60; DI-S 55.3 vs the gate of 55) |
  | Jevfire (27B, best open entry in the 2026-09-22 snapshot) | 55.74 |
  | blink-4b | 52.12 |
  | Laya (421M) | 16.39 |

  - 27B climb: G0 zero-shot base DI-S 56.68 → T2 61.42 → T4 64.18 DI-S (+2.76, 95% CI [−0.07, 7.07],
    P(better) .97).
  - 4B: base DI-S 44.45; T1 53.15.
- **Training-overlap disclosure:**
  - Public train splits of ContractNLI, iSarcasmEval and VAST (the whole Language area) plus Amazon ESCI and
    Humicroedit were in some mixes.
  - For MiMo, with Language set to Jev's score, the index would be 54.96. That's arithmetic, not an ablation.
- **Dedup audit:**
  - Scope: exact ≥30-character strings in any field; 13-word passages in question stems; strings seen in 20+
    suite requests treated as templates and ignored.
  - 0 JevBench hits; blink-4b clean.
  - 27B and MiMo: 16 training rows overlap 31 suite requests (23 VAST, 8 BANKING77), coming from those
    datasets' own train/test splits. Re-scoring without them leaves the index unchanged (63.44, 56.53).
- **JevBench:**
  - v1.2 public items: 231 (48 easy, 72 standard, 111 hard). There are also held-out items and a 308-item
    sealed private set.
  - Official scores come only from the maintainers' own runs.
  - Four axes: Intelligence, Calibration, Speed, Cost. Jev 1.13.0 is 63.3 (#1) and JevK5 v0.2.0 62.0 (#2),
    official.
  - Our proxy, hard items:

    | Model | Hard | Hard ECE |
    |---|---:|---:|
    | blink-4b | 80/111 (0.721) | 0.067 |
    | JevK5 | 79/111 | 0.068 |
    | blink-mimo-9b | 77/111 (0.694) | 0.136 |

  - blink-4b standard: 71/72.
  - The public items were reused for candidate selection (never training), so these are development-set
    numbers.
  - Proxy speed = twice the raw latency + 0.15 s. Proxy cost uses JevBench's 4B tariff of $0.03 per million
    input tokens.
- **Calibration metrics:** ECE = the weighted gap between confidence and accuracy across confidence bins.
  TVD = total variation distance between distributions. Also NLL and Brier.
- **Probes** (ours, not official):
  - Next-click on 500 Multimodal-Mind2Web test steps with 5 offered elements (chance 20%):

    | Model | Page text | Screenshot |
    |---|---:|---:|
    | blink-mimo-9b | 53.8% | 48.4% |
    | MiMo base | 49.0% | 43.0% |
    | blink-4b | 55.4% | — |

  - RouterArena: the blink router scored 68.06 vs 76.17 for always picking one strong model. A negative
    result, not claimed publicly.

## 8. Contrastive learning and CLM (review/clm.md)
- CLM (github.com/Contrastive-LM/CLM) is as flexible as blink at request time. Its zero-shot quality on the
  live DI 0.2 is 6.51 (chance-corrected), against Jev's 51.67, with ECE 0.30 vs 0.065.
- CLM's best headline results use per-benchmark fine-tuned heads. Our decision: no change to blink.

## 9. Free-form authoring prototype (today; `space/author.py`; runs/author-probe-e2e.log)
- Goal: let a visitor type anything, e.g. "How many r in strawberry".
  1. An untouched instruct model, Qwen/Qwen3.5-4B (Apache-2.0, pinned revision 851bf6e), drafts
     `{state, questions}` as JSON: greedy decoding, thinking off, few-shot prompt.
  2. The draft is normalised and checked with blink's validator.
  3. blink decides in one pass.
  Drafting is the only step that generates text.
- The drafter refuses non-decisions ("write a poem", "translate…").
- Qwen3.5-2B drafted worse: it made strawberry a yes/no question and misordered a scale.
- ZeroGPU drafting: 42–91 generated tokens, 1.9–5.0 s. The whole ask takes about 2.4–12 s wall-clock.
- Examples (blink-4b / blink-mimo-9b):

  | Ask | blink-4b | blink-mimo-9b |
  |---|---|---|
  | How many r in strawberry | "3" 59% | 96% |
  | 17 × 23 | 391, 99% | 100% |
  | Is 91 prime? | no, 100% | 94% |
  | Capital of Australia | Canberra, 100% | 99% |
  | Is eval(input()) safe? | no, 99% | 98% |

- Wording matters. With state "strawberry" and a slightly different question, blink-4b split 2 vs 3
  (43.8% / 43.1%) and MiMo said 2 (79%). Letter counting is a tokenisation weak spot.

## 10. Sources to read for details
- `PLAN.md`: the full dated log of decisions and results. Skip anything about machines, paths or other
  projects.
- `release/out/{blink-4b,blink-27b,blink-mimo-9b}/README.md`: the reviewed model cards.
- `space/blink.py`: the runtime. `release/serve.py`: the server. `lab/jevlab/*.py`: training, synthesis,
  merge, soup, graft, anchor and dedup.
- `sources/prior-art-readout.md`: open prior art (JevK5, openjev, SemIf, kev, jevfire, decider, reflex).
- `sources/repos/decision-index/README.md`: kit and metric. `review/clm.md`: contrastive.
- `docs/facts/*.config.json`, `docs/facts/tensor-shapes.json`, `docs/facts/*.tensor-names.json`.
