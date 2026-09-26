# T5 experiment: pre-registration

Written 2026-09-25 02:00Z, before any T5 training, before any held-out read, and before the held-out pack's
baselines exist. Not edited after the first T5 result; any change becomes a dated addendum below the line.

## Two changes, two gates

### A. Shared-prefix prefill (runtime only, no weight change)
- **Change:** `space/blink.py` encodes a request's shared prefix (system prompt plus evidence) once, then runs each
  question's tail against copies of the cached prefix. Same tokens, same positions.
- **Gate** on blink-4b, blink-mimo-9b and blink-27b. Inputs:
  - the TypeSafe public workflow requests (45);
  - the Decision Index 0.1 u1000 latency sample (its multi-question requests);
  - the frozen held-out pack.
- **Pass requires:**
  - argmax agreement ≥ 99.5% of compared questions;
  - max |Δp| ≤ 0.02;
  - every argmax flip has a plain-path top-2 margin < 0.02.
- **Speed:** reported as measured (model time off vs on). No speed threshold; it's a cost claim, not a gate.
- **If it passes:** ship to the Space and `serve.py` as runtime v1.1. Weights are unchanged, and the v1.0 tags stay put.

### B. T5-4B candidate (weight change)
**Candidate:**
- **Starting point:** a new LoRA (r16, α32, same targets) on top of the released blink-4b merged weights, `ckpt/soup-t3-t4s300-t4f`.
- **Data:** `data/workflow/train.jsonl` (new code-labelled multi-question workflow documents) plus a replay of the T4-4B mix (`t4_4b_train.jsonl`) at equal row count.
- **Optimisation:** lr 2e-5, 1 epoch, budget 8,192, warmup 20.
- **Result:** merged into the base.
- **One run. No hyper-parameter search. If it fails, it's reported as failed.**

**Dev** (selection and diagnosis allowed, all measured against the released blink-4b in the same harness):
- JevBench public items (231);
- Decision Index DI-S (3,000);
- `data/workflow/dev.jsonl`;
- TypeSafe public cases (354 questions, `score_chart.py`).

**Held-out:** the frozen pack in `heldout/` (see `heldout/PREREG.md` and `MANIFEST.json`). It's read **once**, after the dev read. There's no second held-out read for this candidate and no training on anything derived from it.

**Ship rule.** All must hold:
1. JevBench public hard ≥ 78/111 (released: 80/111; two items of slack for noise) and hard ECE ≤ 0.087 (released 0.067 + 0.02).
2. DI-S index ≥ the released blink-4b's DI-S − 1.0 (same kit, same settings).
3. Held-out: the lower bound of the paired 95% bootstrap CI of the equal-task mean accuracy difference (T5 − released) ≥ −1.0 point, and the point estimate ≥ 0.
4. Held-out ECE (top-label, 10 bins, equal-task mean) not worse than released + 0.02.

**Claims:** "better" may be claimed only if held-out (3) has a CI lower bound > 0. Otherwise the most we can say is
"no worse on held-out, better on workflow dev". A pass on (1)–(4) alone ships as blink-4b v1.1 with this file linked
from the card. A failure on any rule means no release, and the result is logged.

**Not allowed:** looking at held-out items or per-item held-out outputs before the dev read is complete; changing
thresholds after seeing any T5 number; training on TypeSafe, JevBench, Decision Index or held-out content.

---
Addenda (dated, append-only):
- 2026-09-25 02:05Z (before any GPU run or held-out read). Gate A's held-out component uses the seven short tasks in full, plus the first 25 requests (250 questions) of `ecthr_a_long`.
  - The plain path re-encodes each 18k-token case file once per question, about 46M tokens per model on that task alone. Gate A needs both paths; the baselines need only the gated path.
  - Held-out baselines and the T5 read use the full pack with the runtime that passed Gate A.
  - Requests over blink's 131,072-token limit are refused, identically for every model, and scored as wrong.
- 2026-09-25T02:26Z (before any T5 run). `data/workflow` was regenerated with 60% long documents (2,446–3,136 state tokens: multi-section policies with superseded versions and effective dates, longer histories, distractors) and 40% short (617–731), same seeds and splits, 20 questions per document. Sha256: train `634294036ba77951…`, dev `5be8dc2c48d9f4df…` (full hashes in data/workflow/SHA256SUMS).
- 2026-09-25T02:34Z: **Gate A failed as written.**
  - Results: blink-4b and blink-mimo-9b, all three sets. Argmax agreement 99.60–100% passes. Max |Δp| is 0.032–0.049 and four flips have plain-path margins of 0.034–0.050, so both of those rules fail.
  - The 0.02 limit was set below bf16's own noise. On the only precision result seen so far (blink-mimo-9b, TypeSafe, 169 questions that fit in FP32), the plain bf16 path itself differs from an FP32 reference by up to 0.036; the shared-prefix path differs by up to 0.023.
  - Gate A is therefore not passed, and it is not re-scored.
  - **Replacement gate A′**, registered now, before any further precision results are read:
    - The shared-prefix path must be no less accurate than the plain path, both in bf16, measured against the plain path in FP32 (`lab/precision_check.py`).
    - (i) Argmax agreement with FP32 ≥ the plain path's − 0.5 point.
    - (ii) p99 |Δp| vs FP32 ≤ the plain path's + 0.005.
    - (iii) Max |Δp| vs FP32 ≤ the plain path's + 0.01.
    - Evaluated per model on data not used for this decision: `data/workflow/dev.jsonl` as requests (100 documents × 20 questions), plus the Decision Index u1000 and held-out sets from the precision runs already in progress (not yet read).
    - The MiMo TypeSafe precision result is excluded, because it motivated this revision.
    - Requests too large for the FP32 reference are skipped, and the skips are reported.
    - Ship only if A′ passes for every model evaluated. Report that A′ replaced a failed gate.
- 2026-09-25T03:24Z: **T5 dev read: fails ship rule 1, so no release.**
  - Rule 1 fails:
    - JevBench public hard: 69/111 (0.622) vs the released 80/111 (0.721); the rule requires ≥ 78.
    - Hard ECE: 0.171 vs 0.067; the rule requires ≤ 0.087.
    - Also: standard 0.903 vs 0.986; easy 0.979 vs 1.0.
    - Weaker families: long_policy 0.421 vs 0.579, temporal_numeric 0.333 vs 0.400, ambiguous 0.571 vs 0.857.
  - Rule 2 passes: DI-S 50.16 vs the released 50.08 (same kit settings; environment diff shows only the model path).
  - Workflow dev (in-distribution): 0.7595 → 0.9375.
  - **The held-out pack is not read for T5.** The dev failure already decides the outcome, and not reading it keeps the pack unspent for later candidates.
  - Diagnosis (dev only): code-labelled one-hot workflow rows drove the training loss to about 1e-4, so the model became overconfident. Continuing from the averaged release also undid its calibration.
- 2026-09-25T03:24Z: **New single candidate, T5b**, registered before it is computed.
  - Definition: a uniform weight average of the released blink-4b (`ckpt/soup-t3-t4s300-t4f`) and T5 (`ckpt/t5-4b`), 0.5 / 0.5, made with `lab/jevlab/soup.py`.
  - No other mixing weights will be tried.
  - Same dev reads (JevBench public serial, DI-S with the same settings, TypeSafe proxy) and the same ship rules 1–4. The held-out pack is read once, only if rules 1–2 pass.
- 2026-09-25T03:30Z: **Gate A′ fails, so the shared-prefix prefill is not shipped by default.**
  - blink-4b on the held-out case-file subset (240 questions that fit in FP32): the prefix path's p99 |Δp| vs FP32 is 0.0230, against a limit of plain + 0.005 = 0.0220; its max is 0.0305, against plain + 0.01 = 0.0295. Argmax agreement with FP32 is 0.9958 vs 1.0000, which passes.
  - Passing sets:
    - blink-4b: Decision Index u1000 (4,324 questions) and workflow dev (2,000).
    - blink-mimo-9b: Decision Index u1000 (4,292) and held-out (220).
  - blink-mimo-9b's workflow-dev result is still pending and cannot change the verdict.
  - Decision:
    - `space/blink.py` keeps the path as opt-in (`BLINK_PREFIX_CACHE=1`); the default is off, and the Space pins it off.
    - Measured speed (4.8–6.9× less model time on long multi-question documents) is reported as an opt-in trade-off, not a shipped default.
    - The TypeSafe dev read for T5 (same trainer, plain path): released blink-4b 83.0% vs T5 84.7% (typed proxy); T5 still fails rule 1.
- 2026-09-25T03:45Z: **T5b dev read: fails ship rule 1, so no release; the held-out pack is not read for T5b.**
  - JevBench public hard: 73/111 (0.658) vs the required ≥ 78; hard ECE 0.116 vs ≤ 0.087; standard 0.917.
  - DI-S: 49.77, which passes rule 2 (≥ 49.08).
  - TypeSafe dev proxy: 85.5% vs the released 83.0%.
  - This series is closed. No further candidates are built from this data, and nothing is tuned against the dev reads.
- 2026-09-25T03:45Z: **Held-out baselines for the three releases** (anchors for future candidates; scoring them selects nothing):
  - The full pack was read with the plain path; 2,000/2,000 items were answered and 0 refused.
  - Equal-task mean accuracy (95% CI) and ECE:

    | Model | Accuracy | 95% CI | ECE |
    |---|---|---|---|
    | blink-27b | 78.3% | 76.6–80.0 | 0.096 |
    | blink-mimo-9b | 73.5% | 71.7–75.2 | 0.104 |
    | blink-4b | 68.5% | 66.7–70.4 | 0.119 |

  - Files: heldout/outputs/<model>/score.json.
