# blink held-out preregistration

**Freeze time:** `2026-09-25T01:50:02+00:00`.

From this time onward, `heldout/` is frozen. Do not add, remove, relabel, regenerate, reorder, or hand-edit any request or gold file. If a mistake requires a change, leave this directory intact and create a new frozen directory, for example `heldout-v2/`, with a new manifest and preregistration note.

## Development sets

Model and runtime selection may use only these development sources:

- DI-S: the 3,000-request Decision Index 0.1 sample.
- JevBench public items.
- TypeSafe public workflow cases.
- blink synthetic development splits.

## Held-out sets

This pack is held out. It must never be used for training, prompt tuning, data filtering, threshold fitting, calibration fitting, checkpoint selection, architecture selection, or any other model-selection decision before a final candidate is fixed. Read it once per final candidate and score all tasks.

Optional additional lockbox: a fixed, stratified random 4,000-request slice from Decision Index 0.1 suite minus DI-S. The local `runs/suite-subsets.meta.json` contains subset metadata only:

```json
{
  "suite-minus-s3000": {
    "requests": 129422,
    "ids_sha256": "b738768b0cd2076ead8c637cddd2d1b233094f46524153da16444ff7a98f6aab"
  },
  "suite-s3000": {
    "requests": 3000,
    "ids_sha256": "2d6817eb780cd6b4de8c2b3f942c0ee1fb6195366c8c04c69322544716d751b8"
  }
}
```

The actual request-id list is not present on this local validation environment. To draw it on the evaluation environment with the Decision Index kit: load the 0.1 suite request table, remove every request id in DI-S, group the remaining ids by benchmark/track and gold label where a label exists, then draw 4,000 ids with seed `20260925` using proportional stratified sampling, largest-remainder allocation, and deterministic sorted-id tie breaks. Save the sorted id list and its SHA-256 beside the kit outputs before any model is run.

## Ship rule for a new model

A new model may ship only if all of the following hold:

1. It beats the current release on development sets.
2. It is not worse on this held-out pack: the paired 95% CI lower bound of the equal-task mean accuracy difference is at least `-0.010` (−1.0 point).
3. Its held-out equal-task mean accuracy point estimate is non-negative versus the current release.
4. It does not regress held-out calibration by more than `0.02` ECE.

Use `heldout/score.py --compare-dir <current-release-outputs>` for the paired held-out comparison.

## Runtime-only changes

Runtime-only changes that should not change answers, such as a faster prefill path, must satisfy both checks on this held-out pack:

- argmax agreement at least `99.5%` of questions;
- max absolute probability delta, `max |Δp|`, at most `0.02`.

These checks are separate from model-quality comparisons and should be computed question-by-question from the two output files.

## Leakage controls

The hard exclusions are the blink training source families, all Decision Index 0.1 and 0.2 datasets, JevBench, and TypeSafe workflow cases. Candidate datasets were checked against those lists before inclusion. The local exact-string leakage check hashed each held-out state and searched the local training/dev JSONL files available on this machine (`data/workflow/train.jsonl`, `data/workflow/dev.jsonl`); it found no hits. The large training-mixture JSONL files and remote training environments were not available here, so broader leakage control is by dataset family.
