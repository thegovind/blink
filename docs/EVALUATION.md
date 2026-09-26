# Evaluation methodology

## Decision Index

Decision Index 0.1 numbers in this repository are local runs of the official
kit on the archived edition, not leaderboard submissions. We also report local,
descriptive 0.2 runs of the official kit, not accepted leaderboard results.
Known training exposure remains in the 0.2 scores without the leaderboard's
penalty, so they cannot be ranked against leaderboard results. Details are on
each model card.

## JevBench

JevBench numbers are public-item development proxies from local runs with
JevBench code. Official scores require the maintainers' evaluation. A request
for blink-4b is open at `fstandhartinger/jevbench#81`.

## Held-out pack

The held-out pack was frozen before reads. Held-out numbers were never used for
training or model selection. The public repository includes preregistration and
manifest metadata only, not the held-out items or outputs.

## Reproduction tools

- `lab/run_requests.py`: run a request file through a blink runtime.
- `lab/build_results.py`: build public result summaries.
- `lab/prefix_parity.py` and `lab/precision_check.py`: compare runtime paths.
- `lab/engine_parity.py`: compare server/runtime outputs.
