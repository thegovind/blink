# Evaluation methodology

## Decision Index

Decision Index numbers in this repository are local runs of the official kit on
the archived 0.1 edition. They are not leaderboard submissions. No Decision
Index 0.2 score is reported for blink v1.0.

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
