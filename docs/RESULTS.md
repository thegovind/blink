# Results

All numbers below inherit the caveats in the model cards.

| Model | Decision Index 0.1 | JevBench public hard proxy | Held-out accuracy | Held-out ECE |
|---|---:|---:|---:|---:|
| blink-27b | 63.44 | - | 78.3% | 0.096 |
| blink-mimo-9b | 56.53 | 77/111 | 73.5% | 0.104 |
| blink-4b | 52.12 | 80/111 | 68.5% | 0.119 |

Decision Index 0.1 numbers are local runs of the official kit on the archived
edition, not leaderboard submissions. The live board moved to 0.2 on
2026-09-24, and there is no blink 0.2 result.

JevBench numbers are public-item development proxies, not official scores.
blink-4b public hard was 80/111 with hard ECE 0.067. blink-mimo-9b public hard
was 77/111 with hard ECE 0.136.

Held-out numbers were read after the pack was frozen and were not used for
training or model selection.
