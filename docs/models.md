# Models

| Model | Base | Weights (bf16) | Live in the Space |
|---|---|---:|---|
| [thegovind/blink-4b](https://huggingface.co/thegovind/blink-4b) | Qwen/Qwen3.5-4B | 8.4 GB | yes |
| [thegovind/blink-mimo-9b](https://huggingface.co/thegovind/blink-mimo-9b) | XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B | 18.8 GB | yes |
| [thegovind/blink-27b](https://huggingface.co/thegovind/blink-27b) | Qwen/Qwen3.8-27B | 53.8 GB | no |

Use code revision `v1.4` for optional self-hosted screenshots; see the [API guide](api.md#screenshots-servepy-only-opt-in). Weights are unchanged since `v1.0`.

Code: Apache-2.0. Weights: non-commercial research and evaluation only; see each model card's license.

## Serve it

| Server | Models | Input | Best for |
| --- | --- | --- | --- |
| `serve.py` (default) | [4B](https://huggingface.co/thegovind/blink-4b), [MiMo 9B](https://huggingface.co/thegovind/blink-mimo-9b), [27B](https://huggingface.co/thegovind/blink-27b) | Text; screenshots opt-in | Reference path. |
| `serve_vllm.py` (opt-in) | 4B only | Text only | Higher throughput. |

On a Mac with Apple silicon, `blink_mlx.py` runs all three models with MLX for text decisions, memory permitting; see the [MLX guide](https://github.com/thegovind/blink/tree/main/examples/mlx). It is not a server.

| Model / server | JevBench p50 / p95 (ms) | c16 q/s | Longest TypeSafe p95 (s) |
| --- | ---: | ---: | ---: |
| 4B / default | 64 / 175 | 12.2 | 22.2 |
| 4B / vLLM | 52 / 127 | 34.7 | 16.4 |
| MiMo 9B / default | 63 / 257 | 10.3 | 33.6 |
| 27B / default | 126 / 785 | 4.0 | 106.0 |
| TypeSafe's hosted Jev (network-inclusive) | 652 / 722 | n/a | n/a |

blink JevBench numbers: earlier paired serving checks on 231 public items, one self-hosted replica, loopback HTTP. A release build re-passed the 231-item JevBench c1 quality check (80/111 hard, 199/231 total, hard ECE 0.069). Later changes touched only request checks, error handling and startup cleanup, not scoring. c1 is per-request latency; c16 is completed q/s. Longest TypeSafe documents: 16 requests.
Hosted Jev has a different network, hardware and population. No official JevBench score for blink has been published.

The tried 4B and MiMo INT8 builds changed too many answers. None is published; vLLM here runs BF16 weights.
MiMo's final vLLM read matched its earlier answers on all 231 public JevBench items but failed the 111-hard-item top-label ECE limit (0.122 vs 0.121). No MiMo vLLM server is published; use `serve.py`.
27B uses `serve.py` only. vLLM's agreement with `serve.py` on the 5-option web-action set was 489/500, below the 493/500 limit. This set has 500 text-only questions from the public Multimodal-Mind2Web dataset.
Long documents remain the main latency gap.

Details: [4B setup and checks](https://huggingface.co/thegovind/blink-4b/blob/v1.4/VLLM.md).
Try blink in the [Space](https://huggingface.co/spaces/thegovind/blink).
