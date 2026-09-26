# History

blink v1.0 used plain LoRA supervised fine-tuning over typed-decision rows with
soft and one-hot targets, followed by weight soups.

- blink-4b: a uniform average of T3, T4 step 300, and T4 final.
- blink-27b: T2 followed by T4.
- blink-mimo-9b: text-side decision training grafted back into the multimodal
  parent while preserving the parent vision tower.

The project did not use RL, RLCD, DPO, or quantization for the v1.0 training
recipe. See the model cards in `docs/models/` for per-model details and
licenses.
