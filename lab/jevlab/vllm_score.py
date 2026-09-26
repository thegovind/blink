"""vLLM scoring backend: next-token label distribution restricted to the offered labels.

Uses allowed_token_ids + logprobs_mode="processed_logprobs": the returned logprobs are the model's
log-softmax over the allowed label tokens only (temperature 1), i.e. the same quantity as the HF
Scorer (up to kernel numerics). Adapters must be merged into a checkpoint first (merge.py).
"""
from __future__ import annotations

import os

import torch


class VLLMScorer:
    def __init__(self, model, max_model_len=65536, gpu_mem=0.90, temperature=1.0, tp=1, max_num_seqs=256,
                 max_num_batched_tokens=32768):
        from vllm import LLM

        self.llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                       max_logprobs=256, logprobs_mode="processed_logprobs", enable_prefix_caching=False,
                       max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, seed=0)
        self.max_tokens = max_model_len - 1
        self.temperature = float(temperature)

    def score(self, seqs, cands, temperature=None):
        from vllm import SamplingParams

        T = self.temperature if temperature is None else float(temperature)
        res = [None] * len(seqs)
        idx = [i for i, s in enumerate(seqs) if len(s) <= self.max_tokens]
        prompts = [{"prompt_token_ids": seqs[i]} for i in idx]
        sps = [SamplingParams(max_tokens=1, temperature=1.0, logprobs=len(cands[i]), allowed_token_ids=list(cands[i]),
                              detokenize=False) for i in idx]
        outs = self.llm.generate(prompts, sps, use_tqdm=False)
        for i, o in zip(idx, outs):
            lp = o.outputs[0].logprobs[0]
            z = torch.tensor([lp[t].logprob if t in lp else float("-inf") for t in cands[i]], dtype=torch.float64)
            res[i] = torch.log_softmax(z / T, dim=-1).float()
        return res
