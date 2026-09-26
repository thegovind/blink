"""Exact next-token label scoring on a causal LM (prefill only, no generation).

For each prompt the score of option i is the next-token logit of its label token at
the last prompt position, restricted to the offered labels and normalised in FP32.
Batches are right-padded and run without an attention mask: every layer is causal
(attention and the recurrent DeltaNet/conv layers), so tokens after the last real
position cannot change its hidden state. Only the needed lm_head rows are applied.
"""
from __future__ import annotations

import math
import os
import time

import torch


def load_model(model_id, revision=None, dtype=torch.bfloat16, device="cuda", adapter=None, attn="sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype, device_map=device,
                                                 attn_implementation=attn)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


class Scorer:
    def __init__(self, model, tokenizer, token_budget=32768, max_tokens=131072, temperature=1.0, head_path=None):
        self.model = model
        self.tok = tokenizer
        self.token_budget = token_budget
        self.max_tokens = max_tokens
        self.temperature = float(temperature)
        self.device = next(model.parameters()).device
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.body = base.model
        self.lm_head = base.lm_head.weight
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    @torch.no_grad()
    def hidden_last(self, seqs):
        """Final-norm hidden state at the last position of each sequence (list of id lists)."""
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        ids = ids.to(self.device)
        out = self.body(input_ids=ids, use_cache=False)
        h = out.last_hidden_state
        idx = torch.tensor([len(s) - 1 for s in seqs], device=self.device)
        return h[torch.arange(len(seqs), device=self.device), idx]

    @torch.no_grad()
    def label_logits(self, seqs, cands):
        h = self.hidden_last(seqs).float()
        outs = []
        for i, c in enumerate(cands):
            w = self.lm_head[torch.tensor(c, device=self.device)].float()
            outs.append((w @ h[i]).cpu())
        return outs

    def batches(self, lengths):
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        batch, cur_max = [], 0
        for i in order:
            m = max(cur_max, lengths[i])
            if batch and m * (len(batch) + 1) > self.token_budget:
                yield batch
                batch, m = [], lengths[i]
            batch.append(i)
            cur_max = m
        if batch:
            yield batch

    def score(self, seqs, cands, temperature=None):
        """-> list of FP32 log-prob tensors over each prompt's candidates (None when over capacity)."""
        T = self.temperature if temperature is None else float(temperature)
        res = [None] * len(seqs)
        lengths = [len(s) for s in seqs]
        for b in self.batches(lengths):
            ok = [i for i in b if lengths[i] <= self.max_tokens]
            if not ok:
                continue
            logits = self.label_logits([seqs[i] for i in ok], [cands[i] for i in ok])
            for i, z in zip(ok, logits):
                res[i] = torch.log_softmax(z / T, dim=-1)
        return res


def probs_from_logp(logp):
    p = torch.exp(logp.double())
    p = p / p.sum()
    return [float(x) for x in p]


def timed(fn):
    def wrap(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        return r, (time.perf_counter() - t) * 1000

    return wrap


def assert_finite(logp):
    if not all(math.isfinite(float(x)) for x in logp):
        raise ValueError("non-finite log-probabilities")
