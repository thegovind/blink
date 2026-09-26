"""Served-payload equivalence for the E1 H1 gate sets (CPU only; tokenizer, no model).

  uv run --with transformers==5.17.0 python lab/payload_check.py --tokenizer <dir with the blink tokenizer files> \
      --typesafe typesafe/requests.jsonl --typesafe-refs refs-4b-typesafe.jsonl --workflow data/workflow/dev.jsonl

The offline gate runners render `blink.as_state(state)`: a state that is a JSON-encoded string is parsed first (the
TypeSafe requests store each document that way). serve.py renders the state it receives. This checks, per question,
that a client posting the document as a JSON object (the TypeSafe document itself, and what the Space's systemone
endpoint does with a JSON string) gets token-for-token the prompt the gate measured, and ties that to the input
token counts recorded in the gate's FP32 reference file. It also reports how a client posting the stored string
unchanged differs: that is the release path's own behaviour for strings, unchanged in v1.1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "space"))
os.environ.setdefault("BLINK_MOCK", "1")

import blink  # noqa: E402


def rows(path: str):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def over_http(state, questions):
    """What serve.py's handler hands to blink after the JSON round trip of a request body."""
    body = json.loads(json.dumps({"state": state, "questions": questions}, ensure_ascii=False).encode("utf-8"))
    return body["state"], body["questions"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--typesafe", required=True)
    ap.add_argument("--typesafe-refs", required=True,
                    help="the gate's FP32 reference file (input_tokens: the request's total, on each of its rows)")
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    eng = blink.TorchEngine.__new__(blink.TorchEngine)
    eng.tok = AutoTokenizer.from_pretrained(a.tokenizer)
    eng.labels, eng.label_ids = eng._verify_labels()

    ref_tokens: dict = {}
    for r in rows(a.typesafe_refs):
        assert ref_tokens.setdefault(r["id"], r["input_tokens"]) == r["input_tokens"], r["id"]
    ts = {"requests": 0, "questions": 0, "json_string_states": 0, "object_payload_identical": 0,
          "requests_in_refs": 0, "requests_matching_gate_input_tokens": 0, "string_payload_identical": 0, "tokens_object": 0, "tokens_string": 0}
    for r in rows(a.typesafe):
        state, qs = r["state"], r["questions"]
        ts["requests"] += 1
        ts["json_string_states"] += isinstance(state, str) and blink.as_state(state) is not state
        gate = eng.render(blink.as_state(state), qs)
        obj = eng.render(*over_http(blink.as_state(state), qs))
        raw = eng.render(*over_http(state, qs))
        if r["id"] in ref_tokens:
            ts["requests_in_refs"] += 1
            ts["requests_matching_gate_input_tokens"] += sum(len(g["ids"]) for g in gate) == ref_tokens[r["id"]]
        for g, o, s in zip(gate, obj, raw):
            ts["questions"] += 1
            ts["object_payload_identical"] += g["ids"] == o["ids"] and g["cand"] == o["cand"]
            ts["string_payload_identical"] += g["ids"] == s["ids"]
            ts["tokens_object"] += len(o["ids"])
            ts["tokens_string"] += len(s["ids"])
    wf = {"rows": 0, "json_string_states": 0, "round_trip_identical": 0}
    for r in rows(a.workflow):
        state = r["state"]
        wf["rows"] += 1
        wf["json_string_states"] += isinstance(state, str) and blink.as_state(state) is not state
        wf["round_trip_identical"] += over_http(blink.as_state(state), {})[0] == blink.as_state(state) == state
    report = {"typesafe": ts, "workflow": wf}
    print(json.dumps(report, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
