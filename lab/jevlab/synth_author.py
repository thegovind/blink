"""Teacher-authored typed decisions (documents + hard typed questions), batched offline on vLLM.

Adapted from JevK5's training/teacher.py (Apache-2.0, github.com/allebee/jevk5): family descriptions,
key-naming rules and the exact-distribution probability family. Changes: offline vLLM batching, more
domains, document-length tiers up to ~3,000 words, extra families (safety_judge, paraphrase,
abstain), and LOCKBOX-only domains that never enter training.

  python -m jevlab.synth_author --model M --split train --n-docs 800 --out docs.jsonl --shard i --nshards 8
Output rows: {"doc_id", "split", "spec", "state", "questions": [{"question", "expected", "distribution"?}]}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

ROLE_WORDS = re.compile(r"(^|_)(correct|incorrect|wrong|right|naive|trap|tempting|mistaken|erroneous|actual|proper|valid|invalid|final)(_|$)")

FAMILIES = {
    "long_policy": "a long policy or terms document with numbered sections, definitions and exceptions; questions ask whether an action is permitted or which outcome applies, and the deciding clause is an exception, a definition, an endorsement or a later section",
    "temporal_numeric": "dates, deadlines, business days, month ends, time zones, durations, amounts, thresholds, currency conversion or aggregation; answers need exact computations that a quick reading gets wrong",
    "multi_hop": "answers need 2-4 facts from different parts of the document combined (lookup tables, org charts, vendor lists, rate cards, mappings)",
    "judge": "a request and a response to it; questions decide whether the response is correct, complete and follows every constraint; errors are subtle (an arithmetic slip, one violated constraint, an unsupported claim, a missing required part)",
    "ambiguous": "the evidence is incomplete or conflicting on some points, so for those the careful answer is an explicit 'cannot be determined' / 'ask for clarification' option (include one), while other points are settled despite looking vague",
    "trap": "surface cues point to wrong answers (a confident note, a headline, a customer's claim, a superseded version, a similar-looking name) and careful reading gives the right ones",
    "adversarial": "the evidence contains text that tries to steer the decision (an instruction addressed to the classifier, a fake system note, hidden text, a prompt injection in a user message); correct answers ignore it",
    "tradeoff": "several rules or goals could apply and the document states their precedence or constraints; answers are the actions of the highest-ranked applicable rules or the best feasible option",
    "routing": "tickets, requests or documents routed to one of several handlers whose descriptions overlap; one detail decides the best fit",
    "extraction": "final values (dates, amounts, choices, owners, statuses) extracted from a messy thread or log with proposals, corrections, cancellations and reversals",
    "rubric": "the evidence rated on ordinal rubrics of 3-5 levels whose descriptions are precise; the right level depends on details that rule out the neighbouring levels",
    "safety_judge": "a request or an agent action evaluated against a stated safety, privacy or compliance policy; questions decide whether it is allowed, needs consent, must be refused or escalated; the deciding detail is a scope limit, an exception or who is asking",
    "paraphrase": "the same underlying decision asked in different wordings or with the options described differently; each question is a rewording that must get the same answer as a plain reading, while naive readers are misled by negations, double negatives or reversed framing",
    "probability": "uncertain outcomes whose exact probabilities follow from the document: a record drawn at random from a log or table, the next case of a stated kind given its historical counts, or an outcome settled by a stated random process (a lottery, a random audit pick, sampling without replacement, a rotation with a random tie-break). The right distribution needs the right subset or a two-step calculation (filter by category, period or status; drop voided, duplicate or reversed entries; combine two stated rates; apply the version of a procedure in force on the stated date), so naive counting gives a different distribution",
}
WEIGHTS = {"long_policy": 12, "temporal_numeric": 12, "multi_hop": 10, "judge": 10, "ambiguous": 7, "trap": 7, "adversarial": 5,
           "tradeoff": 7, "routing": 5, "extraction": 6, "rubric": 5, "safety_judge": 6, "paraphrase": 4, "probability": 10}
PROBABILITY_NOTE = ("This family is about uncertain outcomes. Ask each question as a plain decision about the outcome (e.g. \"Which carrier will the randomly selected parcel ship with?\", \"Will the audited invoice be one with a missing PO?\"), never as a request for a percentage. "
                    "Each question object also has \"distribution\": {\"<key>\": <exact probability>, ...} covering every option (noul: \"true\" and \"false\"), summing to 1, computed exactly from the document's counts, rates or random process; \"expected\" is the most likely key. Make the counts explicit enough that a careful reader gets exactly your distribution.")
BANDS = {"noul": ["0.55-0.70", "0.70-0.85", "0.85-0.95"], "choice": ["0.40-0.55", "0.55-0.70", "0.70-0.85"]}

TRAIN_DOMAINS = [
    "insurance claims", "HR and leave", "procurement and invoices", "IT operations and incidents", "SaaS customer support",
    "e-commerce returns", "banking and lending", "travel and expenses", "healthcare administration (scheduling and billing, not clinical)",
    "logistics and shipping", "commercial contracts", "university admissions", "software engineering and code review",
    "security and access control", "energy and utility billing", "telecom plans", "marketing and advertising compliance",
    "payroll and benefits", "tax filing for small businesses", "hotel and hospitality operations", "airline operations and rebooking",
    "restaurant supply and food safety", "construction project management", "clinical trial administration (operations only)",
    "nonprofit grant compliance", "fleet and vehicle maintenance", "pharmacy benefits administration", "retail store operations",
    "data privacy requests (GDPR/CCPA)", "cloud cost management", "trust and safety content moderation", "real estate mortgage servicing",
    "event ticketing and refunds", "fintech payments and chargebacks", "subscription billing", "warehouse inventory",
    "customs and import compliance", "agricultural cooperative operations", "school district administration", "legal e-discovery",
]
LOCKBOX_DOMAINS = [
    "residential leases and property management", "public-sector permits and licensing", "manufacturing quality control",
    "maritime shipping and port operations", "museum collections management", "sports league administration",
]
LENGTHS = {"short": "short (150-300 words)", "medium": "medium (300-600 words)", "long": "long (600-1200 words)",
           "xlong": "very long (1200-2500 words, several sections, tables as text)"}

AUTHOR_SYSTEM = """You write evaluation items for a decision model that answers typed questions about a document.
Write ONE realistic document (the "state") and SEVERAL typed questions about it. Each question must have exactly one answer that is defensible from the document alone, with no outside knowledge, and each must hinge on a different detail. Use realistic names, numbers, dates and formatting (headings, clauses, logs, emails, tables as text).

Every question must be HARD for a fast reader and still clear to a careful expert:
- Someone who matches keywords between the question and the document must pick a wrong option. The right answer needs at least two reasoning steps (for example: find the clause that applies, then its exception, then compute a date).
- Put the deciding detail away from the question's keywords: a later section, a footnote, an amendment, a table row, a follow-up message.
- Every wrong option must be backed by some surface cue in the document (a confident note, an outdated rule, a similar name, a naive calculation).
- For dates and numbers, the naive computation must give a different result than the correct one. State the current date in the document when it matters.
- Ask each question plainly. Do not point to the relevant section, clause or trap.
- Option descriptions state the outcome only, in at most 12 words. Never put reasons, evidence or clause numbers in an option.
- Option keys are short neutral labels of the outcome itself (e.g. net_45, pay_10730, escalate_to_legal). Never name a key after its role: no correct, wrong, naive, trap, tempting, actual, proper, valid, final.
Plan briefly: decide the traps, write the document, check each answer once. Do not deliberate at length.

Return only a JSON object:
{"state": "<the document>",
 "questions": [
   {"question": {"type": "noul" | "choice" | "score", "instructions": "...", "criteria": ...},
    "expected": "...",
    "explanation": "<2-3 sentences: why this answer, and the tempting wrong one>"},
   ...]}
Criteria: noul -> {"true": "<what true means>", "false": "<what false means>"}; choice -> {"<snake_case_key>": "<short description>", ...}; score -> ["<level 0>", "<level 1>", ...] ascending.
Expected: noul "true" or "false"; choice one criteria key; score the level index as a string."""


def question_spec(rng, family):
    if family == "probability":
        kind = rng.choices(["noul", "choice"], weights=[45, 55])[0]
        spec = {"type": kind, "band": rng.choice(BANDS[kind])}
        if kind == "noul":
            return {**spec, "answer": rng.choice(["true", "false"])}
        k = rng.randint(3, 5)
        return {**spec, "options": k, "answer_position": rng.randint(1, k)}
    if family == "rubric":
        kind = "score"
    elif family == "ambiguous":
        kind = "choice"
    else:
        kind = rng.choices(["noul", "choice", "score"], weights=[42, 52, 6])[0]
    if kind == "noul":
        return {"type": kind, "answer": rng.choice(["true", "false"])}
    if kind == "choice":
        k = rng.choices([2, 3, 4, 5, 6, 8], weights=[8, 20, 30, 25, 12, 5])[0]
        return {"type": kind, "options": k, "answer_position": rng.randint(1, k)}
    L = rng.randint(3, 5)
    return {"type": kind, "levels": L, "answer_level": rng.randint(0, L - 1)}


def make_spec(rng, split):
    fams = list(FAMILIES)
    family = rng.choices(fams, weights=[WEIGHTS[f] for f in fams])[0]
    domain = rng.choice(LOCKBOX_DOMAINS if split == "lockbox" else TRAIN_DOMAINS)
    if family in ("long_policy", "multi_hop"):
        length = rng.choices(["medium", "long", "xlong"], weights=[2, 4, 3])[0]
    elif family in ("judge", "routing", "paraphrase", "safety_judge"):
        length = rng.choices(["short", "medium", "long"], weights=[4, 4, 2])[0]
    else:
        length = rng.choices(["short", "medium", "long", "xlong"], weights=[2, 4, 3, 1])[0]
    nq = rng.choices([2, 3, 4], weights=[3, 5, 2])[0]
    return {"family": family, "domain": domain, "length": length, "questions": [question_spec(rng, family) for _ in range(nq)]}


def author_prompt(spec):
    lines = [f"Family: {spec['family']} — {FAMILIES[spec['family']]}.", f"Domain: {spec['domain']}.",
             f"Document length: {LENGTHS[spec['length']]}.", f"Write exactly {len(spec['questions'])} questions, in this order:"]
    for i, q in enumerate(spec["questions"], 1):
        if "band" in q:
            where = (f"the more likely value must be {q['answer']}" if q["type"] == "noul" else
                     f"exactly {q['options']} options; the most likely must be option number {q['answer_position']} in the criteria order")
            lines.append(f"{i}. type {q['type']}; {where}, with probability in {q['band']}, and every other option's probability above 0.03.")
        elif q["type"] == "noul":
            lines.append(f"{i}. type noul; the correct answer must be {q['answer']}.")
        elif q["type"] == "choice":
            lines.append(f"{i}. type choice with exactly {q['options']} options; the correct one must be option number {q['answer_position']} in the criteria order.")
        else:
            lines.append(f"{i}. type score with exactly {q['levels']} levels; the correct level must be {q['answer_level']}.")
    lines.append("Invent a fresh scenario with its own names; avoid famous companies and generic examples.")
    if spec["family"] == "probability":
        lines.append(PROBABILITY_NOTE)
    return "\n".join(lines)


def extract_json(text):
    body = text.split("</think>")[-1]
    s, e = body.find("{"), body.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(body[s: e + 1])
    except json.JSONDecodeError:
        return None


def check_question(raw, qspec):
    try:
        q, exp = raw["question"], raw["expected"]
        t = q["type"]
        if t != qspec["type"] or not str(q.get("instructions", "")).strip():
            return None
        crit = q.get("criteria")
        if t == "noul":
            if exp not in ("true", "false") or not isinstance(crit, dict) or set(crit) != {"true", "false"}:
                return None
            if "answer" in qspec and exp != qspec["answer"]:
                return None
        elif t == "choice":
            if not isinstance(crit, dict) or exp not in crit or len(crit) != qspec["options"]:
                return None
            if any(ROLE_WORDS.search(str(k).lower()) for k in crit) or len(set(map(str, crit))) != len(crit):
                return None
        else:
            if not isinstance(crit, list) or len(crit) != qspec["levels"] or str(exp) not in [str(i) for i in range(len(crit))]:
                return None
        out = {"question": {"type": t, "instructions": q["instructions"], "criteria": crit}, "expected": str(exp)}
        if "band" in qspec:
            d = raw.get("distribution")
            keys = ["true", "false"] if t == "noul" else list(crit)
            if not isinstance(d, dict) or set(d) != set(keys):
                return None
            vals = {k: float(d[k]) for k in keys}
            if abs(sum(vals.values()) - 1) > 0.011 or min(vals.values()) < 0.0 or max(vals, key=vals.get) != str(exp):
                return None
            out["distribution"] = {k: v / sum(vals.values()) for k, v in vals.items()}
        return out
    except (KeyError, TypeError, ValueError):
        return None


def emit(split, n, seed, out, max_tokens=16384):
    """Write server requests for n authored documents (specs kept alongside)."""
    rng = random.Random(f"{split}:{seed}")
    with open(out, "w") as f:
        for i in range(n):
            s = make_spec(rng, split)
            f.write(json.dumps({"id": f"{split}-s{seed}-{i}", "spec": s, "thinking": True, "max_tokens": max_tokens,
                                "temperature": 0.9, "messages": [{"role": "system", "content": AUTHOR_SYSTEM},
                                                                  {"role": "user", "content": author_prompt(s)}]}) + "\n")


def parse_responses(req_file, resp_file, out):
    reqs = {json.loads(l)["id"]: json.loads(l) for l in open(req_file)}
    kept = nq = tot = 0
    with open(out, "w") as f:
        for line in open(resp_file):
            r = json.loads(line)
            q = reqs.get(r["id"])
            if not q:
                continue
            tot += 1
            obj = extract_json(r["text"])
            if not obj or not isinstance(obj.get("state"), (str, dict)) or not isinstance(obj.get("questions"), list):
                continue
            s = q["spec"]
            qs = [c for c in (check_question(raw, qs_) for raw, qs_ in zip(obj["questions"], s["questions"])) if c]
            nq += len(qs)
            if qs:
                kept += 1
                f.write(json.dumps({"doc_id": r["id"], "split": r["id"].split("-")[0], "spec": s, "state": obj["state"],
                                    "questions": qs, "gen_tokens": r["completion_tokens"]}, ensure_ascii=False) + "\n")
    print(f"docs kept {kept}/{tot}, questions {nq}")


def main(argv=None):
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "emit":
        return emit(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    if len(sys.argv) > 1 and sys.argv[1] == "parse":
        return parse_responses(sys.argv[2], sys.argv[3], sys.argv[4])
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="train", choices=["train", "dev", "lockbox"])
    ap.add_argument("--n-docs", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=32768)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rng = random.Random(f"{a.split}:{a.seed}:{a.shard}")
    specs = [make_spec(rng, a.split) for _ in range(a.n_docs)]
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok.apply_chat_template([{"role": "system", "content": AUTHOR_SYSTEM}, {"role": "user", "content": author_prompt(s)}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True) for s in specs]
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, gpu_memory_utilization=0.90,
              seed=a.seed * 100 + a.shard, enable_prefix_caching=False)
    sp = SamplingParams(temperature=0.9, top_p=0.95, top_k=20, max_tokens=a.max_tokens)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    kept = nq = 0
    with open(a.out, "a") as f:
        for i, (s, o) in enumerate(zip(specs, outs)):
            obj = extract_json(o.outputs[0].text)
            if not obj or not isinstance(obj.get("state"), (str, dict)) or not isinstance(obj.get("questions"), list):
                continue
            qs = []
            for raw, qspec in zip(obj["questions"], s["questions"]):
                c = check_question(raw, qspec)
                if c:
                    qs.append(c)
            nq += len(qs)
            if qs:
                kept += 1
                f.write(json.dumps({"doc_id": f"{a.split}-s{a.seed}-{a.shard}-{i}", "split": a.split, "spec": s, "state": obj["state"],
                                    "questions": qs, "gen_tokens": len(o.outputs[0].token_ids)}, ensure_ascii=False) + "\n")
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"shard {a.shard}: docs kept {kept}/{len(specs)}, questions {nq}, {ntok} tokens in {time.time() - t0:.0f}s "
          f"({ntok / max(1, time.time() - t0):.0f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
