"""Programmatic known-posterior decisions: exact probabilities as soft targets (calibration gold).

Every item states a random process or counts explicitly; the target distribution is computed exactly.
Distractor details (superseded procedures, voided rows, other cohorts) make naive readings wrong.
  python -m jevlab.synth_prob OUT.jsonl N [SEED]
"""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import date, timedelta
from fractions import Fraction

NAMES = ["Harlow", "Brightwater", "Kestrel", "Marlowe", "Oakridge", "Tamsin", "Verity", "Calder", "Ashby", "Lindqvist",
         "Okafor", "Petrova", "Nakamura", "Delgado", "Fontaine", "Rasmussen", "Iyer", "Moreau", "Castellano", "Whitlock"]
ITEMS = [("units", "defective"), ("invoices", "missing a purchase order"), ("claims", "flagged for fraud review"),
         ("parcels", "damaged in transit"), ("badges", "expired"), ("samples", "contaminated"), ("tickets", "misrouted"),
         ("laptops", "missing the security agent"), ("contracts", "missing a signature"), ("meters", "reading out of tolerance")]


def dstr(d):
    return d.strftime("%d %B %Y").lstrip("0")


def noul(state, instr, p_yes, t="The event happens.", f="The event does not happen.", fam="prob"):
    return {"state": state, "question": {"type": "noul", "instructions": instr, "criteria": {"true": t, "false": f}},
            "target": {"yes": float(p_yes), "no": 1 - float(p_yes)}, "family": fam}


def choice(state, instr, crit, dist, fam="prob"):
    return {"state": state, "question": {"type": "choice", "instructions": instr, "criteria": crit}, "target": dist, "family": fam}


def hypergeom(rng):
    thing, bad = rng.choice(ITEMS)
    N = rng.randint(8, 40)
    D = rng.randint(1, max(1, N // 3))
    k_new, k_old = rng.sample(range(2, min(8, N - D) + 1), 2) if N - D >= 3 else (2, 3)
    today = date(2026, rng.randint(1, 12), rng.randint(1, 28))
    eff = today - timedelta(days=rng.randint(3, 90)) if rng.random() < 0.7 else today + timedelta(days=rng.randint(3, 60))
    in_force = k_new if eff <= today else k_old
    org = rng.choice(NAMES)
    state = (f"{org} Receiving — Sampling Procedure SP-{rng.randint(2, 9)}\n"
             f"Revision history: the previous revision required inspectors to draw {k_old} {thing} at random, without replacement, from each lot. "
             f"The current revision, effective {dstr(eff)}, requires {k_new} {thing} drawn at random without replacement.\n"
             f"Lot record, {dstr(today)}: lot L-{rng.randint(1000, 9999)} contains {N} {thing}; the supplier's certificate lists {D} of them as {bad}. "
             f"The inspector's note says: \"usually one sample is enough to catch these\".\n"
             f"Today's date: {dstr(today)}.")
    p_none = Fraction(math.comb(N - D, in_force), math.comb(N, in_force))
    if rng.random() < 0.7:
        return noul(state, f"Will today's inspection sample of lot {state.split('lot ')[1].split(' ')[0]} include at least one of the {bad} {thing}?",
                    1 - p_none, f"At least one sampled item is {bad}.", f"No sampled item is {bad}.", "prob_hypergeom")
    p1 = Fraction(math.comb(D, 1) * math.comb(N - D, in_force - 1), math.comb(N, in_force))
    p2 = 1 - p_none - p1
    return choice(state, f"How many {bad} {thing} will today's inspection sample contain?",
                  {"none": f"No {bad} item in the sample", "exactly_one": f"Exactly one {bad} item", "two_or_more": f"Two or more {bad} items"},
                  {"none": float(p_none), "exactly_one": float(p1), "two_or_more": float(p2)}, "prob_hypergeom_count")


def bayes(rng):
    cond = rng.choice(["the fault", "the defect", "the infection", "an account takeover", "a counterfeit part", "a leak"])
    cohorts = {c: rng.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]) for c in rng.sample(["Line A", "Line B", "Line C", "night shift", "day shift", "region North", "region South", "legacy fleet", "new fleet"], 3)}
    c = rng.choice(list(cohorts))
    sens = rng.choice([0.8, 0.85, 0.9, 0.95, 0.99])
    spec = rng.choice([0.8, 0.9, 0.95, 0.97, 0.99])
    pos = rng.random() < 0.75
    prior = cohorts[c]
    if pos:
        post = prior * sens / (prior * sens + (1 - prior) * (1 - spec))
    else:
        post = prior * (1 - sens) / (prior * (1 - sens) + (1 - prior) * spec)
    table = "\n".join(f"  {k}: {v:.0%} of cases have {cond}" for k, v in cohorts.items())
    state = (f"Screening memo. Historical base rates by cohort:\n{table}\n"
             f"The screening check detects {cond} in {sens:.0%} of cases that have it (sensitivity) and correctly clears {spec:.0%} of cases that do not (specificity). "
             f"A vendor sheet claims the check is \"{max(sens, spec):.0%} accurate\".\n"
             f"Case 7731 belongs to {c}. Its screening result was {'positive' if pos else 'negative'}.")
    return noul(state, f"Does case 7731 have {cond}?", post, f"The case has {cond}.", f"The case does not have {cond}.", "prob_bayes")


def filtered_draw(rng):
    cats = rng.sample(["carrier Northline", "carrier Swiftpost", "carrier Redfern", "carrier Bayway", "carrier Orbit"], rng.randint(3, 4))
    rows, counts = [], {c: 0 for c in cats}
    n = rng.randint(8, 18)
    for i in range(n):
        c = rng.choice(cats)
        status = rng.choices(["shipped", "voided", "duplicate"], weights=[7, 2, 1])[0]
        month = rng.choice(["July", "August", "September"])
        rows.append(f"  #{4100 + i} | {month} | {c} | {status}")
        if status == "shipped" and month == "September":
            counts[c] += 1
    if sum(counts.values()) == 0:
        return None
    state = ("Shipment log (id | month | carrier | status):\n" + "\n".join(rows) +
             "\nAudit rule: the auditor picks one September shipment uniformly at random from the shipped entries; voided and duplicate entries are never picked.")
    tot = sum(counts.values())
    keys = {c: c.split()[-1].lower() for c in cats}
    return choice(state, "Which carrier will the audited shipment have used?", {keys[c]: c.title() for c in cats},
                  {keys[c]: counts[c] / tot for c in cats}, "prob_filtered_draw")


def reliability(rng):
    comps = [(n, rng.choice([0.01, 0.02, 0.05, 0.1, 0.2])) for n in rng.sample(["primary database", "replica", "load balancer", "API node 1", "API node 2", "cache", "DNS"], 4)]
    a, b, c, d = comps
    p_up = (1 - a[1]) * (1 - b[1] * c[1]) * (1 - d[1])
    state = (f"Service topology note. The service is available only if {a[0]} is up AND at least one of {b[0]} or {c[0]} is up AND {d[0]} is up. "
             f"Tomorrow's independent outage probabilities: {a[0]} {a[1]:.0%}, {b[0]} {b[1]:.0%}, {c[0]} {c[1]:.0%}, {d[0]} {d[1]:.0%}. "
             "Failures are independent. A dashboard tile shows 'redundant: yes'.")
    return noul(state, "Will the service be available tomorrow?", p_up, "The service is available.", "The service is unavailable.", "prob_reliability")


def lottery(rng):
    bidders = rng.sample(NAMES, rng.randint(3, 5))
    bids = {b: rng.choice([100, 120, 150, 150, 180, 200, 200]) for b in bidders}
    mx = max(bids.values())
    tied = [b for b in bidders if bids[b] == mx]
    late = rng.choice(bidders)
    rule = "Bids received after the deadline are rejected before ranking."
    valid = {b: v for b, v in bids.items() if b != late}
    mx = max(valid.values())
    tied = [b for b in valid if valid[b] == mx]
    lines = "\n".join(f"  {b}: ${bids[b]}{' (received after the deadline)' if b == late else ''}" for b in bidders)
    state = f"Parking permit auction.\nBids:\n{lines}\nRules: the highest valid bid wins; if several valid bids tie for highest, the winner is drawn uniformly at random among them. {rule}"
    return choice(state, "Who will be awarded the permit?", {b.lower(): b for b in bidders},
                  {b.lower(): (1 / len(tied) if b in tied else 0.0) for b in bidders}, "prob_lottery")


def repeated(rng):
    p = rng.choice([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    n = rng.randint(2, 8)
    what = rng.choice(["a support call escalates", "a delivery attempt fails", "a batch job times out", "a login attempt is challenged"])
    state = (f"Operations fact sheet: each time, independently, {what} with probability {p:.0%}. "
             f"Tomorrow there will be exactly {n} attempts. The shift lead remarks that 'it basically never happens twice'.")
    q = rng.random()
    if q < 0.5:
        return noul(state, f"Will it happen at least once tomorrow ({what})?", 1 - (1 - p) ** n, fam="prob_repeated")
    return noul(state, f"Will it happen at least twice tomorrow ({what})?", 1 - (1 - p) ** n - n * p * (1 - p) ** (n - 1), fam="prob_repeated2")


GENS = [hypergeom, bayes, filtered_draw, reliability, lottery, repeated]
W = [3, 3, 2, 1, 1, 2]


def main():
    out, n = sys.argv[1], int(sys.argv[2])
    seed = sys.argv[3] if len(sys.argv) > 3 else "prob"
    rng = random.Random(seed)
    k = 0
    with open(out, "w") as f:
        while k < n:
            g = rng.choices(GENS, weights=W)[0]
            it = g(rng)
            if not it:
                continue
            tgt = it["target"]
            if any(v < -1e-9 for v in tgt.values()) or abs(sum(tgt.values()) - 1) > 1e-6:
                continue
            gold = max(tgt, key=tgt.get)
            f.write(json.dumps({"id": f"{seed}:{k}", "src": it["family"], "state": it["state"], "question": it["question"],
                                "gold": gold, "target": tgt, "meta": {"group": f"{seed}:{k}", "family": it["family"]}}) + "\n")
            k += 1
    print("wrote", k)


if __name__ == "__main__":
    main()
