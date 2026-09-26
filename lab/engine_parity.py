"""Engine parity against blink's own references: gate A' applied to another engine (CPU only).

  python engine_parity.py --refs refs.jsonl --candidate vllm-outputs.jsonl --name vllm-fp32 [--out report.json]

refs.jsonl comes from fp32_reference.py (plain bf16 and FP32 per question); the candidate is any engine's output
rows in run_requests.py / vllm_blink.py shape. Reported, per the pre-registered rule set (same thresholds as A'):
  (i)   argmax agreement with FP32 >= the plain bf16 path's - 0.005
  (ii)  p99 |dp| vs FP32 <= the plain path's + 0.005
  (iii) max |dp| vs FP32 <= the plain path's + 0.01
plus a flip report: flips against the plain path by stable id, with each flip's FP32 top-two margin and
whether the candidate or the plain path matched FP32 on it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _argmax(p):
    return max(range(len(p)), key=lambda i: (p[i], -i))


def _vec(ans: dict, keys: list[str]) -> list[float]:
    if ans["type"] == "noul":
        pr = ans.get("probabilities") or {"yes": ans["noul"], "no": 1 - ans["noul"]}
    else:
        pr = ans["probabilities"]
    return [float(pr[k]) for k in keys]


def _pct(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))] if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--name", default="candidate")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cand = {}
    with open(a.candidate) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("output"):
                    cand[r["id"]] = r["output"]["answers"]
    err = {"plain": [], "cand": []}
    agree = {"plain": 0, "cand": 0}
    flips, n, missing = [], 0, 0
    with open(a.refs) as fh:
        for line in fh:
            r = json.loads(line)
            ans = cand.get(r["id"], {}).get(r["q"])
            if ans is None:
                missing += 1
                continue
            pc, p16, p32 = _vec(ans, r["keys"]), r["bf16"], r["fp32"]
            n += 1
            ref = _argmax(p32)
            for key, p in (("plain", p16), ("cand", pc)):
                err[key].append(max(abs(x - y) for x, y in zip(p, p32)))
                agree[key] += _argmax(p) == ref
            if _argmax(pc) != _argmax(p16):
                top = sorted(p32, reverse=True)
                flips.append({"id": r["id"], "q": r["q"], "fp32_margin": round(top[0] - top[1], 5),
                              "plain_matches_fp32": _argmax(p16) == ref, "candidate_matches_fp32": _argmax(pc) == ref})
    s = {}
    for key in ("plain", "cand"):
        e = err[key]
        s[key] = {"argmax_agreement_with_fp32": agree[key] / max(n, 1), "max_dp_vs_fp32": max(e) if e else 0.0,
                  "p99_dp_vs_fp32": _pct(e, 0.99), "p90_dp_vs_fp32": _pct(e, 0.90), "median_dp_vs_fp32": _pct(e, 0.5)}
    rules = {
        "i_argmax": s["cand"]["argmax_agreement_with_fp32"] >= s["plain"]["argmax_agreement_with_fp32"] - 0.005,
        "ii_p99": s["cand"]["p99_dp_vs_fp32"] <= s["plain"]["p99_dp_vs_fp32"] + 0.005,
        "iii_max": s["cand"]["max_dp_vs_fp32"] <= s["plain"]["max_dp_vs_fp32"] + 0.01,
    }
    report = {"name": a.name, "questions": n, "missing": missing, "plain": s["plain"], "candidate": s["cand"],
              "rules": rules, "passes": all(rules.values()), "flips_vs_plain": flips}
    print(json.dumps({k: v for k, v in report.items() if k != "flips_vs_plain"}, indent=1))
    print(f"flips vs plain: {len(flips)}; candidate matched FP32 on {sum(f['candidate_matches_fp32'] for f in flips)}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
