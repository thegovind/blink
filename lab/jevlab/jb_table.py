"""Tabulate jb_run summaries: python -m jevlab.jb_table DIR_OR_FILES..."""
import glob
import json
import sys


def main():
    files = []
    for a in sys.argv[1:] or ["."]:
        files += sorted(glob.glob(a + "/*.json")) if not a.endswith(".json") else [a]
    hdr = f"{'run':46} {'easy':>5} {'std':>5} {'hard':>5} {'ECE':>5} {'TVD':>5} {'C':>5} {'Ipub':>5} {'tok':>5} {'p50ms':>6} {'p95ms':>6} {'S':>5} {'$':>5} {'est':>5}"
    print(hdr)
    for f in files:
        try:
            s = json.load(open(f))["summary"]
        except Exception:
            continue
        t = s["tiers"]
        name = f.split("/")[-1][:-5]
        print(f"{name[:46]:46} {t['easy']:5.3f} {t['standard']:5.3f} {t['hard']:5.3f} {s['hard_ece']:5.3f} "
              f"{s['prob_tvd']:5.3f} {s['calibration']:5.1f} {s['intelligence_public']:5.1f} "
              f"{s['input_tokens_per_decision']:5.0f} {s.get('p50_s', 0) * 1000:6.1f} {s.get('p95_s', 0) * 1000:6.1f} "
              f"{s.get('speed') or 0:5.1f} {s.get('cost') or 0:5.1f} {s.get('est_score_public_only') or 0:5.1f}")
        if "-v" in sys.argv[0:1] or len(files) <= 12:
            print("     hard:", " ".join(f"{k}={v:.2f}" for k, v in s["hard_by_family"].items()))


if __name__ == "__main__":
    main()
