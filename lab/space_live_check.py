"""Live check of the blink Space: every served model answers three routing questions correctly, through the JSON API.

  uv run --with gradio_client python lab/space_live_check.py [--anonymous] [--wait-minutes 25] [--ask]

--ask also checks free-form asks through /v1_ask (draft with the Space's drafter, then decide with each model):
17 x 23 must come back as 391 and "Is 91 a prime number?" as no; strawberry is reported, not asserted.

With --anonymous no token is sent (checks the Space as a public visitor would see it).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from gradio_client import Client

MODELS = ["thegovind/blink-4b", "thegovind/blink-mimo-9b"]
Q = {"queue": {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "Charges and refunds", "technical": "Bugs and errors", "shipping": "Parcels"}},
     "urgent": {"type": "noul", "instructions": "Does this need a reply today?"}}
ASKS = [("What's 17 x 23?", "391"), ("Is 91 a prime number?", "no"), ("How many r in strawberry", None)]
CASES = [("My card was charged twice for order 88213.", "billing"),
         ("The app crashes every time I open settings.", "technical"),
         ("My parcel has not arrived after two weeks.", "shipping")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anonymous", action="store_true")
    ap.add_argument("--wait-minutes", type=float, default=25)
    ap.add_argument("--ask", action="store_true", help="also check free-form asks via /v1_ask")
    a = ap.parse_args()
    token = None if a.anonymous else os.environ["HF_TOKEN"]
    if a.anonymous:
        os.environ.pop("HF_TOKEN", None)
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    deadline = time.time() + a.wait_minutes * 60
    while True:
        try:
            c = Client("thegovind/blink", token=token, verbose=False)
            rows, bad = [], 0
            for m in MODELS:
                for state, want in CASES:
                    t0 = time.time()
                    out = c.predict(state, Q, None, m, api_name="/v1_systemone")
                    got = out["answers"]["queue"]["choice"]
                    ok = got == want and out["meta"]["model"] == m and out["meta"]["engine"] == "torch"
                    bad += not ok
                    rows.append(f"{m}: {got} (want {want}) p={out['answers']['queue']['probabilities'][got]:.3f} "
                                f"model_ms={out['meta'].get('model_ms')} wall={time.time() - t0:.1f}s {'ok' if ok else 'FAIL'}")
            if a.ask:
                for m in MODELS:
                    for ask, want in ASKS:
                        t0 = time.time()
                        out = c.predict(ask, m, api_name="/v1_ask")
                        (qk, q), = out["draft"]["questions"].items()
                        ans = out["answers"][qk]
                        if ans["type"] == "noul":
                            got = "yes" if ans["noul"] >= 0.5 else "no"
                            p = ans["probabilities"][got]
                        else:
                            got = ans["choice"]
                            p = ans["probabilities"][got]
                            if ans["type"] == "choice" and isinstance(q.get("criteria"), dict):
                                got = str(q["criteria"][got])
                        ok = want is None or got.strip().lower() == want
                        bad += not ok
                        au = out["draft"]["author"]
                        rows.append(f"{m}: ask {ask!r} -> {q['type']} {got!r} p={p:.3f} drafted by {au['model']} "
                                    f"({au['generated_tokens']} tokens, {au['model_ms']:.0f} ms) wall={time.time() - t0:.1f}s "
                                    f"{'ok' if ok and want else ('reported' if ok else 'FAIL')}")
            print("\n".join(rows), flush=True)
            if bad:
                raise RuntimeError(f"{bad} checks failed")
            print("LIVE_CHECK_OK", "(anonymous)" if a.anonymous else "(owner token)")
            return
        except Exception as exc:  # noqa: BLE001 - keep polling until the deadline
            if time.time() > deadline:
                sys.exit(f"live check did not pass: {exc!r}")
            print(f"not ready yet ({type(exc).__name__}: {str(exc)[:160]}); retrying in 60 s", flush=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
