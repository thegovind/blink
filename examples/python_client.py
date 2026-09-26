from __future__ import annotations

import json
import urllib.request


def decide(state, questions, url: str = "http://127.0.0.1:8000/v1/systemone") -> dict:
    body = json.dumps({"state": state, "questions": questions}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


if __name__ == "__main__":
    print(json.dumps(decide(
        "Order 4411 arrived with a cracked screen. The customer wants a refund.",
        {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"returns": "Damaged or wrong items", "billing": "Charges and refunds"},
            }
        },
    ), indent=2))
