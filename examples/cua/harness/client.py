"""Loopback blink requests, optional three-server fanout, and an oracle-only CPU mock."""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx


class BlinkClientError(RuntimeError):
    """A server error or an invalid typed-decision response."""


@dataclass(frozen=True)
class Decision:
    answers: dict
    usage: dict
    wall_ms: float
    server: str
    per_question_ms: dict[str, float] = field(default_factory=dict)


def _validate(data: dict, questions: dict) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise BlinkClientError("Blink response has no answers map")
    for key, q in questions.items():
        a = data["answers"].get(key)
        if not isinstance(a, dict) or not isinstance(a.get("probabilities"), dict):
            raise BlinkClientError(f"Blink answer {key!r} has no probabilities map")
        probs = a["probabilities"]
        if (not probs or not all(isinstance(p, (float, int)) and math.isfinite(p) and 0 <= p <= 1
                                 for p in probs.values()) or abs(sum(probs.values()) - 1) > 0.01):
            raise BlinkClientError(f"Blink answer {key!r} has invalid probabilities")
        if q["type"] == "choice":
            if set(probs) != set(q["criteria"]) or a.get("choice") not in probs:
                raise BlinkClientError(f"Blink answer {key!r} does not match offered choices")
        elif q["type"] == "noul":
            if (set(probs) != {"yes", "no"} or not isinstance(a.get("noul"), (float, int))
                    or not math.isfinite(a["noul"]) or abs(a["noul"] - probs["yes"]) > 0.01):
                raise BlinkClientError(f"Blink answer {key!r} has no valid yes probability")
        else:
            raise BlinkClientError(f"Unsupported question type: {q['type']!r}")
    usage = data.get("usage")
    if not isinstance(usage, dict) or not all(isinstance(usage.get(k), int)
                                               for k in ("input_tokens", "visual_tokens")):
        raise BlinkClientError("Blink response has no input_tokens/visual_tokens usage")


class BlinkClient:
    def __init__(self, url: str, *, timeout: float = 60):
        parsed = urlparse(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Blink server must be an http(s) URL without credentials, query or fragment")
        self.server = url.rstrip("/")
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5), trust_env=False)

    async def ask(self, state: dict, questions: dict, **_mock_only) -> Decision:
        started = time.perf_counter()
        try:
            response = await self.http.post(
                f"{self.server}/v1/systemone",
                json={"model": "blink", "state": state, "questions": questions},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BlinkClientError(f"{self.server}: HTTP {exc.response.status_code}: "
                                   f"{exc.response.text[:400]}") from exc
        except httpx.RequestError as exc:
            raise BlinkClientError(f"{self.server}: {exc}") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise BlinkClientError(f"{self.server}: non-JSON blink response") from exc
        elapsed = (time.perf_counter() - started) * 1000
        _validate(data, questions)
        return Decision(data["answers"], data["usage"], round(elapsed, 2), self.server)

    async def close(self) -> None:
        await self.http.aclose()


class FanoutClient:
    """Send the three independent questions to distinct servers concurrently."""

    def __init__(self, clients: list[BlinkClient]):
        if len(clients) < 3 or len({c.server for c in clients}) < 3:
            raise ValueError("--fanout requires three distinct servers for this model")
        self.clients = clients
        self.server = ",".join(c.server for c in clients[:3])

    async def ask(self, state: dict, questions: dict, **kwargs) -> Decision:
        if len(questions) == 1:
            return await self.clients[0].ask(state, questions, **kwargs)
        started = time.perf_counter()
        results = await asyncio.gather(*(
            self.clients[i].ask(state, {key: q}, **kwargs)
            for i, (key, q) in enumerate(questions.items())
        ))
        usage = {key: sum(r.usage.get(key, 0) for r in results)
                 for key in ("input_tokens", "visual_tokens")}
        return Decision(
            answers={key: r.answers[key] for key, r in zip(questions, results)},
            usage=usage, wall_ms=round((time.perf_counter() - started) * 1000, 2),
            server=",".join(r.server for r in results),
            per_question_ms={key: r.wall_ms for key, r in zip(questions, results)},
        )


class MockClient:
    """A reproducible stand-in: reads the oracle out of band, never the screenshot."""

    def __init__(self, seed: int = 1, noise: float = 0):
        if not 0 <= noise <= 1:
            raise ValueError("Mock noise must be between 0 and 1")
        self.rng = random.Random(seed)
        self.noise = noise
        self.server = "mock"

    async def ask(self, state: dict, questions: dict, *,
                  expected: dict | None = None, oracle_choices: dict | None = None) -> Decision:
        if expected is None:
            raise ValueError("The CPU mock requires an out-of-band oracle")
        oracle_choices = oracle_choices or {}
        answers = {}
        for key, question in questions.items():
            if question["type"] == "noul":
                truth = bool(expected["done"] if key == "done" else expected["risky"])
                picked = not truth if self.rng.random() < self.noise else truth
                p = 0.9 if picked else 0.1
                answers[key] = {"type": "noul", "noul": p,
                                "probabilities": {"yes": p, "no": 1 - p}}
            else:
                options = list(question["criteria"])
                if not options:
                    raise ValueError(f"Mock question {key!r} has no options")
                correct = oracle_choices.get(key)
                if key == "element" and expected.get("done") and correct is None:
                    correct = options[0]  # the element head is immaterial when the done gate wins
                if correct not in options:
                    raise ValueError(f"Mock oracle choice {key}={correct!r} is not in {options}")
                alternatives = [option for option in options if option != correct]
                picked = (self.rng.choice(alternatives)
                          if alternatives and self.rng.random() < self.noise else correct)
                p = 0.9 if alternatives else 1.0
                probs = {option: (p if option == picked else (1 - p) / len(alternatives))
                         for option in options}
                answers[key] = {"type": "choice", "choice": picked, "probabilities": probs}
        _validate({"answers": answers, "usage": {"input_tokens": 0, "visual_tokens": 0}}, questions)
        return Decision(answers, {"input_tokens": 0, "visual_tokens": 0}, 0.0, self.server)
