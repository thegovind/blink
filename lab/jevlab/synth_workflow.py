"""Workflow-document synthetic data for blink typed decisions.

Each seed renders one realistic multi-question document and emits one training
row per typed question:

  python -m jevlab.synth_workflow --out-dir ../data/workflow

The documents are original synthetic workflows with code-computed answers.  The
state alternates between JSON and prose worksheets, includes long policy text
with distractors, and keeps enough explicit fields in the rendered state for the
answers to be recomputed without using external data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

FAMILIES = (
    "security_alert",
    "invoice_ap",
    "support_refund",
    "agent_trace",
    "hr_expense",
    "shipping_claim",
)

NAMES = "Mira Theo Sana Ivo Lina Omar Priya Niko Hana Remy Vale Jules Tessa Corin Asha Milo Kira Dev Arden".split()
SURNAMES = "Voss Keene Rinaldi Soto Mercer Imani Fen Park Lin Ortega Shah Novak Bell Wynn Ivers Quinn".split()
COMPANY_BITS = "Northwind Ashbury Beacon Cascade Delta Ember Finch Grove Harbor Ion Juniper Kestrel Lumen Meridian Nova Orchard Prairie Quill Rowan Summit Terra Umber Vireo Willow Xenon Yarrow Zenith".split()


def _seed_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:16], "big")


def _rng(seed: int | str, salt: str = "") -> random.Random:
    return random.Random(_seed_int(f"workflow:{seed}:{salt}"))


def _person(rng: random.Random) -> str:
    return f"{rng.choice(NAMES)} {rng.choice(SURNAMES)}"


def _company(rng: random.Random) -> str:
    return f"{rng.choice(COMPANY_BITS)} {rng.choice(['Analytics', 'Foods', 'Logistics', 'Studios', 'Health', 'Retail', 'Systems'])}"


def _money(x: Decimal | int | str) -> str:
    d = Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{d:.2f}"


def _d(d: date) -> str:
    return d.isoformat()


def _dt(d: datetime) -> str:
    return d.replace(second=0, microsecond=0).isoformat(timespec="minutes")


def parse_date(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(str(s))


def business_days_after(start: date, n: int, holidays: set[date] | None = None) -> date:
    holidays = holidays or set()
    cur = start
    left = n
    while left:
        cur += timedelta(days=1)
        if cur.weekday() < 5 and cur not in holidays:
            left -= 1
    return cur


def business_days_between(start: date, end: date, holidays: set[date] | None = None) -> int:
    holidays = holidays or set()
    cur = start
    days = 0
    step = 1 if end >= start else -1
    while cur != end:
        cur += timedelta(days=step)
        if cur.weekday() < 5 and cur not in holidays:
            days += step
    return days


def days_between(a: str | date, b: str | date) -> int:
    da = parse_date(a) if isinstance(a, str) else a
    db = parse_date(b) if isinstance(b, str) else b
    return (db - da).days


def hours_between(a: str | datetime, b: str | datetime) -> float:
    da = parse_dt(a) if isinstance(a, str) else a
    db = parse_dt(b) if isinstance(b, str) else b
    return (db - da).total_seconds() / 3600


def _choice(instr: str, criteria: dict[str, str], gold: str, spec: dict[str, Any], target: dict[str, float] | None = None) -> dict[str, Any]:
    return {"type": "choice", "instructions": instr, "criteria": criteria, "_gold": gold, "_target": target, "_spec": spec}


def _noul(instr: str, yes: str, no: str, gold: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {"type": "noul", "instructions": instr, "criteria": {"true": yes, "false": no}, "_gold": gold, "_target": None, "_spec": spec}


def _score(instr: str, levels: list[str], gold: int, spec: dict[str, Any]) -> dict[str, Any]:
    assert 2 <= len(levels) <= 10
    return {"type": "score", "instructions": instr, "criteria": levels, "_gold": "computed", "_target": None, "_spec": spec}


def _maybe_shuffle_criteria(rng: random.Random, criteria: dict[str, str], gold: str, target: dict[str, float] | None = None) -> tuple[dict[str, str], str, dict[str, float] | None]:
    items = list(criteria.items())
    rng.shuffle(items)
    out = dict(items)
    if target is None:
        return out, gold, None
    return out, gold, {k: target[k] for k in out}


def _long_distractor(rng: random.Random, family: str) -> str:
    fragments = [
        "The quarterly dashboard memo remains non-controlling unless a policy section below explicitly adopts it.",
        "Pilot program notes are included for audit context but do not change the live workflow for this case.",
        "Legacy color codes may appear in screenshots; reviewers must use dates, amounts, statuses, and named overrides instead.",
        "Manager comments are useful for routing but cannot create an approval when the policy requires a recorded field.",
        "If a table row and a narrative summary conflict, the row marked current controls over the summary.",
        "Training examples in the margin are intentionally stale and are not case evidence.",
        "The operations team keeps weekend staffing notes here, but weekend staffing does not alter business-day calculations.",
        "Regional nicknames in chat excerpts do not identify the legal entity unless the account or asset table says so.",
        "A green banner in the source system means the form was saved; it does not mean the underlying request is eligible.",
        "Deprecated thresholds from the 2024 handbook are reproduced for comparison and are superseded by the active policy.",
    ]
    rng.shuffle(fragments)
    checklist = (
        "Review checklist: verify the active policy version, read the explicit case fields, compute date and amount thresholds directly, "
        "then ignore any narrative that lacks a matching field. The checklist is repeated in several departments so that reviewers do not "
        "confuse saved-form status, old handbook notes, chat summaries, or dashboard colors with evidence. It is not an extra rule and it "
        "does not override the numbered policy. Before recording an answer, reviewers should confirm whether the question asks for the main "
        "workflow disposition, a date-window fact, a numeric threshold, a sampled historical distribution, or an explicitly missing field."
    )
    return f"Non-controlling context for {family.replace('_', ' ')}:\n" + "\n".join(f"- {x}" for x in fragments) + "\n" + checklist


def _long_policy_sections(rng: random.Random, family: str) -> list[dict[str, str]]:
    active = date(2026, 1, 1)
    prior = date(2025, 4, 1)
    clauses = [
        ("1", "Scope and controlling evidence", "The current packet controls over screenshots, summaries, examples, and copied signatures. A missing field is unknown, not false."),
        ("2", "Version selection", f"Use clauses effective on or after {active.isoformat()} for 2026 cases. The set effective {prior.isoformat()} is audit-only."),
        ("3", "Date calculations", "Calendar windows count every day after the anchor date. Business-day windows skip weekends only when a clause explicitly says business days."),
        ("4", "Amount calculations", "Money thresholds use the rendered numeric amount after cent rounding. Historical percentage tolerances are superseded."),
        ("5", "Overrides", "Fraud, safety, privacy, duplicate, and director-approved exception clauses override ordinary routing only when the named field is present and true."),
        ("6", "Insufficient evidence", "If a required field is absent, choose can't tell when that option is offered instead of guessing from notes or history."),
        ("7", "Sampling questions", "Historical-count questions use a uniform random draw from the stated counts. Anecdotes and stale charts are not part of the draw."),
        ("8", "Tie handling", "When an exact computed distribution ties, keep the first option in the policy's stated key order as deterministic training gold."),
        ("9", "Superseded tolerances", "Any tolerance printed in an archive appendix is superseded unless the active policy repeats it verbatim."),
        ("10", "Manual notes", "A request for an exception in a message is not an exception unless the controlling approval field is present."),
    ]
    return [{"clause": n, "heading": h, "text": t} for n, h, t in clauses]


def _long_history(rng: random.Random, family: str, fields: dict[str, Any]) -> list[dict[str, str]]:
    visible = {k: v for k, v in fields.items() if v is not None}
    keys = list(visible) or ["case"]
    rng.shuffle(keys)
    actors = ["reviewer", "system", "manager", "auditor", "requester", "workflow-bot", "ops-desk"]
    base = datetime(2026, rng.randint(1, 9), rng.randint(1, 20), 8, 0)
    events = []
    for i in range(18):
        k = keys[i % len(keys)]
        v = visible.get(k, "unavailable")
        events.append({
            "time": _dt(base + timedelta(hours=i * rng.choice([2, 3, 5]), minutes=rng.choice([0, 7, 19, 31, 43]))),
            "actor": rng.choice(actors),
            "entry": f"Logged {k} as {v!r}; no controlling change unless the active policy references this field.",
            "control": "current" if i % 5 in {0, 3} else "context",
        })
    return events


def _long_messages(rng: random.Random, family: str) -> list[str]:
    pool = [
        "Can we use the old quick-close path here? It would be faster, but I do not see it in the current policy.",
        "The dashboard looks green; please confirm whether that means eligibility or merely that the form saved.",
        "A prior quarter example had a different threshold. I am leaving it in the thread only so audit can compare versions.",
        "The account owner asked for an exception, but the approval field is the only source that counts for this workflow.",
        "If the active clause and the historical appendix conflict, the active clause wins even when the appendix is more specific.",
        "Please do not infer missing dates from neighboring records; this packet has to stand on its own.",
        "The sampling table is for random-draw questions only and should not alter the main disposition.",
        "I added raw log lines because they are realistic distractors; apply only the named statuses and amounts.",
        "The old handbook used a percentage tolerance. The current handbook uses the explicit threshold stated above.",
        "A later manual review can override this packet, but no later override is included in the provided evidence.",
    ]
    rng.shuffle(pool)
    return [f"{family.replace('_', ' ').title()} message {i + 1}: {text}" for i, text in enumerate(pool)]


def _long_appendix(rng: random.Random, family: str, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "effective_versions": [
            {"version": "2024.4-retired", "effective": "2024-04-01", "status": "superseded", "note": "Retained for audit history only."},
            {"version": "2025.2-archive", "effective": "2025-04-01", "status": "superseded", "note": "Do not apply to the current case."},
            {"version": "2026.1-active", "effective": "2026-01-01", "status": "active", "note": "Use for the rendered case fields."},
        ],
        "numbered_clauses": _long_policy_sections(rng, family),
        "history_log": _long_history(rng, family, fields),
        "messages": _long_messages(rng, family),
        "irrelevant_tables": [
            {"table": "regional staffing", "rows": [{"region": r, "weekday_capacity": rng.randint(3, 11), "weekend_capacity": rng.randint(0, 4)} for r in ["us", "ca", "eu", "apac"]]},
            {"table": "retired thresholds", "rows": [{"year": y, "threshold_note": rng.choice(["percentage tolerance", "manual review at lower severity", "weekend grace period"])} for y in [2023, 2024, 2025]]},
        ],
    }


def _state_from_fields(rng: random.Random, family: str, title: str, policy: str, fields: dict[str, Any], prose: bool, long_doc: bool = False) -> dict[str, Any] | str:
    visible_fields = {k: v for k, v in fields.items() if v is not None}
    distractor = _long_distractor(rng, family)
    appendix = _long_appendix(rng, family, fields) if long_doc else None
    if not prose:
        state = {
            "document_title": title,
            "family": family,
            "case_fields": visible_fields,
            "policy": policy,
            "distractor_context": distractor,
            "reviewer_note": "Use only case_fields and the active policy. Missing fields are genuinely unavailable.",
        }
        if appendix:
            state["long_policy_packet"] = appendix
        return state
    lines = [
        title,
        "",
        "ACTIVE POLICY",
        policy,
    ]
    if appendix:
        lines += ["", "MULTI-SECTION POLICY PACKET", "Version table:"]
        for row in appendix["effective_versions"]:
            lines.append(f"- {row['version']} effective {row['effective']} status {row['status']}: {row['note']}")
        lines.append("Numbered clauses:")
        for row in appendix["numbered_clauses"]:
            lines.append(f"{row['clause']}. {row['heading']}: {row['text']}")
        lines.append("Long case history:")
        for ev in appendix["history_log"]:
            lines.append(f"- {ev['time']} [{ev['actor']}] {ev['entry']} ({ev['control']})")
        lines.append("Messages and plausible distractors:")
        lines.extend(f"- {msg}" for msg in appendix["messages"])
        lines.append("Irrelevant tables:")
        for table in appendix["irrelevant_tables"]:
            lines.append(json.dumps(table, ensure_ascii=False, sort_keys=True))
    lines += [
        "",
        "CASE FIELDS",
    ]
    for k, v in visible_fields.items():
        lines.append(f"{k}: {json.dumps(v, ensure_ascii=False, sort_keys=True)}")
    lines += [
        "END CASE FIELDS",
        "",
        distractor,
        "",
        "Reviewer note: fields absent from CASE FIELDS were not present in the source packet.",
    ]
    return "\n".join(lines)


def parse_state_fields(state: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(state, dict):
        return dict(state.get("case_fields", {}))
    text = str(state)
    m = re.search(r"CASE FIELDS\n(.*?)\nEND CASE FIELDS", text, flags=re.S)
    if not m:
        raise ValueError("state has no CASE FIELDS block")
    out: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = json.loads(v.strip())
    return out


def _ct(criteria: dict[str, str]) -> dict[str, str]:
    out = dict(criteria)
    out["cant_tell"] = "Can't tell from the provided document"
    return out


def _score_band(value: float, cuts: list[float]) -> int:
    idx = 0
    for c in cuts:
        if value >= c:
            idx += 1
    return idx


def solve_spec(fields: dict[str, Any], spec: dict[str, Any]) -> tuple[str, dict[str, float] | None]:
    kind = spec["kind"]
    if kind == "computed_eq":
        inner_gold, _ = solve_spec(fields, spec["inner"])
        return ("yes" if inner_gold == spec["value"] else "no"), None
    req = spec.get("required", [])
    if any(k not in fields for k in req):
        return "cant_tell", None
    if kind == "field_eq":
        return ("yes" if fields[spec["field"]] == spec["value"] else "no"), None
    if kind == "choice_value":
        return str(fields[spec["field"]]), None
    if kind == "score_from_field":
        return str(_score_band(float(fields[spec["field"]]), spec["cuts"])), None
    if kind == "prob_counts":
        counts = fields[spec["counts_field"]]
        keys = spec["keys"]
        total = sum(int(counts.get(k, 0)) for k in keys)
        dist = {k: (int(counts.get(k, 0)) / total if total else 0.0) for k in keys}
        gold = max(keys, key=lambda k: (dist[k], -keys.index(k)))
        return gold, dist
    if kind == "date_after":
        return ("yes" if parse_date(fields[spec["after"]]) > parse_date(fields[spec["before"]]) else "no"), None
    if kind == "days_le":
        return ("yes" if days_between(fields[spec["start"]], fields[spec["end"]]) <= int(spec["limit"]) else "no"), None
    if kind == "money_gt":
        return ("yes" if Decimal(str(fields[spec["field"]])) > Decimal(str(spec["threshold"])) else "no"), None
    if kind == "int_ge":
        return ("yes" if int(fields[spec["field"]]) >= int(spec["threshold"]) else "no"), None

    fam = spec["family"]
    if fam == "security_alert":
        return _solve_security(fields, spec), None
    if fam == "invoice_ap":
        return _solve_invoice(fields, spec), None
    if fam == "support_refund":
        return _solve_support(fields, spec), None
    if fam == "agent_trace":
        return _solve_agent(fields, spec), None
    if fam == "hr_expense":
        return _solve_hr(fields, spec), None
    if fam == "shipping_claim":
        return _solve_shipping(fields, spec), None
    raise AssertionError(kind)


def _solve_security(f: dict[str, Any], spec: dict[str, Any]) -> str:
    patch_age = days_between(f["last_patch_date"], f["detected_date"])
    contains_regulated = bool(f["regulated_data"])
    high_asset = f["asset_tier"] in {"crown_jewel", "production"}
    severity_score = int(f["cvss"]) + (2 if contains_regulated else 0) + (1 if high_asset else 0) + (2 if bool(f["internet_exposed"]) else 0) + int(f["prior_alerts"])
    if spec["kind"] == "security_disposition":
        if f["allowlist_active"] and f["signature_match"] and not contains_regulated:
            return "suppress"
        if f["asset_tier"] == "crown_jewel" and contains_regulated:
            return "critical_escalation"
        if int(f["cvss"]) >= 8 and bool(f["internet_exposed"]):
            return "contain_now"
        if patch_age > 30 or int(f["prior_alerts"]) >= 3:
            return "analyst_review"
        return "monitor"
    if spec["kind"] == "security_sla":
        hours = hours_between(f["detected_at"], f["ack_at"])
        limit = 2 if _solve_security(f, {"kind": "security_disposition"}) in {"critical_escalation", "contain_now"} else 8
        return "yes" if hours <= limit else "no"
    if spec["kind"] == "security_patch_late":
        return "yes" if patch_age > 30 else "no"
    if spec["kind"] == "security_score":
        return str(_score_band(severity_score, [4, 7, 10, 13]))
    if spec["kind"] == "security_privacy":
        return "yes_privacy" if contains_regulated and f["asset_region"] not in {"us", "ca"} else "no_privacy"
    raise AssertionError(spec["kind"])


def _solve_invoice(f: dict[str, Any], spec: dict[str, Any]) -> str:
    invoice = Decimal(str(f["invoice_total"]))
    po = Decimal(str(f["po_total"]))
    received = Decimal(str(f["received_total"]))
    variance = abs(invoice - received)
    duplicate = bool(f["duplicate_invoice"])
    over_po = invoice > po
    due = parse_date(f["invoice_date"]) + timedelta(days=int(f["terms_days"]))
    if spec["kind"] == "invoice_disposition":
        if duplicate:
            return "reject_duplicate"
        if bool(f["vendor_on_hold"]):
            return "hold_vendor"
        if f["approval_missing"] and invoice > Decimal("1000"):
            return "need_approval"
        if over_po or variance > Decimal("25.00"):
            return "three_way_review"
        return "pay"
    if spec["kind"] == "invoice_due_late":
        return "yes" if parse_date(f["scheduled_pay_date"]) > due else "no"
    if spec["kind"] == "invoice_within_tolerance":
        return "yes" if variance <= Decimal("25.00") and not over_po else "no"
    if spec["kind"] == "invoice_risk_score":
        risk = (3 if duplicate else 0) + (2 if bool(f["vendor_on_hold"]) else 0) + (2 if over_po else 0) + (2 if variance > Decimal("25.00") else 0) + (1 if f.get("approval_missing", False) else 0)
        return str(_score_band(risk, [2, 4, 6, 8]))
    if spec["kind"] == "invoice_discount":
        eligible_date = parse_date(f["invoice_date"]) + timedelta(days=int(f["discount_days"]))
        return "yes" if parse_date(f["scheduled_pay_date"]) <= eligible_date and not duplicate and not bool(f["vendor_on_hold"]) else "no"
    raise AssertionError(spec["kind"])


def _solve_support(f: dict[str, Any], spec: dict[str, Any]) -> str:
    outage_hours = hours_between(f["opened_at"], f["resolved_at"])
    since_purchase = days_between(f["purchase_date"], f["opened_date"])
    vip = f["plan"] in {"enterprise", "premium"}
    if spec["kind"] == "support_disposition":
        if bool(f["abuse_flag"]):
            return "deny_abuse"
        if outage_hours >= 24 and vip:
            return "full_refund"
        if outage_hours >= 8 or (since_purchase <= 30 and f["defect_confirmed"]):
            return "partial_credit"
        if business_days_between(parse_date(f["opened_date"]), parse_date(f["first_response_date"])) > (1 if vip else 2):
            return "sla_credit"
        return "no_refund"
    if spec["kind"] == "support_sla_met":
        return "yes" if business_days_between(parse_date(f["opened_date"]), parse_date(f["first_response_date"])) <= (1 if vip else 2) else "no"
    if spec["kind"] == "support_window":
        return "yes" if since_purchase <= 30 else "no"
    if spec["kind"] == "support_score":
        risk = (3 if outage_hours >= 24 else 1 if outage_hours >= 8 else 0) + (2 if vip else 0) + (2 if f.get("defect_confirmed", False) else 0) + (2 if f["abuse_flag"] else 0)
        return str(_score_band(risk, [2, 4, 6, 8]))
    raise AssertionError(spec["kind"])


def _solve_agent(f: dict[str, Any], spec: dict[str, Any]) -> str:
    failed = [s for s in f["steps"] if s["status"] != "ok"]
    violated = [s for s in f["steps"] if s.get("rule_violation")]
    final_ok = f.get("final_answer") == f["goal_expected"] and not failed and not violated
    if spec["kind"] == "agent_outcome":
        if violated:
            return "rule_violation"
        if failed:
            return "tool_failure"
        if final_ok:
            return "succeeded"
        return "wrong_answer"
    if spec["kind"] == "agent_failed_step":
        return failed[0]["name"] if failed else "none"
    if spec["kind"] == "agent_violation":
        return "yes" if bool(violated) else "no"
    if spec["kind"] == "agent_latency":
        return "yes" if sum(int(s["ms"]) for s in f["steps"]) <= int(f["latency_budget_ms"]) else "no"
    if spec["kind"] == "agent_score":
        risk = (4 if violated else 0) + (3 if failed else 0) + (2 if not final_ok else 0) + (1 if sum(int(s["ms"]) for s in f["steps"]) > int(f["latency_budget_ms"]) else 0)
        return str(_score_band(risk, [2, 4, 6, 8]))
    raise AssertionError(spec["kind"])


def _solve_hr(f: dict[str, Any], spec: dict[str, Any]) -> str:
    amount = Decimal(str(f["amount"]))
    submitted_days = days_between(f["expense_date"], f["submitted_date"])
    has_receipt = bool(f["receipt_present"])
    medical = f["category"] == "medical_leave"
    if spec["kind"] == "hr_disposition":
        if bool(f["blackout_exception"]) and bool(f["director_approved"]):
            return "approve_exception"
        if submitted_days > 30 and not medical:
            return "deny_late"
        if amount > Decimal("500") and not bool(f["manager_approved"]):
            return "need_manager"
        if not has_receipt and amount > Decimal("75"):
            return "need_receipt"
        return "approve"
    if spec["kind"] == "hr_late":
        return "yes" if submitted_days > 30 and not medical else "no"
    if spec["kind"] == "hr_receipt_required":
        return "yes" if amount > Decimal("75") else "no"
    if spec["kind"] == "hr_score":
        risk = (2 if submitted_days > 30 else 0) + (2 if amount > Decimal("500") else 0) + (2 if not has_receipt and amount > Decimal("75") else 0) + (3 if f["blackout_exception"] and not f["director_approved"] else 0)
        return str(_score_band(risk, [2, 4, 6, 8]))
    raise AssertionError(spec["kind"])


def _solve_shipping(f: dict[str, Any], spec: dict[str, Any]) -> str:
    delivery_late = parse_date(f["delivered_date"]) > parse_date(f["promised_date"])
    claim_days = days_between(f["delivered_date"], f["claim_date"])
    value = Decimal(str(f["declared_value"]))
    if spec["kind"] == "shipping_disposition":
        if bool(f["fraud_hold"]):
            return "fraud_review"
        if f["damage_code"] == "carrier_fault" and claim_days <= 14:
            return "pay_claim"
        if delivery_late and value <= Decimal("500") and claim_days <= 30:
            return "service_credit"
        if claim_days > 30:
            return "deny_late"
        return "inspect"
    if spec["kind"] == "shipping_timely":
        return "yes" if claim_days <= 30 else "no"
    if spec["kind"] == "shipping_late_delivery":
        return "yes" if delivery_late else "no"
    if spec["kind"] == "shipping_score":
        risk = (3 if f["fraud_hold"] else 0) + (2 if claim_days > 30 else 0) + (2 if value > Decimal("1000") else 0) + (2 if f.get("damage_code") == "unknown" else 0)
        return str(_score_band(risk, [2, 4, 6, 8]))
    raise AssertionError(spec["kind"])


def _finalize_questions(rng: random.Random, family: str, fields: dict[str, Any], qs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible = {k: v for k, v in fields.items() if v is not None}
    out = []
    for q in qs:
        spec = q.pop("_spec")
        spec.setdefault("family", family)
        gold, target = solve_spec(visible, spec)
        if gold == "cant_tell" and q["type"] == "choice":
            q["criteria"] = _ct(q["criteria"])
        if q["type"] == "choice":
            q["criteria"], gold, target = _maybe_shuffle_criteria(rng, q["criteria"], gold, target)
        assert q["_gold"] in {gold, "computed"}
        if q["_target"] is not None:
            assert target is not None
        del q["_gold"]
        q["_target"] = target
        q["_spec"] = spec
        out.append(q)
    return out


def gen_security(seed: int, prose: bool, long_doc: bool = False) -> dict[str, Any]:
    rng = _rng(seed, "security")
    today = date(2026, rng.randint(1, 9), rng.randint(1, 24))
    detected_at = datetime.combine(today, datetime.min.time()).replace(hour=rng.randint(6, 16), minute=rng.choice([0, 15, 30, 45]))
    ack_at = detected_at + timedelta(hours=rng.choice([1, 2, 4, 7, 10, 16]))
    fields = {
        "asset_id": f"ast-{rng.randint(1000, 9999)}",
        "asset_tier": rng.choice(["crown_jewel", "production", "internal", "sandbox"]),
        "asset_region": rng.choice(["us", "ca", "eu", "apac"]),
        "regulated_data": rng.choice([True, False, False]),
        "internet_exposed": rng.choice([True, False]),
        "cvss": rng.choice([4, 5, 6, 7, 8, 9]),
        "prior_alerts": rng.randint(0, 4),
        "allowlist_active": rng.choice([True, False, False]),
        "signature_match": rng.choice([True, False]),
        "last_patch_date": _d(today - timedelta(days=rng.choice([7, 14, 28, 31, 45, 70]))),
        "detected_date": _d(today),
        "detected_at": _dt(detected_at),
        "ack_at": _dt(ack_at),
        "sample_counts": {"critical_escalation": rng.randint(1, 5), "contain_now": rng.randint(1, 5), "analyst_review": rng.randint(1, 5), "monitor": rng.randint(1, 5)},
    }
    if rng.random() < 0.16:
        fields["asset_region"] = None
    policy = (
        "Security escalation policy S-17. First, suppress only when the allowlist is active, the signature matches, and the asset does not contain regulated data. "
        "Second, crown-jewel assets with regulated data require critical escalation regardless of any lower rule. Third, CVSS 8 or higher on an internet-exposed asset requires immediate containment. "
        "Fourth, alerts with a patch age over 30 calendar days or at least 3 prior alerts go to analyst review. Otherwise monitor. Critical escalation and containment must be acknowledged within 2 hours; other alerts within 8 hours. "
        "Privacy counsel is required for regulated data outside the US and Canada. A prior 2024 rule used a CVSS 7 threshold but is retired."
    )
    crit = {"critical_escalation": "Escalate as critical", "contain_now": "Contain immediately", "analyst_review": "Send to analyst review", "monitor": "Monitor only", "suppress": "Suppress as allowed"}
    qs = [
        _choice("Apply the active security policy. What disposition should be entered?", crit, "computed", {"kind": "security_disposition"}),
        _noul("Was the acknowledgement within the applicable SLA window?", "Acknowledged within the applicable window", "Acknowledgement missed the applicable window", "computed", {"kind": "security_sla"}),
        _noul("Is the patch age over the 30-day escalation threshold?", "Patch age is over 30 calendar days", "Patch age is not over 30 calendar days", "computed", {"kind": "security_patch_late"}),
        _score("Score security urgency from the stated fields.", ["minimal", "low", "moderate", "high", "severe"], 0, {"kind": "security_score"}),
        _choice("Can privacy counsel routing be determined from this document?", {"yes_privacy": "Privacy counsel required", "no_privacy": "Privacy counsel not required"}, "computed", {"kind": "security_privacy", "required": ["asset_region"]}),
        _choice("If one historical alert in the sample is drawn uniformly, which disposition is most likely?", {"critical_escalation": "Critical escalation", "contain_now": "Contain now", "analyst_review": "Analyst review", "monitor": "Monitor"}, "computed", {"kind": "prob_counts", "counts_field": "sample_counts", "keys": ["critical_escalation", "contain_now", "analyst_review", "monitor"]}, {}),
    ]
    qs += [
        _noul("Does the asset contain regulated data?", "Regulated data is present", "Regulated data is not present", "computed", {"kind": "field_eq", "field": "regulated_data", "value": True}),
        _noul("Is the asset internet exposed?", "The asset is internet exposed", "The asset is not internet exposed", "computed", {"kind": "field_eq", "field": "internet_exposed", "value": True}),
        _choice("Which asset tier is stated in the inventory?", {k: k.replace("_", " ") for k in ["crown_jewel", "production", "internal", "sandbox"]}, "computed", {"kind": "choice_value", "field": "asset_tier"}),
        _choice("Which region is stated for the asset?", {k: k.upper() for k in ["us", "ca", "eu", "apac"]}, "computed", {"kind": "choice_value", "field": "asset_region", "required": ["asset_region"]}),
    ]
    return {"family": "security_alert", "state": _state_from_fields(rng, "security_alert", f"Security alert packet {fields['asset_id']}", policy, fields, prose, long_doc), "fields": fields, "questions": _finalize_questions(rng, "security_alert", fields, qs)}


def gen_invoice(seed: int, prose: bool, long_doc: bool = False) -> dict[str, Any]:
    rng = _rng(seed, "invoice")
    inv_date = date(2026, rng.randint(1, 9), rng.randint(1, 20))
    po = Decimal(rng.choice(["850.00", "1200.00", "2450.00", "5100.00"]))
    received = po + Decimal(str(rng.choice([-35, -10, 0, 12, 40])))
    invoice = received + Decimal(str(rng.choice([-30, -12, 0, 18, 60])))
    terms = rng.choice([15, 30, 45])
    fields = {
        "vendor": _company(rng),
        "invoice_id": f"INV-{rng.randint(10000, 99999)}",
        "po_total": _money(po),
        "received_total": _money(received),
        "invoice_total": _money(invoice),
        "invoice_date": _d(inv_date),
        "terms_days": terms,
        "scheduled_pay_date": _d(inv_date + timedelta(days=terms + rng.choice([-8, -2, 0, 3, 10]))),
        "discount_days": rng.choice([10, 12]),
        "duplicate_invoice": rng.choice([True, False, False, False]),
        "vendor_on_hold": rng.choice([True, False, False, False]),
        "approval_missing": rng.choice([True, False]),
        "audit_counts": {"pay": rng.randint(1, 6), "three_way_review": rng.randint(1, 6), "need_approval": rng.randint(1, 6), "hold_vendor": rng.randint(1, 6), "reject_duplicate": rng.randint(1, 6)},
    }
    if rng.random() < 0.14:
        fields["approval_missing"] = None
    policy = (
        "Accounts payable policy AP-42. Reject exact duplicate invoices before all other checks. Vendors on finance hold cannot be paid and must be held. "
        "Invoices over $1,000 with missing approval require approval before payment. If invoice total exceeds the PO or differs from receiving by more than $25.00, use three-way review. "
        "Otherwise pay under the PO terms. Payment is late after invoice date plus terms days. Early-payment discount applies only if payment is scheduled by the discount day and the invoice is not duplicate or held. "
        "The receiving team's informal 5 percent tolerance note is obsolete; use the $25.00 rule."
    )
    crit = {"pay": "Pay invoice", "three_way_review": "Three-way review", "need_approval": "Need approval", "hold_vendor": "Hold for vendor status", "reject_duplicate": "Reject duplicate"}
    qs = [
        _choice("Apply AP-42. What should accounts payable do with this invoice?", crit, "computed", {"kind": "invoice_disposition", "required": ["approval_missing"]}),
        _noul("Is the scheduled payment late under the stated terms?", "Payment is late", "Payment is not late", "computed", {"kind": "invoice_due_late"}),
        _noul("Is the invoice within PO/receiving tolerance?", "Within tolerance", "Outside tolerance", "computed", {"kind": "invoice_within_tolerance"}),
        _noul("Does the early-payment discount apply?", "Discount applies", "Discount does not apply", "computed", {"kind": "invoice_discount"}),
        _score("Score the AP exception risk.", ["clean", "minor", "moderate", "high", "blocker"], 0, {"kind": "invoice_risk_score"}),
        _choice("If one recent AP audit case is drawn uniformly, which outcome is most likely?", crit, "computed", {"kind": "prob_counts", "counts_field": "audit_counts", "keys": list(crit)}, {}),
        _noul("Is this invoice marked as a duplicate?", "Duplicate flag is set", "Duplicate flag is not set", "computed", {"kind": "field_eq", "field": "duplicate_invoice", "value": True}),
        _noul("Is the vendor on finance hold?", "Vendor is on hold", "Vendor is not on hold", "computed", {"kind": "field_eq", "field": "vendor_on_hold", "value": True}),
    ]
    return {"family": "invoice_ap", "state": _state_from_fields(rng, "invoice_ap", f"AP review packet {fields['invoice_id']}", policy, fields, prose, long_doc), "fields": fields, "questions": _finalize_questions(rng, "invoice_ap", fields, qs)}


def gen_support(seed: int, prose: bool, long_doc: bool = False) -> dict[str, Any]:
    rng = _rng(seed, "support")
    opened = datetime(2026, rng.randint(1, 9), rng.randint(1, 22), rng.randint(7, 18), 0)
    resolved = opened + timedelta(hours=rng.choice([3, 7, 9, 14, 25, 36]))
    purchase = opened.date() - timedelta(days=rng.choice([5, 18, 31, 45, 90]))
    first = opened.date() + timedelta(days=rng.choice([0, 1, 2, 3, 4]))
    fields = {
        "account_id": f"acct-{rng.randint(1000, 9999)}",
        "plan": rng.choice(["basic", "standard", "premium", "enterprise"]),
        "opened_at": _dt(opened),
        "resolved_at": _dt(resolved),
        "opened_date": _d(opened.date()),
        "first_response_date": _d(first),
        "purchase_date": _d(purchase),
        "defect_confirmed": rng.choice([True, False]),
        "abuse_flag": rng.choice([True, False, False, False, False]),
        "refund_counts": {"full_refund": rng.randint(1, 6), "partial_credit": rng.randint(1, 6), "sla_credit": rng.randint(1, 6), "no_refund": rng.randint(1, 6)},
    }
    if rng.random() < 0.14:
        fields["defect_confirmed"] = None
    policy = (
        "Support refund policy R-8. Abuse or chargeback misuse denies refund before other benefits. Enterprise and premium accounts receive a full refund for outages of at least 24 hours. "
        "Any account receives partial credit for outages of at least 8 hours, or for a confirmed defect reported within 30 calendar days of purchase. "
        "First response SLA is one business day for premium/enterprise and two business days for other plans; missed SLA grants SLA credit unless a higher refund applies. Otherwise no refund. "
        "Marketing language about customer delight is not a refund rule."
    )
    crit = {"full_refund": "Full refund", "partial_credit": "Partial credit", "sla_credit": "SLA credit", "no_refund": "No refund", "deny_abuse": "Deny for abuse"}
    qs = [
        _choice("Apply support refund policy R-8. What resolution follows?", crit, "computed", {"kind": "support_disposition", "required": ["defect_confirmed"]}),
        _noul("Was the first response SLA met?", "First response met the SLA", "First response missed the SLA", "computed", {"kind": "support_sla_met"}),
        _noul("Was the ticket opened within 30 days of purchase?", "Within 30 calendar days", "Outside 30 calendar days", "computed", {"kind": "support_window"}),
        _score("Score refund/escalation severity.", ["none", "low", "moderate", "high", "severe"], 0, {"kind": "support_score"}),
        _choice("If one historical refund case is sampled uniformly, which outcome is most likely?", {"full_refund": "Full refund", "partial_credit": "Partial credit", "sla_credit": "SLA credit", "no_refund": "No refund"}, "computed", {"kind": "prob_counts", "counts_field": "refund_counts", "keys": ["full_refund", "partial_credit", "sla_credit", "no_refund"]}, {}),
        _choice("Which customer plan is stated?", {k: k.title() for k in ["basic", "standard", "premium", "enterprise"]}, "computed", {"kind": "choice_value", "field": "plan"}),
        _noul("Is the account flagged for abuse?", "Abuse flag is present", "No abuse flag is present", "computed", {"kind": "field_eq", "field": "abuse_flag", "value": True}),
    ]
    return {"family": "support_refund", "state": _state_from_fields(rng, "support_refund", f"Support ticket {fields['account_id']}", policy, fields, prose, long_doc), "fields": fields, "questions": _finalize_questions(rng, "support_refund", fields, qs)}


def gen_agent(seed: int, prose: bool, long_doc: bool = False) -> dict[str, Any]:
    rng = _rng(seed, "agent")
    names = ["fetch_record", "check_policy", "calculate_deadline", "write_result", "notify_user"]
    fail_name = rng.choice(names + ["none"])
    violation_name = rng.choice(names + ["none", "none"])
    steps = []
    for n in names:
        steps.append({"name": n, "status": "error" if n == fail_name else "ok", "ms": rng.randint(80, 900), "rule_violation": n == violation_name})
    expected = rng.choice(["approve", "deny", "escalate"])
    fields = {
        "goal": "decide a workflow case and send only the approved disposition",
        "goal_expected": expected,
        "final_answer": expected if rng.random() < 0.75 else rng.choice([x for x in ["approve", "deny", "escalate"] if x != expected]),
        "steps": steps,
        "latency_budget_ms": rng.choice([1800, 2200, 3000]),
        "trace_counts": {"succeeded": rng.randint(1, 7), "tool_failure": rng.randint(1, 7), "rule_violation": rng.randint(1, 7), "wrong_answer": rng.randint(1, 7)},
    }
    if rng.random() < 0.12:
        fields["final_answer"] = None
    policy = (
        "Agent trace policy A-3. A run succeeds only if every tool step is ok, no step violates a rule, and the final answer equals the expected goal result. "
        "Any recorded rule violation overrides tool success and is classified as rule_violation. If there is no violation but any tool status is not ok, classify as tool_failure. "
        "If tools were ok but the final answer differs from the expected result, classify as wrong_answer. Total latency must be within the per-trace budget. "
        "A retry suggestion in a tool log is not itself a failure unless the step status says error."
    )
    crit = {"succeeded": "Run succeeded", "tool_failure": "A tool failed", "rule_violation": "A rule was violated", "wrong_answer": "Final answer was wrong"}
    qs = [
        _choice("Classify the agent run under policy A-3.", crit, "computed", {"kind": "agent_outcome", "required": ["final_answer"]}),
        _choice("Which step failed first, if any?", {k: k.replace("_", " ") for k in names + ["none"]}, "computed", {"kind": "agent_failed_step"}),
        _noul("Did the agent violate a rule?", "At least one rule violation occurred", "No rule violation occurred", "computed", {"kind": "agent_violation"}),
        _noul("Was total tool latency within the budget?", "Latency is within budget", "Latency exceeds budget", "computed", {"kind": "agent_latency"}),
        _score("Score trace risk.", ["clean", "watch", "moderate", "high", "severe"], 0, {"kind": "agent_score"}),
        _choice("If one similar trace is drawn uniformly, which class is most likely?", crit, "computed", {"kind": "prob_counts", "counts_field": "trace_counts", "keys": list(crit)}, {}),
        _choice("What final answer is visible in the trace?", {"approve": "Approve", "deny": "Deny", "escalate": "Escalate"}, "computed", {"kind": "choice_value", "field": "final_answer", "required": ["final_answer"]}),
    ]
    return {"family": "agent_trace", "state": _state_from_fields(rng, "agent_trace", "Agent tool-use trace review", policy, fields, prose, long_doc), "fields": fields, "questions": _finalize_questions(rng, "agent_trace", fields, qs)}


def gen_hr(seed: int, prose: bool, long_doc: bool = False) -> dict[str, Any]:
    rng = _rng(seed, "hr")
    exp = date(2026, rng.randint(1, 9), rng.randint(1, 20))
    amount = Decimal(str(rng.choice([42, 84, 140, 515, 760, 1200])))
    fields = {
        "employee": _person(rng),
        "category": rng.choice(["travel_meal", "lodging", "equipment", "medical_leave"]),
        "amount": _money(amount),
        "expense_date": _d(exp),
        "submitted_date": _d(exp + timedelta(days=rng.choice([3, 12, 29, 31, 45]))),
        "receipt_present": rng.choice([True, False]),
        "manager_approved": rng.choice([True, False]),
        "director_approved": rng.choice([True, False]),
        "blackout_exception": rng.choice([True, False, False]),
        "request_counts": {"approve": rng.randint(1, 6), "need_receipt": rng.randint(1, 6), "need_manager": rng.randint(1, 6), "deny_late": rng.randint(1, 6), "approve_exception": rng.randint(1, 6)},
    }
    if rng.random() < 0.13:
        fields["manager_approved"] = None
    policy = (
        "HR expense and leave policy H-12. Director-approved blackout exceptions are approved even if normal blackout language would block them. "
        "Non-medical requests submitted more than 30 calendar days after the expense or leave date are denied as late. Amounts over $500 require manager approval. "
        "Receipts are required for amounts over $75. Medical leave uses the medical exception and is not denied solely for the 30-day expense window. "
        "A team lead's chat approval is not manager approval unless the manager_approved field is present."
    )
    crit = {"approve": "Approve", "need_receipt": "Need receipt", "need_manager": "Need manager approval", "deny_late": "Deny late", "approve_exception": "Approve exception"}
    qs = [
        _choice("Apply policy H-12. What disposition follows?", crit, "computed", {"kind": "hr_disposition", "required": ["manager_approved"]}),
        _noul("Is the request late under the 30-day rule?", "Late under the policy", "Not late under the policy", "computed", {"kind": "hr_late"}),
        _noul("Is a receipt required by amount?", "Receipt required", "Receipt not required by amount", "computed", {"kind": "hr_receipt_required"}),
        _score("Score HR exception risk.", ["clean", "low", "moderate", "high", "severe"], 0, {"kind": "hr_score"}),
        _choice("If one similar request is sampled uniformly, which outcome is most likely?", crit, "computed", {"kind": "prob_counts", "counts_field": "request_counts", "keys": list(crit)}, {}),
        _choice("Which category is stated?", {k: k.replace("_", " ") for k in ["travel_meal", "lodging", "equipment", "medical_leave"]}, "computed", {"kind": "choice_value", "field": "category"}),
        _noul("Is a blackout exception recorded?", "Blackout exception is recorded", "No blackout exception is recorded", "computed", {"kind": "field_eq", "field": "blackout_exception", "value": True}),
    ]
    return {"family": "hr_expense", "state": _state_from_fields(rng, "hr_expense", f"HR request for {fields['employee']}", policy, fields, prose, long_doc), "fields": fields, "questions": _finalize_questions(rng, "hr_expense", fields, qs)}


def gen_shipping(seed: int, prose: bool, long_doc: bool = False) -> dict[str, Any]:
    rng = _rng(seed, "shipping")
    ship = date(2026, rng.randint(1, 9), rng.randint(1, 18))
    promised = ship + timedelta(days=rng.choice([2, 3, 5, 7]))
    delivered = promised + timedelta(days=rng.choice([-1, 0, 1, 4, 9]))
    fields = {
        "claim_id": f"CLM-{rng.randint(10000, 99999)}",
        "shipped_date": _d(ship),
        "promised_date": _d(promised),
        "delivered_date": _d(delivered),
        "claim_date": _d(delivered + timedelta(days=rng.choice([1, 7, 14, 29, 35, 52]))),
        "declared_value": _money(Decimal(str(rng.choice([90, 240, 499, 800, 1500])))),
        "damage_code": rng.choice(["carrier_fault", "packaging_fault", "unknown", "none"]),
        "fraud_hold": rng.choice([True, False, False, False]),
        "claim_counts": {"pay_claim": rng.randint(1, 6), "service_credit": rng.randint(1, 6), "deny_late": rng.randint(1, 6), "inspect": rng.randint(1, 6), "fraud_review": rng.randint(1, 6)},
    }
    if rng.random() < 0.13:
        fields["damage_code"] = None
    policy = (
        "Shipping claims policy C-6. Fraud hold routes to fraud review before all other decisions. Carrier-fault damage reported within 14 calendar days is paid. "
        "Late delivery claims with declared value at or below $500 receive a service credit if filed within 30 calendar days after delivery. Claims filed after 30 days are denied late. "
        "Other claims go to inspection. A warehouse note saying 'probably carrier' is not enough unless damage_code is carrier_fault."
    )
    crit = {"pay_claim": "Pay claim", "service_credit": "Service credit", "deny_late": "Deny late", "inspect": "Inspect", "fraud_review": "Fraud review"}
    qs = [
        _choice("Apply shipping claims policy C-6. What disposition follows?", crit, "computed", {"kind": "shipping_disposition", "required": ["damage_code"]}),
        _noul("Was the claim filed within 30 days after delivery?", "Filed within 30 days", "Filed after 30 days", "computed", {"kind": "shipping_timely"}),
        _noul("Was delivery later than promised?", "Delivery was late", "Delivery was not late", "computed", {"kind": "shipping_late_delivery"}),
        _score("Score claim review risk.", ["clean", "low", "moderate", "high", "severe"], 0, {"kind": "shipping_score"}),
        _choice("If one recent claim is sampled uniformly, which disposition is most likely?", crit, "computed", {"kind": "prob_counts", "counts_field": "claim_counts", "keys": list(crit)}, {}),
        _choice("Which damage code is stated?", {k: k.replace("_", " ") for k in ["carrier_fault", "packaging_fault", "unknown", "none"]}, "computed", {"kind": "choice_value", "field": "damage_code", "required": ["damage_code"]}),
        _noul("Is the claim on fraud hold?", "Fraud hold is present", "No fraud hold is present", "computed", {"kind": "field_eq", "field": "fraud_hold", "value": True}),
    ]
    return {"family": "shipping_claim", "state": _state_from_fields(rng, "shipping_claim", f"Shipping claim {fields['claim_id']}", policy, fields, prose, long_doc), "fields": fields, "questions": _finalize_questions(rng, "shipping_claim", fields, qs)}


GENS = {
    "security_alert": gen_security,
    "invoice_ap": gen_invoice,
    "support_refund": gen_support,
    "agent_trace": gen_agent,
    "hr_expense": gen_hr,
    "shipping_claim": gen_shipping,
}


ENUM_OPTIONS = {
    "asset_tier": {"crown_jewel": "Crown jewel", "production": "Production", "internal": "Internal", "sandbox": "Sandbox"},
    "asset_region": {"us": "US", "ca": "Canada", "eu": "EU", "apac": "APAC"},
    "plan": {"basic": "Basic", "standard": "Standard", "premium": "Premium", "enterprise": "Enterprise"},
    "category": {"travel_meal": "Travel meal", "lodging": "Lodging", "equipment": "Equipment", "medical_leave": "Medical leave"},
    "damage_code": {"carrier_fault": "Carrier fault", "packaging_fault": "Packaging fault", "unknown": "Unknown", "none": "No damage"},
    "final_answer": {"approve": "Approve", "deny": "Deny", "escalate": "Escalate"},
}

MAIN_DECISIONS = {
    "security_alert": (
        {"critical_escalation": "Escalate as critical", "contain_now": "Contain immediately", "analyst_review": "Send to analyst review", "monitor": "Monitor only", "suppress": "Suppress as allowed"},
        {"kind": "security_disposition", "family": "security_alert"},
        "security disposition",
    ),
    "invoice_ap": (
        {"pay": "Pay invoice", "three_way_review": "Three-way review", "need_approval": "Need approval", "hold_vendor": "Hold for vendor status", "reject_duplicate": "Reject duplicate"},
        {"kind": "invoice_disposition", "family": "invoice_ap", "required": ["approval_missing"]},
        "AP disposition",
    ),
    "support_refund": (
        {"full_refund": "Full refund", "partial_credit": "Partial credit", "sla_credit": "SLA credit", "no_refund": "No refund", "deny_abuse": "Deny for abuse"},
        {"kind": "support_disposition", "family": "support_refund", "required": ["defect_confirmed"]},
        "support resolution",
    ),
    "agent_trace": (
        {"succeeded": "Run succeeded", "tool_failure": "A tool failed", "rule_violation": "A rule was violated", "wrong_answer": "Final answer was wrong"},
        {"kind": "agent_outcome", "family": "agent_trace", "required": ["final_answer"]},
        "agent run class",
    ),
    "hr_expense": (
        {"approve": "Approve", "need_receipt": "Need receipt", "need_manager": "Need manager approval", "deny_late": "Deny late", "approve_exception": "Approve exception"},
        {"kind": "hr_disposition", "family": "hr_expense", "required": ["manager_approved"]},
        "HR disposition",
    ),
    "shipping_claim": (
        {"pay_claim": "Pay claim", "service_credit": "Service credit", "deny_late": "Deny late", "inspect": "Inspect", "fraud_review": "Fraud review"},
        {"kind": "shipping_disposition", "family": "shipping_claim", "required": ["damage_code"]},
        "shipping disposition",
    ),
}


def _augment_document(seed: int, doc: dict[str, Any]) -> None:
    """Pad each document to ~20 typed questions with field, temporal, and numeric checks."""
    rng = _rng(seed, "augment")
    visible = {k: v for k, v in doc["fields"].items() if v is not None}
    qs = doc["questions"]

    def add(q: dict[str, Any]) -> None:
        q["_target"] = None
        qs.append(q)

    bools = [k for k, v in visible.items() if isinstance(v, bool)]
    rng.shuffle(bools)
    for k in bools[:5]:
        label = k.replace("_", " ")
        add(_noul(f"Does the document state that {label} is true?", f"{label} is true", f"{label} is false", "computed", {"kind": "field_eq", "field": k, "value": True}))

    enums = [k for k in ENUM_OPTIONS if k in visible]
    rng.shuffle(enums)
    for k in enums[:3]:
        crit = dict(ENUM_OPTIONS[k])
        items = list(crit.items())
        rng.shuffle(items)
        add(_choice(f"Which {k.replace('_', ' ')} is stated in the document?", dict(items), "computed", {"kind": "choice_value", "field": k}))

    date_pairs = [
        ("invoice_date", "scheduled_pay_date", 30, "Is scheduled payment within 30 days of invoice date?"),
        ("opened_date", "first_response_date", 2, "Is first response within two calendar days of opening?"),
        ("purchase_date", "opened_date", 30, "Was the ticket opened within 30 days of purchase?"),
        ("expense_date", "submitted_date", 30, "Was the request submitted within 30 days?"),
        ("delivered_date", "claim_date", 30, "Was the claim filed within 30 days of delivery?"),
        ("shipped_date", "delivered_date", 7, "Was delivery completed within seven days of shipment?"),
        ("last_patch_date", "detected_date", 30, "Was the asset patched within 30 days before detection?"),
    ]
    for start, end, limit, instr in date_pairs:
        if start in visible and end in visible:
            add(_noul(instr, "The interval is within the stated limit", "The interval exceeds the stated limit", "computed", {"kind": "days_le", "start": start, "end": end, "limit": limit}))

    after_pairs = [
        ("scheduled_pay_date", "invoice_date", "Is scheduled payment after the invoice date?"),
        ("claim_date", "delivered_date", "Is claim date after delivery date?"),
        ("submitted_date", "expense_date", "Is submission after the expense or leave date?"),
        ("detected_date", "last_patch_date", "Is detection after the last patch date?"),
    ]
    for after, before, instr in after_pairs:
        if after in visible and before in visible:
            add(_noul(instr, "The later field is after the earlier field", "The later field is not after the earlier field", "computed", {"kind": "date_after", "after": after, "before": before}))

    money_checks = [
        ("invoice_total", "1000.00", "Is the invoice total over $1,000?"),
        ("po_total", "1000.00", "Is the purchase order total over $1,000?"),
        ("amount", "75.00", "Is the request amount over $75?"),
        ("amount", "500.00", "Is the request amount over $500?"),
        ("declared_value", "500.00", "Is declared value over $500?"),
        ("declared_value", "1000.00", "Is declared value over $1,000?"),
    ]
    for field, threshold, instr in money_checks:
        if field in visible:
            add(_noul(instr, f"{field.replace('_', ' ')} exceeds {threshold}", f"{field.replace('_', ' ')} does not exceed {threshold}", "computed", {"kind": "money_gt", "field": field, "threshold": threshold}))

    int_checks = [
        ("cvss", 8, "Is CVSS at least 8?"),
        ("prior_alerts", 3, "Are there at least three prior alerts?"),
    ]
    for field, threshold, instr in int_checks:
        if field in visible:
            add(_noul(instr, f"{field.replace('_', ' ')} is at least {threshold}", f"{field.replace('_', ' ')} is below {threshold}", "computed", {"kind": "int_ge", "field": field, "threshold": threshold}))

    criteria, main_spec, name = MAIN_DECISIONS[doc["family"]]
    variants = [
        f"Which {name} should be recorded after applying all overrides?",
        f"Select the controlling {name} for this workflow document.",
        f"What is the policy result for the case?",
        f"Choose the outcome supported by the visible fields.",
    ]
    vi = 0
    while len(qs) < 20:
        req_missing = any(k not in visible for k in main_spec.get("required", []))
        if vi % 3 == 0 or req_missing:
            items = list(criteria.items())
            rng.shuffle(items)
            crit = dict(items)
            if req_missing:
                crit = _ct(crit)
            add(_choice(variants[vi % len(variants)], crit, "computed", dict(main_spec)))
        else:
            key = list(criteria)[vi % len(criteria)]
            add(_noul(f"Is {criteria[key].lower()} the correct {name}?", f"The correct result is {criteria[key].lower()}", f"The correct result is not {criteria[key].lower()}", "computed", {"kind": "computed_eq", "inner": dict(main_spec), "value": key}))
        vi += 1

    del qs[24:]


def generate_document(seed: int) -> dict[str, Any]:
    family = FAMILIES[seed % len(FAMILIES)]
    prose = bool(seed % 2)
    long_doc = seed % 5 in {0, 1, 2}
    doc = GENS[family](seed, prose, long_doc)
    doc["seed"] = seed
    doc["doc_id"] = f"workflow-{family}-{seed}"
    doc["length_bucket"] = "long" if long_doc else "short"
    _augment_document(seed, doc)
    return doc


def rows_for_seed(seed: int) -> list[dict[str, Any]]:
    doc = generate_document(seed)
    rows = []
    for i, q in enumerate(doc["questions"]):
        qq = {k: v for k, v in q.items() if not k.startswith("_")}
        spec = q["_spec"]
        fields = parse_state_fields(doc["state"])
        gold, target = solve_spec(fields, spec)
        row = {
            "id": f"{doc['doc_id']}:q{i:02d}",
            "state": doc["state"],
            "question": qq,
            "gold": gold,
            "target": target,
            "weight": 1.0,
            "src": f"workflow_{doc['family']}",
            "meta": {"family": doc["family"], "group": doc["doc_id"], "seed": seed, "q_index": i, "spec": spec, "length_bucket": doc["length_bucket"]},
        }
        rows.append(row)
    return rows


def generate_rows(seeds: Iterable[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seed in seeds:
        out.extend(rows_for_seed(seed))
    return out


def recompute_row(row: dict[str, Any]) -> tuple[str, dict[str, float] | None]:
    return solve_spec(parse_state_fields(row["state"]), row["meta"]["spec"])


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _question_keys(q: dict[str, Any]) -> list[str]:
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return ["yes", "no"]
    return [str(i) for i in range(len(q["criteria"]))]


def census(rows: list[dict[str, Any]], token_stats: dict[str, Any] | None = None, leakage: dict[str, Any] | None = None) -> dict[str, Any]:
    by_family = Counter(row["meta"]["family"] for row in rows)
    by_type = Counter(row["question"]["type"] for row in rows)
    by_length_bucket = Counter(row["meta"].get("length_bucket", "unknown") for row in rows)
    answer = Counter(f"{row['question']['type']}:{row['gold']}" for row in rows)
    family_type = Counter((row["meta"]["family"], row["question"]["type"]) for row in rows)
    family_gold = Counter((row["meta"]["family"], str(row["gold"])) for row in rows)
    cant = sum(1 for row in rows if row["gold"] == "cant_tell")
    pos = Counter()
    key_gold = Counter()
    for row in rows:
        keys = _question_keys(row["question"])
        if row["gold"] in keys:
            pos[keys.index(row["gold"])] += 1
        key_gold[str(row["gold"])] += 1
    docs = sorted({row["meta"]["group"] for row in rows})
    doc_buckets = Counter()
    seen_docs = set()
    for row in rows:
        g = row["meta"]["group"]
        if g not in seen_docs:
            seen_docs.add(g)
            doc_buckets[row["meta"].get("length_bucket", "unknown")] += 1
    return {
        "rows": len(rows),
        "documents": len(docs),
        "by_family": dict(sorted(by_family.items())),
        "by_question_type": dict(sorted(by_type.items())),
        "by_length_bucket_rows": dict(sorted(by_length_bucket.items())),
        "by_length_bucket_documents": dict(sorted(doc_buckets.items())),
        "family_question_type": {f"{k[0]}:{k[1]}": v for k, v in sorted(family_type.items())},
        "answer_distribution": dict(answer.most_common()),
        "family_answer_distribution": {f"{k[0]}:{k[1]}": v for k, v in sorted(family_gold.items())},
        "cant_tell": {"rows": cant, "rate": cant / len(rows) if rows else 0.0},
        "gold_position_distribution": dict(sorted(pos.items())),
        "gold_key_top20": dict(key_gold.most_common(20)),
        "token_stats": token_stats or {},
        "leakage": leakage or {},
    }


def token_stats(rows: list[dict[str, Any]], model: str = "Qwen/Qwen3.5-4B") -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
        from .render import Renderer
    except Exception as ex:  # noqa: BLE001
        return {"available": False, "error": f"{type(ex).__name__}: {ex}"}
    tok = AutoTokenizer.from_pretrained(model)
    rd = Renderer(tok, template="semif")
    state_lens, prompt_lens = [], []
    doc_states: dict[str, tuple[Any, str]] = {}
    for row in rows:
        state_text = json.dumps(row["state"], ensure_ascii=False) if not isinstance(row["state"], str) else row["state"]
        state_lens.append(len(tok(state_text, add_special_tokens=False)["input_ids"]))
        doc_states.setdefault(row["meta"]["group"], (row["state"], row["meta"].get("length_bucket", "unknown")))
        prompt, _, _ = rd.render(row["state"], row["question"])
        prompt_lens.append(len(tok(prompt, add_special_tokens=False)["input_ids"]))
    def stats(xs: list[int]) -> dict[str, float]:
        xs = sorted(xs)
        return {
            "min": xs[0],
            "p50": xs[len(xs) // 2],
            "p90": xs[int(len(xs) * 0.90)],
            "p99": xs[int(len(xs) * 0.99)],
            "max": xs[-1],
            "mean": statistics.fmean(xs),
        }
    doc_lens = []
    bucket_lens: dict[str, list[int]] = defaultdict(list)
    for state, bucket in doc_states.values():
        text = json.dumps(state, ensure_ascii=False) if not isinstance(state, str) else state
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        doc_lens.append(n)
        bucket_lens[bucket].append(n)
    bands = Counter()
    for n in doc_lens:
        if 600 <= n <= 1000:
            bands["600_1000"] += 1
        elif 1500 <= n <= 4000:
            bands["1500_4000"] += 1
        elif n < 600:
            bands["under_600"] += 1
        elif n < 1500:
            bands["1001_1499"] += 1
        else:
            bands["over_4000"] += 1
    return {
        "available": True,
        "tokenizer": model,
        "state_tokens": stats(state_lens),
        "prompt_tokens": stats(prompt_lens),
        "document_state_tokens": stats(doc_lens),
        "document_length_bands": dict(sorted(bands.items())),
        "document_length_buckets": {k: {"count": len(v), **stats(v)} for k, v in sorted(bucket_lens.items())},
    }


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")


def _ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    words = [w.lower() for w in _WORD.findall(text)]
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def leakage_check(rows: list[dict[str, Any]], roots: list[Path]) -> dict[str, Any]:
    row_grams: dict[tuple[str, ...], str] = {}
    for row in rows:
        text = json.dumps({"state": row["state"], "question": row["question"]}, ensure_ascii=False)
        for gram in _ngrams(text):
            row_grams.setdefault(gram, row["id"])
    checked_files = 0
    matches: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.stat().st_size > 20_000_000:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            checked_files += 1
            for gram in _ngrams(text):
                if gram in row_grams:
                    matches.append({"row": row_grams[gram], "path": str(path), "passage": " ".join(gram)})
                    if len(matches) >= 20:
                        return {"checked_files": checked_files, "matches": matches, "ok": False}
    return {"checked_files": checked_files, "matches": matches, "ok": not matches}


def write_readme(out_dir: Path) -> None:
    (out_dir / "README.md").write_text(
        """# blink workflow synthetic data

This directory contains code-labelled workflow training rows for blink.  Each
row is one typed question over a larger workflow document; documents come from
six original synthetic families: security alert triage, accounts-payable invoice
checks, support refund/SLA review, agent tool-use traces, HR expense/leave
requests, and shipping claims.

Design rules:
- every answer is computed by code from fields rendered inside the state;
- documents mix JSON states and prose worksheets;
- the seed schedule emits 40% short documents around 600-1,000 Qwen3.5-4B
  tokens and 60% long documents around 1,500-4,000 tokens;
- long documents add multi-section policies, effective-date version tables,
  numbered clauses, overrides, superseded sections, longer event logs, messages,
  line-item-style histories, and plausible distractors;
- policies include overrides, exceptions, distractors, dates, amounts, windows,
  thresholds, business-day checks, sums, and missing-evidence cases;
- `cant_tell` is offered only when a field needed for that question is absent;
- deterministic facts use one-hot gold; random-draw questions use exact target
  distributions computed from stated counts;
- no benchmark item text or TypeSafe/JevBench templates are used.

Regenerate from the repository root:

```bash
cd lab
python3 -m jevlab.synth_workflow --out-dir ../data/workflow
python3 -m unittest tests.test_synth_workflow
```
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../data/workflow")
    ap.add_argument("--train-start", type=int, default=1000)
    ap.add_argument("--train-end", type=int, default=1999)
    ap.add_argument("--dev-start", type=int, default=9000)
    ap.add_argument("--dev-end", type=int, default=9099)
    ap.add_argument("--skip-token-stats", action="store_true")
    ap.add_argument("--skip-leakage", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    train = generate_rows(range(a.train_start, a.train_end + 1))
    dev = generate_rows(range(a.dev_start, a.dev_end + 1))
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "dev.jsonl", dev)
    stats_rows = train + dev
    ts = {} if a.skip_token_stats else token_stats(stats_rows)
    roots = [Path("../heldout"), Path("../typesafe/raw")]
    leak = {} if a.skip_leakage else leakage_check(stats_rows, roots)
    c = {"train": census(train), "dev": census(dev), "all": census(stats_rows, ts, leak)}
    (out / "census.json").write_text(json.dumps(c, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(out)
    print(json.dumps({"train_rows": len(train), "dev_rows": len(dev), "out_dir": str(out), "token_stats": ts, "leakage": leak}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
