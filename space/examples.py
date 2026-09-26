"""Example states, question sets, and the decision policy for each use case.

The policy functions below are the real code the tabs run, and the same source is
shown in the interface — the thresholds on screen are the thresholds that fired.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Callable

# --- playground defaults ------------------------------------------------------------

PLAYGROUND_STATE = (
    "From: contact at example dot test\n"
    "Subject: Overdue invoice 88213 — final notice\n\n"
    "Our records show invoice 88213 for $4,180 is 21 days overdue. Wire the balance to the "
    "updated account below today to avoid suspension. Account details changed this quarter; "
    "use the new IBAN, not the one on the original invoice."
)

PLAYGROUND_QUESTIONS = json.dumps(
    {
        "intent": {
            "type": "choice",
            "instructions": "What is the sender trying to get the reader to do?",
            "criteria": {
                "pay_invoice": "Pay money against an invoice or bill",
                "share_credentials": "Hand over a password, code, or login",
                "book_meeting": "Agree to a call or meeting",
                "no_action": "Nothing; the message is informational",
            },
        },
        "suspicious": {
            "type": "noul",
            "instructions": "Does this message show signs of payment fraud?",
            "criteria": {
                "true": "Pressure, deadlines, or changed payment details",
                "false": "Ordinary correspondence with no fraud signals",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How quickly does a human need to look at this?",
            "criteria": [
                "No action needed",
                "This week",
                "Today",
                "Within the hour",
                "Immediately",
            ],
        },
    },
    indent=2,
)


# --- structure ----------------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    label: str
    state: str


@dataclass(frozen=True)
class UseCase:
    key: str
    title: str
    blurb: str
    questions: dict
    examples: list[Example]
    verdict: Callable[[dict], tuple]
    policy_note: str = ""
    state_label: str = "Evidence"
    state_lines: int = 10
    _src: str = field(default="", repr=False)

    @property
    def source(self) -> str:
        return inspect.getsource(self.verdict).rstrip()


def _p(answers: dict, key: str) -> dict:
    return answers[key]["probabilities"]


# --- 1. support triage --------------------------------------------------------------

SUPPORT_QUESTIONS = {
    "queue": {
        "type": "choice",
        "instructions": "Which team should own this ticket?",
        "criteria": {
            "billing": "Charges, invoices, refunds, plan changes",
            "technical": "Errors, outages, broken features, data loss",
            "account": "Login, access, permissions, seats",
            "shipping": "Orders, delivery, returns of physical goods",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this for the customer?",
        "criteria": [
            "Can wait a week",
            "Answer within two days",
            "Answer today",
            "Answer within an hour",
            "Drop everything",
        ],
    },
    "refund_eligible": {
        "type": "noul",
        "instructions": "Does the customer describe a charge that qualifies for a refund?",
        "criteria": {
            "true": "A duplicate, unauthorised, or undelivered charge",
            "false": "No charge problem, or the charge was correct",
        },
    },
    "angry": {
        "type": "noul",
        "instructions": "Is the customer angry or threatening to leave?",
    },
}


def support_verdict(a: dict) -> tuple:
    urgency = a["urgency"]["score"]
    refund = a["refund_eligible"]["noul"]
    angry = a["angry"]["noul"]
    queue = a["queue"]["choice"]
    confidence = a["queue"]["confidence"]

    if confidence < 0.35:
        return "hold", "Route by hand", "No clear queue; a person picks."
    if urgency >= 3.0 or (angry > 0.70 and urgency >= 2.0):
        sla = "1 hour"
    elif urgency >= 2.0:
        sla = "same day"
    elif urgency >= 1.0:
        sla = "2 days"
    else:
        sla = "1 week"
    tone = "stop" if sla == "1 hour" else "go"
    detail = f"Response target: {sla}."
    if refund > 0.60:
        detail += " An agent checks the refund."
    return tone, f"Send to {queue} · {sla}", detail


SUPPORT = UseCase(
    key="support",
    title="Support tickets",
    blurb="Pick a team, a response target and a refund flag from a ticket.",
    questions=SUPPORT_QUESTIONS,
    policy_note="Urgency can fall between levels; the response target follows that value.",
    examples=[
        Example(
            "Double charge",
            "I was billed twice for the Pro plan this month — $49 on the 3rd and $49 again on the 4th. "
            "Same card, same plan, one subscription. I need the second charge back before my card statement closes on Friday.",
        ),
        Example(
            "Data loss, about to churn",
            "Your last release wiped three months of our saved reports. My whole team is blocked and we present to "
            "our board tomorrow. This is the third outage this quarter. If it is not fixed today we are moving to a competitor.",
        ),
        Example(
            "Seat request",
            "Hi — we hired two analysts starting next month and I would like to add them to our workspace. "
            "No rush, just let me know how seats are billed when you get a chance.",
        ),
    ],
    verdict=support_verdict,
)


# --- 2. email and phishing ----------------------------------------------------------

EMAIL_QUESTIONS = {
    "phishing": {
        "type": "noul",
        "instructions": "Is this message a phishing or fraud attempt?",
        "criteria": {
            "true": "Impersonation, forged sender, or a lure to a credential or payment",
            "false": "A legitimate message from who it claims to be",
        },
    },
    "intent": {
        "type": "choice",
        "instructions": "What action is the sender asking for?",
        "criteria": {
            "credentials": "Log in, reset a password, or share a code",
            "payment": "Pay, wire, or change payment details",
            "attachment": "Open a file or enable content",
            "reply": "Send information back by email",
            "none": "No action; informational only",
        },
    },
    "impersonates_staff": {
        "type": "noul",
        "instructions": "Does the sender claim to be a colleague, executive, or an internal system?",
    },
    "pressure": {
        "type": "score",
        "instructions": "How much time pressure does the message apply?",
        "criteria": ["None", "Mild", "A firm deadline", "Threats of consequences"],
    },
}


def email_verdict(a: dict) -> tuple:
    phish = a["phishing"]["noul"]
    intent = a["intent"]["choice"]
    internal = a["impersonates_staff"]["noul"]
    pressure = a["pressure"]["score"]

    risky_ask = intent in ("credentials", "payment")
    if phish >= 0.80 or (phish >= 0.55 and risky_ask and pressure >= 2.0):
        return "stop", "Quarantine", f"Blocked before delivery; asks for {intent}."
    if phish >= 0.45 or (risky_ask and internal >= 0.60):
        return "hold", "Show warning", "Delivered with a warning."
    return "go", "Deliver", "No fraud signal above the review threshold."


EMAIL = UseCase(
    key="email",
    title="Email screening",
    blurb="Deliver, warn or quarantine based on the message.",
    questions=EMAIL_QUESTIONS,
    policy_note="Quarantine needs a strong fraud signal, or a moderate one with a risky ask and deadline.",
    examples=[
        Example(
            "Fake CEO wire",
            "From: Dana Whitfield <contact at example dot test>\nSubject: quick favour, in a board meeting\n\n"
            "Are you at your desk? I need a supplier paid before close of business and I cannot take calls for the "
            "next two hours. Send $28,400 to the account in the attached PDF and confirm here when it is done. "
            "Please keep this between us until the announcement.",
        ),
        Example(
            "Internal MFA re-enrolment",
            "From: contact at example dot test\nSubject: Re-enrol your authenticator by Friday\n\n"
            "We are migrating everyone to the new authenticator app. Please re-enrol from the IT portal "
            "before Friday — accounts that have not moved by then lose access to email and VPN until a "
            "helpdesk ticket is raised. The portal link is on the intranet home page.",
        ),
        Example(
            "Ordinary vendor mail",
            "From: contact at example dot test\nSubject: Q3 delivery schedule\n\n"
            "Hi team — attaching the updated delivery schedule for Q3. Two dates moved by a day because of the "
            "bank holiday. No action needed from your side; shout if anything clashes.",
        ),
    ],
    verdict=email_verdict,
)


# --- 3. policy decisions ------------------------------------------------------------

POLICY_QUESTIONS = {
    "outcome": {
        "type": "choice",
        "instructions": "Apply the policy to the claim and choose the outcome.",
        "criteria": {
            "approve": "The claim meets every condition in the policy",
            "partial": "Some of the claim qualifies and some does not",
            "deny": "An exclusion or condition clearly rules the claim out",
            "insufficient": "The document does not say enough to decide",
        },
    },
    "needs_review": {
        "type": "noul",
        "instructions": "Does a human have to sign this off before it is sent?",
        "criteria": {
            "true": "Ambiguous wording, a large amount, or an exception applies",
            "false": "A routine application of a clear rule",
        },
    },
    "within_window": {
        "type": "noul",
        "instructions": "Was the claim filed inside the time limit the policy sets?",
    },
    "complexity": {
        "type": "score",
        "instructions": "How much reading does this claim take?",
        "criteria": ["One clause", "A few clauses", "Cross-references", "Conflicting clauses"],
    },
}


def policy_verdict(a: dict) -> tuple:
    outcome = a["outcome"]["choice"]
    conf = a["outcome"]["confidence"]
    review = a["needs_review"]["noul"]
    in_window = a["within_window"]["noul"]
    complexity = a["complexity"]["score"]

    if outcome == "insufficient" or conf < 0.40:
        return "hold", "Ask for details", "The policy does not settle this claim."
    if in_window < 0.50:
        return "stop", "Deny: filed late", "Past the filing deadline."
    if review > 0.55 or complexity >= 2.0 or outcome == "partial":
        return "hold", f"{outcome.title()} — pending sign-off", "Drafted for a human to approve."
    return "go", outcome.title(), f"Clear policy rule; confidence {conf:.0%}."


POLICY = UseCase(
    key="policy",
    title="Policy checks",
    blurb="Apply the policy, check the deadline and flag human review.",
    questions=POLICY_QUESTIONS,
    state_lines=14,
    policy_note="The deadline wins: a late claim is denied even if the other rules allow it.",
    examples=[
        Example(
            "Laptop, clear approval",
            "POLICY 4.2 — EQUIPMENT DAMAGE\n"
            "(a) The company replaces a damaged laptop once per 24 months.\n"
            "(b) Claims must be filed within 30 days of the damage.\n"
            "(c) Liquid damage is covered. Deliberate damage and cosmetic wear are not.\n"
            "(d) Replacements above $2,500 need director approval.\n\n"
            "CLAIM 2026-0418 — filed 9 days after the incident.\n"
            "Employee spilled coffee on a company MacBook Air; it no longer powers on. "
            "Last replacement was 31 months ago. Quoted replacement: $1,380.",
        ),
        Example(
            "Filed after the window",
            "POLICY 4.2 — EQUIPMENT DAMAGE\n"
            "(a) The company replaces a damaged laptop once per 24 months.\n"
            "(b) Claims must be filed within 30 days of the damage.\n"
            "(c) Liquid damage is covered. Deliberate damage and cosmetic wear are not.\n"
            "(d) Replacements above $2,500 need director approval.\n\n"
            "CLAIM 2026-0503 — damage occurred 2 March, filed 7 May.\n"
            "Screen cracked when the bag was dropped. Last replacement 40 months ago. Quoted: $1,910.",
        ),
        Example(
            "Conflicting clauses",
            "POLICY 7.1 — TRAVEL\n"
            "(a) Economy fares are reimbursed in full.\n"
            "(b) Premium cabins are reimbursed only for flights over 8 hours.\n"
            "(c) Where a client contract sets a travel policy, that contract governs.\n\n"
            "CLAIM 2026-0461.\n"
            "Premium economy, London to Dubai, 6h 55m block time. Booked under the Halvorsen engagement, "
            "whose contract allows premium cabins on any international flight. Fare $2,240.",
        ),
    ],
    verdict=policy_verdict,
)


# --- 4. RAG passage filter ----------------------------------------------------------

RAG_QUESTIONS = {
    "relevant": {
        "type": "noul",
        "instructions": "Is this passage about the subject of the question?",
        "criteria": {"true": "On topic", "false": "A different subject"},
    },
    "answerable": {
        "type": "noul",
        "instructions": "Does this passage alone contain the facts needed to answer the question?",
        "criteria": {
            "true": "The answer can be read straight out of the passage",
            "false": "The passage is related but does not state the answer",
        },
    },
    "contradicts": {
        "type": "noul",
        "instructions": "Does the passage contradict the premise of the question?",
    },
    "specificity": {
        "type": "score",
        "instructions": "How specific is the passage about the thing asked?",
        "criteria": ["General background", "Mentions it", "Exact figures or rules"],
    },
}


def rag_verdict(a: dict) -> tuple:
    relevant = a["relevant"]["noul"]
    answerable = a["answerable"]["noul"]
    contradicts = a["contradicts"]["noul"]
    specificity = a["specificity"]["score"]

    if contradicts > 0.60:
        return "hold", "Keep and flag", "The passage contradicts the question."
    if relevant < 0.40:
        return "stop", "Drop", "Off topic; omit it from the answer."
    if answerable >= 0.60 and specificity >= 1.5:
        return "go", "Cite", "Enough to answer; cite this passage."
    return "hold", "Keep as context", "On topic, but not enough to answer alone."


RAG = UseCase(
    key="rag",
    title="Passage check",
    blurb="Keep passages that help answer the question; flag contradictions.",
    questions=RAG_QUESTIONS,
    state_lines=12,
    policy_note="A passage can be on topic without containing the answer.",
    examples=[
        Example(
            "Sufficient passage",
            "QUESTION: How long is the free trial for the Team plan, and does it need a card?\n\n"
            "PASSAGE: Team plan trials run for 14 days from first sign-in. A payment card is not required to start "
            "a trial; billing begins only if the workspace is converted to a paid plan before the trial ends. "
            "Trials cannot be extended, but a workspace may start a second trial after 12 months.",
        ),
        Example(
            "Related but not sufficient",
            "QUESTION: How long is the free trial for the Team plan, and does it need a card?\n\n"
            "PASSAGE: Our plans are Free, Team, and Enterprise. Team adds shared workspaces, audit logs, and "
            "priority support, and is billed per seat per month with an annual discount. Enterprise adds SSO, "
            "data residency, and a dedicated success manager.",
        ),
        Example(
            "Off topic",
            "QUESTION: How long is the free trial for the Team plan, and does it need a card?\n\n"
            "PASSAGE: To rotate an API key, open Settings → Developers, choose the key, and select Rotate. "
            "The previous key keeps working for 24 hours so running jobs are not interrupted.",
        ),
    ],
    verdict=rag_verdict,
)


# --- 5. moderation and safety -------------------------------------------------------

MODERATION_QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "Which policy does this content violate, if any?",
        "criteria": {
            "none": "No violation",
            "harassment": "Targeted abuse, threats, or hate directed at a person or group",
            "self_harm": "Suicide or self-injury content",
            "violence": "Threats or glorification of violence",
            "fraud": "Scams, impersonation, or deceptive selling",
            "sexual": "Sexual content involving adults where it is not allowed",
        },
    },
    "escalate": {
        "type": "noul",
        "instructions": "Does this need a trained human reviewer now?",
        "criteria": {
            "true": "Possible real-world harm to a person",
            "false": "A routine moderation call",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the violation?",
        "criteria": ["None", "Borderline", "Clear", "Severe"],
    },
    "targets_person": {
        "type": "noul",
        "instructions": "Is a specific, identifiable person targeted?",
    },
}


def moderation_verdict(a: dict) -> tuple:
    category = a["category"]["choice"]
    conf = a["category"]["confidence"]
    escalate = a["escalate"]["noul"]
    severity = a["severity"]["score"]
    targeted = a["targets_person"]["noul"]

    if escalate > 0.60 or category == "self_harm":
        return "stop", "Escalate to a person now", "Sent to the on-call safety reviewer."
    if category == "none" and severity < 1.0:
        return "go", "Publish", "Nothing above the borderline level."
    if severity >= 2.0 and targeted > 0.50:
        return "stop", f"Remove · {category}", f"Clear violation targeting a person; confidence {conf:.0%}."
    if severity >= 1.0 or conf < 0.40:
        return "hold", f"Queue for review · {category}", "Borderline or uncertain; hidden until reviewed."
    return "go", "Publish", "Below every action threshold."


MODERATION = UseCase(
    key="moderation",
    title="Content review",
    blurb="Flag posts needing a person now; handle the rest by category.",
    questions=MODERATION_QUESTIONS,
    state_lines=8,
    policy_note="Escalation comes first, even when the category is uncertain.",
    examples=[
        Example(
            "Crisis language",
            "i've been awake for three days and i don't see the point anymore. i have the pills counted out on "
            "the desk. sorry for posting this here, i just didn't know where else to put it.",
        ),
        Example(
            "Targeted abuse",
            "everyone should mass-report @jenna_liu_ until her account is gone. post her workplace in the replies, "
            "she deserves everything coming to her.",
        ),
        Example(
            "Heated but allowed",
            "this update is genuinely the worst design decision this company has ever made and whoever signed it "
            "off should be embarrassed. three years of muscle memory thrown away for nothing.",
        ),
    ],
    verdict=moderation_verdict,
)


# --- 6. model routing ---------------------------------------------------------------

ROUTING_QUESTIONS = {
    "task": {
        "type": "choice",
        "instructions": "What kind of work is this prompt asking for?",
        "criteria": {
            "code": "Write, fix, or explain code",
            "math": "Calculate or prove something",
            "writing": "Draft or edit prose",
            "extraction": "Pull structured fields out of a text",
            "knowledge": "Answer a factual question",
            "chitchat": "Small talk or a greeting",
        },
    },
    "difficulty": {
        "type": "score",
        "instructions": "How hard is this for a language model?",
        "criteria": [
            "Trivial",
            "Easy",
            "Moderate",
            "Hard, needs careful work",
            "Expert level",
        ],
    },
    "needs_tools": {
        "type": "noul",
        "instructions": "Does answering this need a tool, a search, or fresh data?",
    },
    "long_output": {
        "type": "noul",
        "instructions": "Does a good answer run longer than a few paragraphs?",
    },
}

ROUTE_TABLE = {
    "small": "4B · $0.03/M",
    "mid": "27B · $0.20/M",
    "frontier": "large model · $3/M",
}


def routing_verdict(a: dict) -> tuple:
    task = a["task"]["choice"]
    difficulty = a["difficulty"]["score"]
    tools = a["needs_tools"]["noul"]
    long_out = a["long_output"]["noul"]

    if tools > 0.60:
        return "hold", "Check the facts first", "Get current information, then choose a model."
    if difficulty >= 3.0 or (difficulty >= 2.4 and task in ("code", "math")):
        tier = "frontier"
    elif difficulty >= 1.4 or long_out > 0.60:
        tier = "mid"
    else:
        tier = "small"
    tone = {"small": "go", "mid": "go", "frontier": "hold"}[tier]
    return tone, f"Route to {ROUTE_TABLE[tier]}", f"{task} · difficulty {difficulty:.2f} of 4."


ROUTING = UseCase(
    key="routing",
    title="Pick a model",
    blurb="Choose a model for the task, or get fresh information first.",
    questions=ROUTING_QUESTIONS,
    state_lines=8,
    policy_note="Difficulty can fall between levels; code and maths use a stricter cutoff.",
    examples=[
        Example(
            "Trivial",
            "hey, what's a good short name for a golden retriever puppy? something two syllables.",
        ),
        Example(
            "Hard code",
            "Our Postgres read replica drifts up to 40 seconds behind under write bursts, but only when the "
            "logical replication slot is shared with a CDC consumer. Walk me through diagnosing whether the "
            "bottleneck is WAL sender throughput or apply-side lock contention, and write the queries I should run.",
        ),
        Example(
            "Needs fresh data",
            "What did the Bank of England decide at its meeting this month, and how does that compare with the "
            "market pricing from the week before?",
        ),
    ],
    verdict=routing_verdict,
)


# --- 7. next click -------------------------------------------------------------------

NEXTCLICK_QUESTIONS = {
    "element": {
        "type": "choice",
        "instructions": "Which element should the agent act on next?",
        "criteria": [
            "checkbox: In stock only",
            "input: Max price",
            "button: Apply filters",
            "button: Clear all",
            "link: Sort by price",
        ],
    },
    "done": {
        "type": "noul",
        "instructions": "Is the task already finished?",
        "criteria": {
            "true": "The page shows the task completed",
            "false": "Something still has to happen",
        },
    },
}


def nextclick_verdict(a: dict) -> tuple:
    done = a["done"]["noul"]
    element = a["element"]
    top = max(element["probabilities"].values())
    if done >= 0.60:
        return "go", "Nothing left to click", f"Task appears finished ({done:.0%} yes)."
    if top < 0.45:
        return "hold", "Hand back to a person", f"Top option: {top:.0%}."
    return "go", f"Act on {element['choice']}", f"Top option {top:.0%}; task not finished ({done:.0%} yes)."


def _page(task: str, where: str, steps: list[str]) -> str:
    done = "\n".join(f"  {i + 1}. {t}" for i, t in enumerate(steps))
    return f"Task: {task}\n\nPage: {where}\nDone so far:\n{done}"


NEXTCLICK = UseCase(
    key="nextclick",
    title="Next click",
    blurb="Pick the next page element from the task and current page.",
    questions=NEXTCLICK_QUESTIONS,
    state_label="Task and page",
    state_lines=10,
    policy_note="The options are page elements. This demo does not click anything.",
    examples=[
        Example(
            "Nothing set yet",
            _page(
                "narrow the lighting list to in-stock lamps under 50",
                "a lighting shop, 212 results, the filter panel open on the left",
                ['Opened the "Filters" panel.'],
            ),
        ),
        Example(
            "Ready to apply",
            _page(
                "narrow the lighting list to in-stock lamps under 50",
                "a lighting shop, 212 results, the filter panel open on the left",
                [
                    'Opened the "Filters" panel.',
                    'Ticked "In stock only".',
                    'Typed 50 into the "Max price" box.',
                ],
            ),
        ),
        Example(
            "Already filtered",
            _page(
                "narrow the lighting list to in-stock lamps under 50",
                'a lighting shop showing 6 results under the chips "In stock" and "Under 50"',
                [
                    'Opened the "Filters" panel.',
                    'Ticked "In stock only".',
                    'Typed 50 into the "Max price" box.',
                    'Pressed "Apply filters".',
                ],
            ),
        ),
    ],
    verdict=nextclick_verdict,
)


USE_CASES = [SUPPORT, EMAIL, POLICY, RAG, MODERATION, ROUTING, NEXTCLICK]

# (label, state text, questions JSON text) — the playground's preset buttons.
PLAYGROUND_PRESETS = [
    ("Invoice email", PLAYGROUND_STATE, PLAYGROUND_QUESTIONS),
    ("Support ticket", SUPPORT.examples[0].state, json.dumps(SUPPORT.questions, indent=2)),
    ("Retrieved passage", RAG.examples[1].state, json.dumps(RAG.questions, indent=2)),
]
CASES_BY_KEY = {c.key: c for c in USE_CASES}


# --- demo shaping -------------------------------------------------------------------
# Read only by the mock engine, which has no idea what any of this text means. These
# nudges give the bundled examples a realistic shape on a machine with no GPU. The
# trained model never sees them — it reads the evidence.

PLAYGROUND_BIAS = {
    "intent": {"pay_invoice": 3.4, "share_credentials": 0.4},
    "suspicious": {"yes": 2.9},
    "urgency": {"3": 2.8, "2": 1.6, "4": 0.7},
}

MOCK_BIAS = {
    ("support", "Double charge"): {
        "queue": {"billing": 3.6},
        "urgency": {"2": 3.4, "3": 1.4, "1": 1.0},
        "refund_eligible": {"yes": 3.1},
        "angry": {"no": 1.6},
    },
    ("support", "Data loss, about to churn"): {
        "queue": {"technical": 3.7},
        "urgency": {"4": 3.2, "3": 2.1},
        "refund_eligible": {"no": 1.0},
        "angry": {"yes": 3.3},
    },
    ("support", "Seat request"): {
        "queue": {"account": 2.8, "billing": 1.4},
        "urgency": {"0": 2.7, "1": 1.8},
        "refund_eligible": {"no": 2.6},
        "angry": {"no": 3.0},
    },
    ("email", "Fake CEO wire"): {
        "phishing": {"yes": 3.5},
        "intent": {"payment": 3.6},
        "impersonates_staff": {"yes": 3.4},
        "pressure": {"3": 2.6, "2": 2.0},
    },
    ("email", "Internal MFA re-enrolment"): {
        "phishing": {"no": 0.9},
        "intent": {"credentials": 3.2},
        "impersonates_staff": {"yes": 2.4},
        "pressure": {"2": 2.5, "1": 1.6},
    },
    ("email", "Ordinary vendor mail"): {
        "phishing": {"no": 3.3},
        "intent": {"none": 2.6, "attachment": 1.4},
        "impersonates_staff": {"no": 3.0},
        "pressure": {"0": 2.8, "1": 1.4},
    },
    ("policy", "Laptop, clear approval"): {
        "outcome": {"approve": 3.8},
        "needs_review": {"no": 2.7},
        "within_window": {"yes": 3.4},
        "complexity": {"1": 2.6, "0": 1.6},
    },
    ("policy", "Filed after the window"): {
        "outcome": {"deny": 3.4},
        "needs_review": {"no": 1.2},
        "within_window": {"no": 3.5},
        "complexity": {"1": 2.4, "0": 1.4},
    },
    ("policy", "Conflicting clauses"): {
        "outcome": {"approve": 2.1, "partial": 1.9, "insufficient": 0.8},
        "needs_review": {"yes": 3.2},
        "within_window": {"yes": 2.4},
        "complexity": {"3": 3.0, "2": 2.2},
    },
    ("rag", "Sufficient passage"): {
        "relevant": {"yes": 3.6},
        "answerable": {"yes": 3.4},
        "contradicts": {"no": 3.0},
        "specificity": {"2": 3.1, "1": 1.5},
    },
    ("rag", "Related but not sufficient"): {
        "relevant": {"yes": 3.0},
        "answerable": {"no": 2.8},
        "contradicts": {"no": 3.0},
        "specificity": {"1": 2.6, "0": 1.8},
    },
    ("rag", "Off topic"): {
        "relevant": {"no": 3.3},
        "answerable": {"no": 3.2},
        "contradicts": {"no": 3.0},
        "specificity": {"0": 2.8},
    },
    ("moderation", "Crisis language"): {
        "category": {"self_harm": 3.9},
        "escalate": {"yes": 3.7},
        "severity": {"3": 2.4, "2": 2.2},
        "targets_person": {"no": 1.4},
    },
    ("moderation", "Targeted abuse"): {
        "category": {"harassment": 3.7},
        "escalate": {"no": 0.6},
        "severity": {"3": 2.9, "2": 2.3},
        "targets_person": {"yes": 3.5},
    },
    ("moderation", "Heated but allowed"): {
        "category": {"none": 3.4},
        "escalate": {"no": 3.2},
        "severity": {"0": 3.0, "1": 1.1},
        "targets_person": {"no": 2.6},
    },
    ("routing", "Trivial"): {
        "task": {"writing": 2.7, "chitchat": 2.2},
        "difficulty": {"0": 3.0, "1": 1.7},
        "needs_tools": {"no": 3.2},
        "long_output": {"no": 3.1},
    },
    ("routing", "Hard code"): {
        "task": {"code": 3.8},
        "difficulty": {"3": 3.0, "4": 2.0},
        "needs_tools": {"no": 1.0},
        "long_output": {"yes": 3.0},
    },
    ("routing", "Needs fresh data"): {
        "task": {"knowledge": 3.3},
        "difficulty": {"2": 2.5, "1": 1.8},
        "needs_tools": {"yes": 3.6},
        "long_output": {"no": 1.6},
    },
}


NEXTCLICK_BIAS = {
    ("nextclick", "Nothing set yet"): {
        "element": {"checkbox: In stock only": 3.4, "input: Max price": 1.6},
        "done": {"no": 3.2},
    },
    ("nextclick", "Ready to apply"): {
        "element": {"button: Apply filters": 3.7, "input: Max price": 0.7},
        "done": {"no": 3.6},
    },
    ("nextclick", "Already filtered"): {
        "element": {"link: Sort by price": 1.1, "button: Clear all": 0.8},
        "done": {"yes": 0.6},
    },
}
MOCK_BIAS.update(NEXTCLICK_BIAS)


def bias_for(case_key: str, example_label: str) -> dict:
    return MOCK_BIAS.get((case_key, example_label), {})
