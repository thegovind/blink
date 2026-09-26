from __future__ import annotations

import itertools
import json
import random
import re
import sys
import unittest
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevlab import synth_core

getcontext().prec = 28

ROLE_WORDS = {"correct", "wrong", "naive", "trap", "tempting", "actual", "proper", "valid", "final"}
DIVERSITY_STATS: dict[str, dict[str, float]] = {}


def parse_d(s: str) -> date:
    return date.fromisoformat(s)


def date_key(d: date) -> str:
    return "date_" + d.isoformat().replace("-", "_")


def amount_key(x: Decimal) -> str:
    q = Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return "amount_" + str(q).replace(".", "_").replace("-", "neg_")


def business_add(start: date, n: int, holidays: set[date]) -> date:
    cur = start
    left = n
    while left:
        cur += timedelta(days=1)
        if cur.weekday() < 5 and cur not in holidays:
            left -= 1
    return cur


def month_end(d: date) -> date:
    return date(d.year + (d.month == 12), 1 if d.month == 12 else d.month + 1, 1) - timedelta(days=1)


def validate_row(row: dict) -> None:
    assert set(row) == {"id", "src", "state", "question", "gold", "target", "meta"}
    assert isinstance(row["id"], str) and row["id"]
    assert isinstance(row["src"], str) and row["src"].startswith("core_")
    assert isinstance(row["state"], (str, dict))
    assert isinstance(row["question"], dict)
    assert row["target"] is None or isinstance(row["target"], dict)
    assert isinstance(row["meta"], dict)
    assert row["meta"].get("family") in synth_core.FAMILIES
    assert isinstance(row["meta"].get("group"), str)
    q = row["question"]
    assert q["type"] in {"choice", "noul", "score"}
    assert isinstance(q["instructions"], str) and q["instructions"]
    assert not re.search(r"\b(?:cl|tk|ap|inv|rt|pol|case|ref)-\d{5}\b", q["instructions"].lower())
    assert not re.search(r"\b[a-z]{3,}-(?:atlas|harbor|signal|ledger|ribbon|cobalt|folio|matrix)-(?:amber|violet|cedar|lumen|prairie|orbit)\b", stringify_state(row["state"]).lower() + " " + q["instructions"].lower())
    forbidden = [
        "belongs to",
        "falls in",
        "has value true",
        "has value false",
        "measures less than",
        "measures at least",
        "measures more than",
        "measures no more than",
        "with where",
        "with for",
        "confirming reported",
        "for records showing for",
        "; with ",
        "  ;",
        "  ",
    ]
    blob = stringify_state(row["state"]).lower()
    for phrase in forbidden:
        assert phrase not in blob, phrase
    assert not re.search(r"\bsections \d+ are\b", blob), blob
    if q["type"] == "choice":
        assert isinstance(q["criteria"], dict)
        assert 2 <= len(q["criteria"]) <= 8
        assert row["gold"] in q["criteria"]
        for key, desc in q["criteria"].items():
            assert key == synth_core._snake(key), key
            assert not any(part in ROLE_WORDS for part in key.split("_")), key
            assert isinstance(desc, str) and len(desc.split()) <= 12, desc
            assert not desc.endswith(("inciden", "approv", "escalat")), desc
        if row["meta"].get("family") == "underdetermined":
            present = {"cannot_determine", "ask_for_clarification"} & set(q["criteria"])
            assert len(present) == 1, present
    elif q["type"] == "noul":
        assert set(q["criteria"]) == {"true", "false"}
        assert row["gold"] in {"yes", "no"}
        joined = (q["instructions"] + " " + q["criteria"]["true"] + " " + q["criteria"]["false"]).lower()
        assert "not permitted" not in joined
        assert "not apply" not in joined
    else:
        assert isinstance(q["criteria"], list)
        assert 3 <= len(q["criteria"]) <= 5
        assert row["gold"].isdigit()
        assert 0 <= int(row["gold"]) < len(q["criteria"])
        assert all(not re.match(r"^\s*\d+\s*:", c) for c in q["criteria"])


def eval_cond(cond: dict, facts: dict) -> bool:
    if cond["attr"] not in facts:
        return False
    val, tgt, op = facts[cond["attr"]], cond["value"], cond["op"]
    if op == "in":
        return val in tgt
    if op == "eq":
        return val == tgt
    if op == "gt":
        return val > tgt
    if op == "ge":
        return val >= tgt
    if op == "lt":
        return val < tgt
    if op == "le":
        return val <= tgt
    raise AssertionError(op)


def facts_with_definitions(latent: dict, facts: dict | None = None) -> dict:
    out = dict(latent["facts"] if facts is None else facts)
    for d in latent.get("definitions", []):
        if d["attr"] in out:
            out[d["term"]] = eval_cond({"attr": d["attr"], "op": d["op"], "value": d["value"]}, out)
    return out


def solve_rules_latent(latent: dict) -> str:
    facts = facts_with_definitions(latent)
    matches = [r for r in latent["rules"] if all(eval_cond(c, facts) for c in r["conditions"])]
    if not matches:
        return latent["base_outcome"]
    scheme = latent["precedence"]
    if scheme == "first_match":
        win = min(matches, key=lambda r: r["section"])
        return win.get("outcome", win.get("handler"))
    if scheme == "last_match":
        win = max(matches, key=lambda r: r["section"])
        return win.get("outcome", win.get("handler"))
    if scheme == "most_specific":
        win = max(matches, key=lambda r: (len(r["conditions"]) + (1 if r.get("exception_to") is not None else 0), r["section"]))
        return win.get("outcome", win.get("handler"))
    override_ids = set(latent.get("overrides", []))
    ovr = [r for r in matches if r["section"] in override_ids]
    win = max(ovr, key=lambda r: r["section"]) if ovr else min(matches, key=lambda r: r["section"])
    return win.get("outcome", win.get("handler"))


def solve_rules_with_facts(latent: dict, facts: dict) -> str:
    tmp = dict(latent)
    tmp["facts"] = facts
    return solve_rules_latent(tmp)


def solve_dates(row: dict) -> str:
    meta = row["meta"]["latent"]
    op = meta["operation"]
    if op == "deadline":
        anchor = parse_d(meta["anchor"])
        holidays = {parse_d(x) for x in meta["holidays"]}
        due = anchor + timedelta(days=meta["n"]) if meta["mode"] == "calendar" else business_add(anchor, meta["n"], holidays) if meta["mode"] == "business" else month_end(anchor)
        assert due.isoformat() == meta["due"]
        if row["question"]["type"] == "noul":
            submitted = parse_d(meta["submitted"])
            timely = submitted <= due
            late_question = "late" in row["question"]["instructions"].lower()
            return "yes" if (not timely if late_question else timely) else "no"
        return date_key(due)
    if op == "version":
        qd = parse_d(meta["query_date"])
        return max([v for v, eff in meta["versions"] if parse_d(eff) <= qd], key=lambda v: dict(meta["versions"])[v])
    if op == "age":
        born, asof = parse_d(meta["born"]), parse_d(meta["asof"])
        age = asof.year - born.year - ((asof.month, asof.day) < (born.month, born.day))
        return f"age_{age}"
    if op == "duration":
        minutes = int((datetime.fromisoformat(meta["end"]) - datetime.fromisoformat(meta["start"])).total_seconds() // 60)
        return f"minutes_{minutes}"
    raise AssertionError(op)


def plan_cost(plan: dict, usage: Decimal) -> Decimal:
    cost = Decimal(plan["fixed"])
    remaining = max(Decimal(0), usage - Decimal(plan["included"]))
    if plan["kind"] == "linear":
        cost += remaining * Decimal(plan["rate"])
    elif plan["kind"] == "capped":
        cost += min(remaining * Decimal(plan["rate"]), Decimal(plan["cap"]))
    else:
        block = Decimal(plan["block"])
        blocks = (remaining / block).to_integral_value(rounding=ROUND_HALF_UP)
        if blocks * block < remaining:
            blocks += 1
        cost += blocks * Decimal(plan["block_price"])
    if Decimal(plan["minimum"]) > cost:
        cost = Decimal(plan["minimum"])
    return cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def solve_arithmetic(row: dict) -> str:
    meta = row["meta"]["latent"]
    if meta["operation"] == "plan":
        usage = Decimal(meta["usage"])
        costs = {k: plan_cost(v, usage) for k, v in meta["plans"].items()}
        return min(costs, key=costs.get)
    subtotal = sum(Decimal(l["qty"]) * Decimal(l["unit"]) for l in meta["lines"])
    if meta["discount_type"] == "percent":
        discount = subtotal * Decimal(meta["discount_pct"])
    elif meta["discount_type"] == "fixed":
        discount = min(subtotal, Decimal(meta["fixed_discount"]))
    else:
        discount = subtotal * {"none": Decimal("0"), "silver": Decimal("0.04"), "gold": Decimal("0.09")}[meta["loyalty"]]
    after = subtotal - discount
    taxable = sum(Decimal(l["qty"]) * Decimal(l["unit"]) for l in meta["lines"] if l["taxable"])
    tax_base = max(Decimal(0), taxable - discount * (taxable / subtotal if subtotal else Decimal(0)))
    total = (after + tax_base * Decimal(meta["tax_rate"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if row["question"]["type"] == "noul":
        above = total > Decimal(meta["threshold"])
        within = "within" in row["question"]["instructions"].lower()
        return "yes" if (not above if within else above) else "no"
    return amount_key(total)


def solve_counting(row: dict) -> str:
    meta = row["meta"]["latent"]
    cond = meta["conditions"]
    matching = [r["id"] for r in meta["records"] if r["category"] == cond["category"] and r["segment"] == cond["segment"] and r["amount"] >= cond["min_amount"]]
    if row["meta"]["variant"] == "membership_choice":
        present = [rid for rid in matching if synth_core._snake(rid) in row["question"]["criteria"]]
        return synth_core._snake(present[0]) if present else "none_listed"
    return f"count_{len(matching)}"


def stringify_state(state) -> str:
    return json.dumps(state, sort_keys=True) if isinstance(state, dict) else str(state)


def skeleton(text: str) -> str:
    text = re.sub(r"\d", "#", text.lower())
    people = "|".join(s.lower() for s in synth_core.PEOPLE_FIRST + synth_core.PEOPLE_LAST)
    text = re.sub(rf"\b({people})\b", "N", text)
    text = re.sub(r"\b[A-Z][a-z]+(?:an|ar|en|ia|io|la|li|ma|na|on|or|ra|re|ta|ti|ve)\s+[A-Z][a-z]+\b", "N", text)
    return re.sub(r"\s+", " ", text)


def instruction_template(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d", "#", text)
    text = re.sub(r"\b(" + "|".join(re.escape(d.replace("_", " ")) for d in synth_core.DOMAINS) + r")\b", "DOMAIN", text)
    text = re.sub(r"\b(?:cl|tk|ap|inv|rt|pol|case|ref)-#+\b", "REF", text)
    text = re.sub(r"\b[A-Z][a-z]+\b", "N", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fivegrams(text: str) -> set[tuple[str, ...]]:
    toks = re.findall(r"[a-z0-9_#]+", text.lower())
    return set(zip(*(toks[i:] for i in range(5)))) if len(toks) >= 5 else set()


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def classify_precedence_text(text: str) -> str:
    text = text.lower()
    modes = []
    if "lowest-numbered applicable section controls" in text and "override" not in text:
        modes.append("first_match")
    if "highest-numbered applicable section controls" in text and "later amendments supersede" in text:
        modes.append("last_match")
    if "section with the most conditions controls" in text:
        modes.append("most_specific")
    if ("overrides all other sections" in text or "override all other sections" in text or "no section has override status" in text) and "otherwise the lowest-numbered applicable section controls" in text:
        modes.append("explicit_overrides")
    assert len(modes) == 1, (modes, text)
    return modes[0]


def hidden_values(latent: dict, attr: str) -> list:
    spec = latent.get("attr_specs", {}).get(attr)
    values = {latent["facts"][attr]}
    if isinstance(spec, list):
        values.update(spec)
    elif isinstance(spec, tuple) and spec and spec[0] == "bool":
        values.update([True, False])
    elif isinstance(spec, tuple) and spec and spec[0] == "int":
        low, high = spec[1], spec[2]
        values.update([low, high])
        for rule in latent["rules"]:
            for cond in rule["conditions"]:
                if cond["attr"] == attr and isinstance(cond["value"], int):
                    values.update([cond["value"] - 1, cond["value"], cond["value"] + 1])
        for d in latent.get("definitions", []):
            if d.get("attr") == attr and isinstance(d.get("value"), int):
                values.update([d["value"] - 1, d["value"], d["value"] + 1])
        values = {max(low, min(high, int(v))) for v in values}
    return sorted(values, key=lambda x: str(x))


def completion_outcomes(latent: dict, visible: dict, hidden_attrs: list[str]) -> set[str]:
    domains = [hidden_values(latent, a) for a in hidden_attrs]
    out = set()
    for combo in itertools.product(*domains):
        facts = dict(visible)
        facts.update(dict(zip(hidden_attrs, combo)))
        out.add(solve_rules_with_facts(latent, facts))
    return out


def satisfying_choice_options(row: dict) -> list[str]:
    if row["question"]["type"] != "choice":
        return []
    fam = row["meta"]["family"]
    keys = list(row["question"]["criteria"])
    if fam == "counting_sets":
        meta = row["meta"]["latent"]
        if row["meta"]["variant"] == "membership_choice":
            matching = set(meta["matching_ids"])
            out = []
            for key in keys:
                rec_id = key
                if key == "none_listed":
                    if not any(rid.lower() in keys for rid in matching):
                        out.append(key)
                elif key.upper() or key:
                    if key in {rid.lower() for rid in matching}:
                        out.append(key)
            return out
        return [k for k in keys if k.startswith("count_") and int(k.split("_", 1)[1]) == meta["count"]]
    if fam == "dates":
        return [solve_dates(row)]
    if fam == "arithmetic":
        return [solve_arithmetic(row)]
    if fam == "rules_exceptions":
        return [solve_rules_latent(row["meta"]["latent"])]
    if fam == "routing":
        return [solve_rules_latent(row["meta"]["latent"])]
    if fam == "underdetermined":
        meta = row["meta"]["latent"]
        outcomes = set(meta["completion_outcomes"])
        if len(outcomes) > 1:
            return [meta["uncertainty_key"]]
        return [next(iter(outcomes))]
    if fam == "table_join":
        return [synth_core._snake(row["meta"]["latent"]["answer"])]
    if fam == "event_log":
        lat = row["meta"]["latent"]
        return [synth_core._snake(f"{lat['field']}_{lat['final']}")]
    if fam == "rubric_score":
        return [row["gold"]]
    if fam == "answer_judge":
        return [row["meta"]["latent"]["defect"]] if row["question"]["type"] == "choice" else []
    if fam == "quantifiers":
        return [row["gold"]]
    if fam == "injection":
        return [row["gold"]]
    return [row["gold"]]


def bow_predict_accuracy(rows: list[dict]) -> tuple[float, dict[tuple[str, str, str], tuple[int, float]]]:
    labels = [1 if r["gold"] in {"cannot_determine", "ask_for_clarification"} else 0 for r in rows]
    docs = [re.findall(r"[a-z0-9_]+", stringify_state(r["state"]).lower()) for r in rows]
    split = int(len(rows) * 0.75)
    class_counts = Counter(labels[:split])
    feat_counts = {0: Counter(), 1: Counter()}
    totals = {0: 0, 1: 0}
    vocab = set()
    for toks, y in zip(docs[:split], labels[:split]):
        counts = Counter(toks)
        feat_counts[y].update(counts)
        totals[y] += sum(counts.values())
        vocab.update(counts)
    v = max(1, len(vocab))
    correct = 0
    for toks, y in zip(docs[split:], labels[split:]):
        scores = {}
        for cls in [0, 1]:
            score = __import__("math").log((class_counts[cls] + 1) / (split + 2))
            denom = totals[cls] + v
            for tok in toks:
                score += __import__("math").log((feat_counts[cls][tok] + 1) / denom)
            scores[cls] = score
        pred = 1 if scores[1] > scores[0] else 0
        correct += pred == y
    trigram_stats: dict[tuple[str, str, str], tuple[int, float]] = {}
    base = sum(labels) / len(labels)
    tri_docs = []
    for toks in docs:
        tri_docs.append(set(zip(toks, toks[1:], toks[2:])))
    all_tris = Counter(t for tris in tri_docs for t in tris)
    for tri, support in all_tris.items():
        if support >= 50:
            p = sum(y for y, tris in zip(labels, tri_docs) if tri in tris) / support
            trigram_stats[tri] = (support, abs(p - base))
    return correct / max(1, len(rows) - split), trigram_stats


class SynthCoreTests(unittest.TestCase):
    def test_each_family_generates_valid_balanced_rows(self):
        for family in synth_core.FAMILIES:
            rows = list(synth_core.generate(200, f"seed-{family}", [family]))
            self.assertEqual(len(rows), 200, family)
            self.assertEqual(len({r["id"] for r in rows}), 200, family)
            for row in rows:
                validate_row(row)
                self.assertEqual(row["meta"]["family"], family)
            noul = [r["gold"] for r in rows if r["question"]["type"] == "noul"]
            if len(noul) >= 20:
                c = Counter(noul)
                self.assertGreater(c["yes"], 0, family)
                self.assertGreater(c["no"], 0, family)
                self.assertLess(abs(c["yes"] - c["no"]), len(noul) * 0.75, family)
            positions = [list(r["question"]["criteria"]).index(r["gold"]) for r in rows if r["question"]["type"] == "choice"]
            if len(positions) >= 50:
                self.assertGreaterEqual(len(set(positions)), min(3, max(positions) + 1), family)

    def test_determinism(self):
        self.assertEqual(list(synth_core.generate(120, "same-seed")), list(synth_core.generate(120, "same-seed")))
        self.assertNotEqual(list(synth_core.generate(120, "same-seed")), list(synth_core.generate(120, "other-seed")))

    def test_independent_dates_solver(self):
        for row in synth_core.generate(200, "dates-recompute", ["dates"]):
            self.assertEqual(solve_dates(row), row["gold"], row["id"])

    def test_independent_arithmetic_solver(self):
        for row in synth_core.generate(200, "arith-recompute", ["arithmetic"]):
            self.assertEqual(solve_arithmetic(row), row["gold"], row["id"])

    def test_independent_counting_solver(self):
        for row in synth_core.generate(200, "count-recompute", ["counting_sets"]):
            self.assertEqual(solve_counting(row), row["gold"], row["id"])

    def test_choice_rows_have_exactly_one_satisfying_option(self):
        for family in synth_core.FAMILIES:
            rows = list(synth_core.generate(500, f"unique-choice-{family}", [family]))
            for row in rows:
                if row["question"]["type"] != "choice":
                    continue
                satisfying = satisfying_choice_options(row)
                self.assertEqual(len(satisfying), 1, (row["id"], satisfying, row["gold"]))
                self.assertEqual(satisfying[0], row["gold"], row["id"])

    def test_independent_rules_solver(self):
        for row in synth_core.generate(200, "rules-recompute", ["rules_exceptions"]):
            expected = solve_rules_latent(row["meta"]["latent"])
            if row["question"]["type"] == "choice":
                self.assertEqual(expected, row["gold"], row["id"])
            else:
                true_desc = row["question"]["criteria"]["true"].replace(" applies", "")
                expected_desc = row["meta"]["latent"]["outcomes"][expected]
                self.assertEqual(("yes" if true_desc == expected_desc else "no"), row["gold"], row["id"])

    def test_independent_routing_solver(self):
        for row in synth_core.generate(200, "routing-recompute", ["routing"]):
            self.assertEqual(solve_rules_latent(row["meta"]["latent"]), row["gold"], row["id"])

    def test_precedence_text_matches_evaluator_mode(self):
        for row in synth_core.generate(300, "precedence-text", ["rules_exceptions"]):
            text = stringify_state(row["state"])
            self.assertEqual(classify_precedence_text(text), row["meta"]["latent"]["precedence"], row["id"])
        for row in synth_core.generate(300, "precedence-under", ["underdetermined"]):
            text = stringify_state(row["state"])
            self.assertEqual(classify_precedence_text(text), row["meta"]["latent"]["latent_rules"]["precedence"], row["id"])

    def test_underdetermined_does_not_leak_hidden_values(self):
        rows = list(synth_core.generate(300, "under-leak", ["underdetermined"]))
        blurred = [r for r in rows if not r["meta"]["latent"]["control"]]
        self.assertGreater(len(blurred), 50)
        for row in blurred:
            state = stringify_state(row["state"]).lower()
            self.assertNotIn("formerly", state)
            case_state = state.split("case file", 1)[-1]
            lat = row["meta"]["latent"]
            original = lat["latent_rules"]["facts"]
            for attr in lat["hidden_attrs"]:
                value = str(original[attr]).lower()
                attr_text = attr.replace("_", " ").lower()
                snake_patterns = [f"{attr.lower()}: {value}", f"{attr.lower()} as {value}", f"{attr.lower()}={value}"]
                text_patterns = [f"{attr_text}: {value}", f"{attr_text} as {value}", f"{attr_text} is {value}", f"{attr_text} was reported as {value}"]
                for pat in snake_patterns + text_patterns:
                    self.assertNotIn(pat, case_state, (row["id"], attr, value, pat))

    def test_underdetermined_enumeration_labels(self):
        rows = list(synth_core.generate(500, "under-enum", ["underdetermined"]))
        for row in rows:
            meta = row["meta"]["latent"]
            latent = meta["latent_rules"]
            outcomes = completion_outcomes(latent, meta["visible_facts"], meta["hidden_attrs"])
            self.assertEqual(set(meta["completion_outcomes"]), outcomes, row["id"])
            uncertainty_key = meta["uncertainty_key"]
            if len(outcomes) > 1:
                self.assertEqual(row["gold"], uncertainty_key, row["id"])
                self.assertFalse(meta["control"], row["id"])
            else:
                self.assertEqual(row["gold"], next(iter(outcomes)), row["id"])

    def test_underdetermined_surface_leakage_resistance(self):
        rows = list(synth_core.generate(2_000, "under-surface-leakage", ["underdetermined"]))
        acc, trigram_stats = bow_predict_accuracy(rows)
        self.assertLessEqual(acc, 0.60)
        leaky = [(tri, support, delta) for tri, (support, delta) in trigram_stats.items() if delta > 0.25]
        self.assertFalse(leaky[:10], leaky[:10])

    def test_definitions_affect_evaluation_when_present(self):
        rows = list(synth_core.generate(600, "definition-matter", ["rules_exceptions"]))
        with_defs = [r for r in rows if r["meta"]["latent"].get("definitions")]
        self.assertGreater(len(with_defs), 100)
        changed = 0
        for row in with_defs:
            latent = row["meta"]["latent"]
            before = solve_rules_latent(latent)
            mutated = dict(latent)
            mutated["definitions"] = [dict(d) for d in latent["definitions"]]
            for d in mutated["definitions"]:
                if isinstance(d.get("value"), int):
                    d["value"] = d["value"] + 10_000 if d["op"] in {"gt", "ge"} else d["value"] - 10_000
                elif isinstance(d.get("value"), list):
                    d["value"] = ["__no_such_value__"]
            if solve_rules_latent(mutated) != before:
                changed += 1
        self.assertGreaterEqual(changed / len(with_defs), 0.30)

    def test_diversity_per_family(self):
        rng = random.Random(12345)
        for family in synth_core.FAMILIES:
            rows = list(synth_core.generate(500, f"diversity-{family}", [family]))
            states = [stringify_state(r["state"]) for r in rows]
            skels = [skeleton(s) for s in states]
            unique_frac = len(set(skels)) / len(skels)
            distinct_instr = len({instruction_template(r["question"]["instructions"]) for r in rows})
            grams = [fivegrams(s) for s in skels]
            pairs = [(rng.randrange(len(grams)), rng.randrange(len(grams))) for _ in range(160)]
            sims = [jaccard(grams[i], grams[j]) for i, j in pairs if i != j]
            mean_sim = sum(sims) / len(sims)
            DIVERSITY_STATS[family] = {"unique_skeleton_fraction": unique_frac, "distinct_instructions": distinct_instr, "mean_5gram_jaccard": mean_sim}
            self.assertGreaterEqual(unique_frac, 0.9, family)
            self.assertGreaterEqual(distinct_instr, 12, family)
            self.assertLess(mean_sim, 0.25, family)

    def test_cli_writes_jsonl(self):
        out = ROOT / "tests" / "_tmp_synth_core.jsonl"
        try:
            rc = synth_core._main(["synth_core", str(out), "24", "cli-seed", "dates,arithmetic"])
            self.assertEqual(rc, 0)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 24)
            for row in rows:
                validate_row(row)
        finally:
            out.unlink(missing_ok=True)

    def test_large_generation_smoke(self):
        counts = Counter()
        for row in synth_core.generate(50_000, "core-train-v4"):
            counts[row["meta"]["family"]] += 1
        self.assertEqual(sum(counts.values()), 50_000)
        self.assertEqual(set(counts), set(synth_core.FAMILIES))
        self.assertIsInstance(getattr(synth_core.generate, "last_skipped", Counter()), Counter)


def tearDownModule():
    print("\nDIVERSITY_STATS")
    for fam in sorted(DIVERSITY_STATS):
        print(f"{fam}: {DIVERSITY_STATS[fam]}")
    print("\nEXAMPLES_BY_FAMILY")
    for fam in synth_core.FAMILIES:
        row = next(synth_core.generate(1, f"example-{fam}", [fam]))
        s = json.dumps(row, sort_keys=True)
        print(f"{fam}: {s[:800]}{'...' if len(s) > 800 else ''}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
