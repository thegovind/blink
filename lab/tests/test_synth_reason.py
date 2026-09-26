from __future__ import annotations

import ast
import json
import re
import sys
import unittest
from fractions import Fraction
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevlab import synth_reason as sr  # noqa: E402


def literal_value(text):
    return ast.literal_eval(text)


def validate_row(row):
    assert set(row) == {"id", "src", "state", "question", "gold", "target", "meta"}
    assert row["target"] is None
    assert row["question"]["type"] == "choice"
    criteria = row["question"]["criteria"]
    assert row["gold"] in criteria
    values = list(criteria.values())
    if row["meta"]["family"] == "logic_grid":
        assert len(values) in (2, 3, 4)
    else:
        assert len(values) in (2, 4, 5, 6, 8, 10)
    if row["meta"]["family"] == "code_exec" and row["meta"].get("kind") == "output_prediction":
        keys = [repr(literal_value(v)) for v in values]
        assert len(keys) == len(set(keys))
    else:
        assert len(values) == len(set(values))


def parents(graph, var):
    return [a for a, bs in graph.items() if var in bs]


def p_var(cpt, ps, assignment):
    key = "".join("1" if assignment[p] else "0" for p in ps)
    n, d = cpt[key]
    return Fraction(n, d)


def joint(scm, intervention=None):
    intervention = intervention or {}
    rows = []
    for bits in product([0, 1], repeat=len(scm["variables"])):
        a = dict(zip(scm["variables"], bits))
        p = Fraction(1)
        ok = True
        for v in scm["variables"]:
            if v in intervention:
                ok = a[v] == intervention[v]
                if not ok:
                    break
                continue
            pv = p_var(scm["cpts"][v], parents(scm["graph"], v), a)
            p *= pv if a[v] else 1 - pv
        if ok:
            rows.append((a, p))
    z = sum(p for _, p in rows)
    return [(a, p / z) for a, p in rows] if z else []


def prob(scm, event, given=None, intervention=None):
    n = d = Fraction(0)
    for a, p in joint(scm, intervention):
        if given is None or all(a[k] == v for k, v in given.items()):
            d += p
            if all(a[k] == v for k, v in event.items()):
                n += p
    return n / d if d else Fraction(0)


def descendants(graph, x):
    out, stack = set(), list(graph.get(x, []))
    while stack:
        v = stack.pop()
        if v in out:
            continue
        out.add(v)
        stack.extend(graph.get(v, []))
    return out


def arrow(graph, a, b):
    return b in graph.get(a, [])


def undirected_paths(graph, x, y):
    nbrs = {v: set(graph.get(v, [])) | {a for a, bs in graph.items() if v in bs} for v in graph}
    stack = [(x, [x])]
    out = []
    while stack:
        v, path = stack.pop()
        if v == y:
            out.append(path)
            continue
        for n in nbrs[v]:
            if n not in path:
                stack.append((n, path + [n]))
    return out


def path_active(graph, path, zset):
    zset = set(zset)
    for j in range(1, len(path) - 1):
        a, b, c = path[j - 1], path[j], path[j + 1]
        collider = arrow(graph, a, b) and arrow(graph, c, b)
        if collider:
            if b not in zset and not (descendants(graph, b) & zset):
                return False
        elif b in zset:
            return False
    return True


def d_separated(graph, x, y, zset):
    return not any(path_active(graph, p, zset) for p in undirected_paths(graph, x, y))


def independent_backdoor_sufficient(scm, x, y, zset):
    if any(z in descendants(scm["graph"], x) for z in zset):
        return False
    mutilated = {v: [c for c in cs if v != x] for v, cs in scm["graph"].items()}
    return d_separated(mutilated, x, y, zset)


def recompute_causal(row):
    meta = row["meta"]
    scm = meta["scm"]
    kind = meta["query_kind"]
    if kind == "structure":
        role = meta["role"]
        z = meta["variable"]
        g = scm["graph"]
        if role == "confounder":
            xs = g.get(z, [])
            return len(xs) >= 2
        if role == "mediator":
            return any(z in g.get(a, []) and g.get(z) for a in g)
        if role == "collider":
            return len(parents(g, z)) >= 2
    if kind == "adjustment":
        x, y = meta["x"], meta["y"]
        adj = sr._adjustment_set(scm, x, y)
        return "{" + ", ".join(sorted(adj)) + "}" if adj else "{}"
    x, y = meta["x"], meta["y"]
    if kind == "marginal":
        return prob(scm, {y: 1}) - Fraction(1, 2)
    if kind == "association":
        return prob(scm, {y: 1}, {x: 1}) - prob(scm, {y: 1}, {x: 0})
    if kind == "ate":
        return prob(scm, {y: 1}, intervention={x: 1}) - prob(scm, {y: 1}, intervention={x: 0})
    if kind == "ett":
        adj = sr._adjustment_set(scm, x, y)
        if not adj:
            return prob(scm, {y: 1}, {x: 1}) - prob(scm, {y: 1}, {x: 0})
        q = Fraction(0)
        for bits in product([0, 1], repeat=len(adj)):
            z = dict(zip(adj, bits))
            q += (prob(scm, {y: 1}, {x: 1, **z}) - prob(scm, {y: 1}, {x: 0, **z})) * prob(scm, z, {x: 1})
        return q
    if kind == "collider_bias":
        return len(parents(scm["graph"], meta["variable"])) >= 2
    m = meta["mediator"]
    nie = nde = Fraction(0)
    for mv in [0, 1]:
        py1m = prob(scm, {y: 1}, {x: 1, m: mv})
        py0m = prob(scm, {y: 1}, {x: 0, m: mv})
        pm1 = prob(scm, {m: mv}, intervention={x: 1})
        pm0 = prob(scm, {m: mv}, intervention={x: 0})
        nie += py1m * (pm1 - pm0)
        nde += (py1m - py0m) * pm0
    return nie if meta["effect"] == "nie" else nde


def recompute_word(latent):
    vals = {}
    for step in latent["steps"]:
        args = [vals[a] if isinstance(a, str) else Fraction(*a) for a in step["args"]]
        if step["op"] == "add":
            val = args[0] + args[1]
        elif step["op"] == "sub":
            val = args[0] - args[1]
        elif step["op"] == "mul":
            val = args[0] * args[1]
        elif step["op"] == "div":
            val = args[0] / args[1]
        elif step["op"] == "pct":
            val = args[0] * args[1] / 100
        elif step["op"] == "round_cents":
            val = Fraction(int(round(float(args[0]) * 100)), 100)
        else:
            raise AssertionError(step["op"])
        self_value = Fraction(*step["value"])
        assert val == self_value
        vals[step["label"]] = val
    return vals["answer"]


def parse_num(s):
    return Fraction(s)


def skeleton(row):
    txt = json.dumps({"s": row["state"], "q": row["question"]["instructions"]}, sort_keys=True)
    for name in sr.NAMES + ["Ada", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus"]:
        txt = re.sub(rf"\b{re.escape(name)}\b", "N", txt)
    txt = re.sub(r"\d+(?:\.\d+)?", "#", txt)
    return txt


def _set_from_meta(values):
    return "{" + ", ".join(sorted(values)) + "}" if values else "{}"


class SynthReasonTests(unittest.TestCase):
    def test_300_rows_per_family_format_and_determinism(self):
        for fam in sr.FAMILIES:
            rows = sr.generate(300, "unit", families=[fam])
            rows2 = sr.generate(300, "unit", families=[fam])
            self.assertEqual(rows, rows2)
            self.assertEqual(len(rows), 300)
            for row in rows:
                validate_row(row)
                self.assertEqual(row["src"], f"reason_{fam}")
                self.assertEqual(row["meta"]["family"], fam)

    def test_code_exec_labels(self):
        rows = sr.generate(300, "code", families=["code_exec"])
        saw_input_prediction = False
        for row in rows:
            state = row["state"]
            self.assertEqual(set(state), {"code", "input", "task"})
            criteria = row["question"]["criteria"]
            if row["meta"].get("kind") == "input_prediction":
                saw_input_prediction = True
                observed = row["meta"]["observed_output"]
                correct = 0
                for key, call in criteria.items():
                    got = repr(sr.safe_call(state["code"], call))
                    self.assertEqual(got == observed, key == row["gold"])
                    correct += got == observed
                self.assertEqual(correct, 1)
            else:
                got = repr(sr.safe_call(state["code"], state["input"]))
                self.assertEqual(criteria[row["gold"]], got)
                for key, value in criteria.items():
                    if key != row["gold"]:
                        self.assertNotEqual(literal_value(value), literal_value(got))
        self.assertTrue(saw_input_prediction)

    def test_causal_recompute(self):
        rows = sr.generate(300, "causal", families=["causal"])
        yes = no = 0
        for row in rows:
            crit = row["question"]["criteria"]
            actual = recompute_causal(row)
            if isinstance(actual, bool):
                expected = "yes" if actual else "no"
            elif isinstance(actual, str):
                expected = next(k for k, v in crit.items() if v == actual)
            else:
                direction = row["meta"].get("direction", 1)
                expected = "yes" if actual * direction > 0 else "no"
                qn, qd = row["meta"]["quantity"]
                self.assertEqual(actual, Fraction(qn, qd))
            self.assertEqual(row["gold"], expected)
            yes += row["gold"] == "yes"
            no += row["gold"] == "no"
        self.assertGreater(yes, 30)
        self.assertGreater(no, 30)

    def test_word_math_recompute_and_forbidden_fraction(self):
        rows = sr.generate(500, "word", families=["word_math"])
        bad = total = neg_bad = type_bad = range_bad = 0
        saw_k = set()
        for row in rows:
            latent = row["meta"]["latent"]
            ans = recompute_word(latent)
            self.assertEqual(ans, Fraction(*latent["answer"]))
            self.assertGreaterEqual(len(latent["steps"]), 3)
            self.assertIn(latent["answer_type"], {"integer", "money"})
            if latent["answer_type"] == "integer":
                self.assertEqual(ans.denominator, 1)
            if latent["answer_type"] == "money":
                self.assertEqual((ans * 100).denominator, 1)
            gold_text = sr._fmt_num(ans, latent["answer_type"])
            self.assertEqual(row["question"]["criteria"][row["gold"]], gold_text)
            saw_k.add(len(row["question"]["criteria"]))
            forbidden = {ans + x for x in [1, 2, 5, 10]} | {ans - x for x in [1, 2, 5, 10]} | {ans * 2, ans * 10, ans // 2}
            if ans >= 20:
                lo, hi = ans / 4, ans * 4
            elif ans > 0:
                lo, hi = ans / 2, ans * Fraction(3, 2)
            else:
                lo, hi = Fraction(0), Fraction(20)
            for key, text in row["question"]["criteria"].items():
                if key == row["gold"]:
                    continue
                val = parse_num(text)
                total += 1
                bad += val in forbidden
                neg_bad += ans >= 0 and val < 0
                type_bad += (latent["answer_type"] == "integer" and val.denominator != 1)
                type_bad += (latent["answer_type"] == "money" and (val * 100).denominator != 1)
                range_bad += not (lo <= val <= hi)
        self.assertEqual(saw_k, {4, 10})
        self.assertLess(bad / total, 0.10)
        self.assertEqual(neg_bad, 0)
        self.assertEqual(type_bad, 0)
        self.assertEqual(range_bad, 0)

    def test_logic_grid_unique(self):
        rows = sr.generate(300, "logic", families=["logic_grid"])
        saw_unknown = 0
        for row in rows:
            sols = sr.solve_logic(row["meta"])
            if row["meta"].get("determined"):
                self.assertEqual(len(sols), 1)
            else:
                self.assertGreater(len(sols), 1)
                self.assertEqual(row["question"]["criteria"][row["gold"]], "Cannot be determined")
                saw_unknown += 1
                continue
            q = row["question"]["instructions"]
            for c in row["meta"]["clues"]:
                self.assertNotEqual(c["type"], "is")
            self.assertGreaterEqual(sum(c["type"] in {"same", "left_entity_attr", "left_attr_attr", "attr_pos"} for c in row["meta"]["clues"]), 2)
            sol = sols[0]
            if row["question"]["criteria"] == sr.YESNO:
                text = row["question"]["instructions"]
                truth = row["gold"] == "yes"
                self.assertIsInstance(truth, bool)
                self.assertIn("is this statement true", text)
            else:
                wanted = row["question"]["criteria"][row["gold"]]
                self.assertIn(wanted, sol)
                if "which entity has" in q:
                    attr_val = q.split("which entity has ", 1)[1]
                    value, attr = attr_val.split(" as their ", 1)
                    attr = attr.rstrip("?")
                    count = sum(sol[e][attr] == value for e in row["meta"]["entities"])
                    self.assertEqual(count, 1)
        self.assertGreater(saw_unknown, 20)

    def test_causal_adjustment_options_and_mismatch(self):
        rows = sr.generate(500, "causal_more", families=["causal"])
        intervention = mismatch = 0
        instruction_strings = set()
        for row in rows:
            instruction_strings.add(row["question"]["instructions"])
            if row["meta"]["query_kind"] == "adjustment":
                vars_ = set(row["meta"]["scm"]["variables"])
                sufficient = 0
                for text in row["question"]["criteria"].values():
                    inside = text.strip("{}")
                    members = set() if not inside else {x.strip() for x in inside.split(",")}
                    self.assertLessEqual(members, vars_)
                    sufficient += independent_backdoor_sufficient(row["meta"]["scm"], row["meta"]["x"], row["meta"]["y"], members)
                self.assertEqual(row["question"]["criteria"][row["gold"]], _set_from_meta(row["meta"]["adjustment_set"]))
                self.assertEqual(sufficient, 1)
            if row["meta"]["query_kind"] == "ate":
                intervention += 1
                mismatch += bool(row["meta"].get("naive_differs"))
        self.assertGreaterEqual(len(instruction_strings), 20)
        self.assertGreaterEqual(mismatch / intervention, 0.40)

    def test_diversity(self):
        for fam, threshold in [("word_math", 0.9), ("causal", 0.8), ("logic_grid", 0.8)]:
            rows = sr.generate(500, f"div-{fam}", families=[fam])
            ratio = len({skeleton(r) for r in rows}) / len(rows)
            self.assertGreaterEqual(ratio, threshold, (fam, ratio))

    def test_group_semantics(self):
        code_rows = sr.generate(120, "groups", families=["code_exec"])
        code_by_group = {}
        for row in code_rows:
            code_by_group.setdefault(row["meta"]["group"], row["state"]["code"])
            self.assertEqual(code_by_group[row["meta"]["group"]], row["state"]["code"])
        self.assertGreater(len(code_by_group), 100)
        self.assertNotIn("code_exec:groups", code_by_group)

        causal_rows = sr.generate(80, "groups", families=["causal"])
        scm_by_group = {}
        kinds_by_group = {}
        for row in causal_rows:
            g = row["meta"]["group"]
            scm_by_group.setdefault(g, row["meta"]["scm"])
            self.assertEqual(scm_by_group[g], row["meta"]["scm"])
            kinds_by_group.setdefault(g, set()).add(row["meta"]["query_kind"])
        self.assertTrue(any(len(v) > 1 for v in kinds_by_group.values()))

        logic_rows = sr.generate(40, "groups", families=["logic_grid"])
        puzzle_by_group = {}
        for row in logic_rows:
            g = row["meta"]["group"]
            puzzle = (row["meta"]["entities"], row["meta"]["attributes"], row["meta"]["solution"])
            puzzle_by_group.setdefault(g, puzzle)
            self.assertEqual(puzzle_by_group[g], puzzle)
        self.assertTrue(any(sum(r["meta"]["group"] == g for r in logic_rows) > 1 for g in puzzle_by_group))

        word_rows = sr.generate(80, "groups", families=["word_math"])
        self.assertEqual(len({r["meta"]["group"] for r in word_rows}), len(word_rows))
        for row in word_rows:
            self.assertIn("template", row["meta"])
            self.assertNotEqual(row["meta"]["group"], f"word_math:{row['meta']['template']}")

    def test_cli_jsonl(self):
        out = ROOT / "tests" / "_tmp_synth_reason.jsonl"
        try:
            sr.main([str(out), "12", "cli", "code_exec,causal,word_math,logic_grid"])
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows), 12)
            for row in rows:
                validate_row(row)
        finally:
            out.unlink(missing_ok=True)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SynthReasonTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("\nExamples:")
    for fam in sr.FAMILIES:
        row = sr.generate(1, "example", families=[fam])[0]
        text = json.dumps(row, indent=2, sort_keys=True)
        print(f"\n[{fam}]\n{text[:1200]}{'...' if len(text) > 1200 else ''}")
    raise SystemExit(0 if result.wasSuccessful() else 1)
