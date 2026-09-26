"""Programmatic reasoning-decision item generators.

The rows emitted here are intended for multiple-choice decision training.  All
labels are computed by small exact programs; no model calls, network access, or
non-stdlib dependencies are used.

CLI:
  cd lab && python -m jevlab.synth_reason OUT.jsonl N SEED [families]
"""
from __future__ import annotations

import ast
import json
import random
import sys
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path


YESNO = {"yes": "Yes", "no": "No"}
NAMES = [
    "Ari", "Bex", "Cato", "Dina", "Enzo", "Fia", "Gus", "Hale", "Ira",
    "Juno", "Kira", "Lio", "Mina", "Nico", "Ona", "Pax", "Quin", "Rhea",
]


def _stable_seed(seed: object, family: str, i: int) -> str:
    return f"{seed!r}:{family}:{i}"


def _shuffle_options(rng: random.Random, values: list[str], gold_value: str, prefix: str = "option_"):
    vals = list(values)
    rng.shuffle(vals)
    criteria = {f"{prefix}{i}": v for i, v in enumerate(vals)}
    for k, v in criteria.items():
        if v == gold_value:
            return criteria, k
    raise AssertionError("gold value missing")


def _literal_key(text: str):
    def freeze(x):
        if isinstance(x, list):
            return ("list", tuple(freeze(v) for v in x))
        if isinstance(x, tuple):
            return ("tuple", tuple(freeze(v) for v in x))
        if isinstance(x, dict):
            return ("dict", tuple(sorted((freeze(k), freeze(v)) for k, v in x.items())))
        if isinstance(x, set):
            return ("set", tuple(sorted(freeze(v) for v in x)))
        return x

    try:
        return ("lit", freeze(ast.literal_eval(text)))
    except Exception:
        return ("raw", text)


def _distinct_repr(values):
    out, seen = [], set()
    for v in values:
        r = repr(v)
        key = _literal_key(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# code_exec

SAFE_BUILTINS = {
    "len": len,
    "range": range,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "abs": abs,
    "reversed": reversed,
}


class StepLimit(RuntimeError):
    pass


def safe_call(source: str, call: str, limit: int = 10_000):
    """Compile and execute generated source with a trace step budget."""
    code = compile(source + "\n_result = " + call, "<generated-code-exec>", "exec")
    glb = {"__builtins__": SAFE_BUILTINS}
    steps = 0

    def tracer(frame, event, arg):
        nonlocal steps
        if event == "line":
            steps += 1
            if steps > limit:
                raise StepLimit("generated program exceeded step limit")
        return tracer

    old = sys.gettrace()
    try:
        sys.settrace(tracer)
        exec(code, glb, glb)
    finally:
        sys.settrace(old)
    return glb["_result"]


def _code_templates(rng: random.Random):
    """Return (source, call, mutants, perturbed_calls, parameterized_calls)."""
    choice = rng.randrange(10)
    if choice == 0:
        text = rng.choice(["delta-river", "paper-lantern", "north-star", "cedar-box"])
        n = rng.randint(1, 4)
        src = """def f(text, n):
    parts = text.split("-")
    out = []
    for i, part in enumerate(parts):
        chunk = part[:n]
        if i % 2:
            chunk = chunk.upper()
        else:
            chunk = chunk[::-1]
        out.append(f"{i}:{chunk}")
    return "|".join(out)"""
        muts = [
            src.replace("part[:n]", "part[:n+1]"),
            src.replace("if i % 2:", "if not i % 2:"),
            src.replace("parts = text.split(\"-\")", "parts = list(reversed(text.split(\"-\")))"),
            src.replace("chunk = part[::-1]", "chunk = part"),
        ]
        calls = [f"f({text!r}, {n})", f"f({text!r}, {n+1})", f"f({text.replace('-', '--')!r}, {n})"]
        return src, calls[0], muts, calls[1:], calls
    if choice == 1:
        nums = [rng.randint(-5, 12) for _ in range(rng.randint(5, 8))]
        src = """def f(nums):
    total = 0
    kept = []
    for i, x in enumerate(nums):
        if x < 0:
            continue
        if x % 2 == i % 2:
            kept.append(x + i)
        else:
            kept.insert(0, x - i)
        total += kept[-1]
    return (kept, total)"""
        muts = [
            src.replace("x % 2 == i % 2", "x % 2 != i % 2"),
            src.replace("kept.insert(0, x - i)", "kept.append(x - i)"),
            src.replace("total += kept[-1]", "total += x"),
            src.replace("if x < 0:", "if x <= 0:"),
        ]
        calls = [f"f({nums!r})", f"f({(nums + [rng.randint(0, 5)])!r})", f"f({list(reversed(nums))!r})"]
        return src, calls[0], muts, calls[1:], calls
    if choice == 2:
        words = rng.sample(["iris", "mint", "cedar", "opal", "linen", "basalt", "plum"], 5)
        src = """def f(words):
    counts = {}
    for word in words:
        key = (word[0], len(word) % 3)
        counts[key] = counts.get(key, 0) + 1
    pairs = sorted(counts.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    return [f"{a}{b}:{v}" for ((a, b), v) in pairs]"""
        muts = [
            src.replace("len(word) % 3", "len(word) % 2"),
            src.replace("counts.get(key, 0) + 1", "counts.get(key, 1) + 1"),
            src.replace("sorted(counts.items(), key=lambda kv: (kv[0][1], kv[0][0]))", "counts.items()"),
            src.replace("word[0]", "word[-1]"),
        ]
        calls = [f"f({words!r})", f"f({(words + [words[0]])!r})", f"f({list(reversed(words))!r})"]
        return src, calls[0], muts, calls[1:], calls
    if choice == 3:
        a = rng.randint(9, 50)
        b = rng.randint(2, 9)
        src = """def f(start, step):
    seen = []
    value = start
    while value > 0:
        rem = value % step
        seen.append((value // step, rem))
        value = value // 2 - rem
        if len(seen) >= 6:
            break
    return seen"""
        muts = [
            src.replace("value > 0", "value > 1"),
            src.replace("value // 2 - rem", "value // 2 + rem"),
            src.replace("len(seen) >= 6", "len(seen) > 6"),
            src.replace("value % step", "(value + 1) % step"),
        ]
        calls = [f"f({a}, {b})", f"f({a+1}, {b})", f"f({a}, {max(2, b-1)})"]
        return src, calls[0], muts, calls[1:], calls
    if choice == 4:
        items = rng.sample(["red", "reed", "blue", "glue", "green", "grain", "gold"], 5)
        mark = rng.choice(["e", "r", "l"])
        src = """def f(items, mark):
    found = set()
    out = []
    for item in items:
        if mark not in item:
            out.append(item.upper())
            continue
        found.add(item[-1])
        out.append(item.replace(mark, "*", 1))
    return (tuple(out), "".join(sorted(found)))"""
        muts = [
            src.replace("replace(mark, \"*\", 1)", "replace(mark, \"*\")"),
            src.replace("item[-1]", "item[0]"),
            src.replace("sorted(found)", "found"),
            src.replace("if mark not in item:", "if mark in item:"),
        ]
        calls = [f"f({items!r}, {mark!r})", f"f({items!r}, {'z'!r})", f"f({list(reversed(items))!r}, {mark!r})"]
        return src, calls[0], muts, calls[1:], calls
    if choice == 5:
        left = [rng.randint(1, 9) for _ in range(4)]
        right = [rng.randint(1, 9) for _ in range(4)]
        src = """def f(left, right):
    acc = []
    for a, b in zip(left, right):
        v = a * b
        if v % 3 == 0:
            acc.append(v // 3)
        else:
            acc.append(v % 5)
    acc.sort(key=lambda x: (x % 2, -x))
    return acc"""
        muts = [
            src.replace("a * b", "a + b"),
            src.replace("v % 3 == 0", "v % 3 != 0"),
            src.replace("acc.sort(key=lambda x: (x % 2, -x))", "acc.sort()"),
            src.replace("v % 5", "v % 4"),
        ]
        calls = [f"f({left!r}, {right!r})", f"f({right!r}, {left!r})", f"f({left[:-1]!r}, {right!r})"]
        return src, calls[0], muts, calls[1:], calls
    if choice == 6:
        s = rng.choice(["aabccdde", "mississippi", "committee", "bookkeeper"])
        src = """def f(text):
    runs = []
    last = None
    count = 0
    for ch in text:
        if ch == last:
            count += 1
        else:
            if last is not None:
                runs.append((last, count))
            last = ch
            count = 1
    runs.append((last, count))
    return "-".join(f"{c}{n}" for c, n in runs if n > 1)"""
        muts = [
            src.replace("if n > 1", "if n >= 1"),
            src.replace("count = 1", "count = 0"),
            src.replace("runs.append((last, count))\n    return", "return"),
            src.replace("ch == last", "ch != last"),
        ]
        calls = [f"f({s!r})", f"f({(s + s[-1])!r})", f"f({s[::-1]!r})"]
        return src, calls[0], muts, calls[1:], calls
    if choice == 7:
        nums = [rng.randint(0, 6) for _ in range(6)]
        src = """def f(nums):
    def score(x):
        return x ** 2 - 3 * x
    best = []
    for x in nums:
        y = score(x)
        if y <= 0:
            best.append((x, y))
    return max(best, key=lambda p: (p[1], -p[0]))"""
        muts = [
            src.replace("x ** 2 - 3 * x", "x ** 2 + 3 * x"),
            src.replace("y <= 0", "y < 0"),
            src.replace("max(best", "min(best"),
            src.replace("(p[1], -p[0])", "(p[1], p[0])"),
        ]
        calls = [f"f({nums!r})", f"f({(nums + [7])!r})", f"f({list(reversed(nums))!r})"]
        return src, calls[0], muts, calls[1:], calls
    if choice == 8:
        rows = [(rng.choice(["a", "b", "c"]), rng.randint(1, 5)) for _ in range(5)]
        src = """def f(rows):
    totals = {}
    for name, amount in rows:
        if amount % 2:
            totals[name] = totals.get(name, 0) + amount
        else:
            totals[name] = totals.get(name, 0) - amount
    return tuple(sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])))"""
        muts = [
            src.replace("amount % 2", "amount % 2 == 0"),
            src.replace("+ amount", "+ 1"),
            src.replace("- amount", "+ amount"),
            src.replace("(-kv[1], kv[0])", "(kv[0], kv[1])"),
        ]
        calls = [f"f({rows!r})", f"f({(rows + [('a', 2)])!r})", f"f({list(reversed(rows))!r})"]
        return src, calls[0], muts, calls[1:], calls
    text = rng.choice(["alpha beta gamma", "red blue red", "one two three two"])
    src = """def f(text):
    words = text.split()
    out = []
    for i in range(len(words) - 1, -1, -1):
        w = words[i]
        if i == 1:
            out.append(w.title())
            continue
        out.append(w[:2] + str(len(w)))
    return "/".join(out)"""
    muts = [
        src.replace("range(len(words) - 1, -1, -1)", "range(len(words))"),
        src.replace("if i == 1:", "if i <= 1:"),
        src.replace("w[:2]", "w[-2:]"),
        src.replace("continue", "break"),
    ]
    calls = [f"f({text!r})", f"f({(text + ' end')!r})", f"f({text.upper()!r})"]
    return src, calls[0], muts, calls[1:], calls


def _mutate_call_literals(call: str):
    try:
        expr = ast.parse(call, mode="eval").body
        args = [ast.literal_eval(a) for a in expr.args]
    except Exception:
        return []
    calls = []
    for idx, val in enumerate(args):
        variants = []
        if isinstance(val, int):
            variants = [val + 1, val - 1 if val > 0 else val + 2, max(0, val * 2)]
        elif isinstance(val, str):
            variants = [val + "x", val[::-1], val.upper()]
        elif isinstance(val, list):
            variants = [list(reversed(val)), val + val[:1], val[1:] if len(val) > 1 else val + [0]]
        elif isinstance(val, tuple):
            variants = [tuple(reversed(val)), val + val[:1]]
        for nv in variants:
            new_args = list(args)
            new_args[idx] = nv
            calls.append("f(" + ", ".join(repr(a) for a in new_args) + ")")
    return calls


def _code_item(seed: object, i: int):
    rng = random.Random(_stable_seed(seed, "code_exec", i))
    for _ in range(50):
        src, call, mutants, perturbed, candidate_calls = _code_templates(rng)
        try:
            gold_val = safe_call(src, call)
        except Exception:
            continue
        if i % 5 == 0:
            observed = repr(gold_val)
            candidates = []
            seen_outputs = set()
            for c in list(dict.fromkeys(candidate_calls + perturbed + _mutate_call_literals(call))):
                try:
                    out = safe_call(src, c)
                except Exception:
                    continue
                key = _literal_key(repr(out))
                if key in seen_outputs:
                    continue
                seen_outputs.add(key)
                candidates.append(c)
            more = list(_mutate_call_literals(call))
            rng.shuffle(more)
            for c in more:
                if len(candidates) >= rng.choice([4, 5, 6]):
                    break
                try:
                    out = safe_call(src, c)
                except Exception:
                    continue
                if repr(out) != observed:
                    candidates.append(c)
            candidates = list(dict.fromkeys(candidates))
            if call not in candidates or len(candidates) < 4:
                continue
            criteria, gold = _shuffle_options(rng, candidates[: rng.choice([4, 5, 6])], call)
            state = {"code": src, "input": call, "task": "input prediction"}
            return _row("code_exec", seed, i, state,
                        "Choose the input call that produces the supplied output from f. "
                        f"The supplied output is {observed}. Candidates are Python call literals.",
                        criteria, gold, {"group": f"code_exec:{seed}:{i}", "kind": "input_prediction", "observed_output": observed})
        vals = [gold_val]
        for m in mutants:
            try:
                vals.append(safe_call(m, call))
            except Exception:
                pass
        for c in perturbed:
            try:
                vals.append(safe_call(src, c))
            except Exception:
                pass
        reps = _distinct_repr(vals)
        if repr(gold_val) not in reps or len(reps) < 4:
            continue
        extra_literals = [[], (), "", 0, 1, False, None]
        reps += [repr(x) for x in extra_literals]
        unique = []
        seen = set()
        for r in reps:
            k = _literal_key(r)
            if k not in seen:
                seen.add(k)
                unique.append(r)
        k = rng.choice([4, 5, 6, 8, 10])
        selected = [repr(gold_val)] + [r for r in unique if r != repr(gold_val)][: k - 1]
        if len(selected) < k:
            continue
        criteria, gold = _shuffle_options(rng, selected, repr(gold_val))
        state = {"code": src, "input": call, "task": "output prediction"}
        return _row("code_exec", seed, i, state,
                    "Choose the correct output of f called with the supplied input arguments. "
                    "Candidates are Python literal values.",
                    criteria, gold, {"group": f"code_exec:{seed}:{i}", "kind": "output_prediction"})
    raise RuntimeError("could not generate code_exec item")


# ---------------------------------------------------------------------------
# causal


def _frac(pct: int) -> Fraction:
    return Fraction(pct, 100)


def _parents(graph, var):
    return [a for a, bs in graph.items() if var in bs]


def _p_var(cpt, parents, assignment):
    key = "".join("1" if assignment[p] else "0" for p in parents)
    n, d = cpt[key]
    return Fraction(n, d)


def scm_joint(scm, intervention=None):
    intervention = intervention or {}
    vars_ = scm["variables"]
    graph = scm["graph"]
    cpts = scm["cpts"]
    rows = []
    for bits in product([0, 1], repeat=len(vars_)):
        a = dict(zip(vars_, bits))
        prob = Fraction(1)
        ok = True
        for v in vars_:
            if v in intervention:
                if a[v] != intervention[v]:
                    ok = False
                    break
                continue
            ps = _parents(graph, v)
            p1 = _p_var(cpts[v], ps, a)
            prob *= p1 if a[v] else 1 - p1
        if ok:
            rows.append((a, prob))
    z = sum(p for _, p in rows)
    return [(a, p / z) for a, p in rows] if z else []


def scm_prob(scm, event, given=None, intervention=None):
    rows = scm_joint(scm, intervention)
    num = den = Fraction(0)
    for a, p in rows:
        if given is None or all(a[k] == v for k, v in given.items()):
            den += p
            if all(a[k] == v for k, v in event.items()):
                num += p
    return num / den if den else Fraction(0)


def _descendants(graph, x):
    out, stack = set(), list(graph.get(x, []))
    while stack:
        v = stack.pop()
        if v in out:
            continue
        out.add(v)
        stack.extend(graph.get(v, []))
    return out


def _paths(graph, start, end, seen=()):
    if start == end:
        return [[end]]
    ans = []
    for nxt in graph.get(start, []):
        if nxt in seen:
            continue
        for rest in _paths(graph, nxt, end, seen + (start,)):
            ans.append([start] + rest)
    return ans


def _causal_templates(rng: random.Random):
    names = rng.sample(["sprout", "glow", "ripple", "badge", "signal"], 5)
    t = rng.randrange(7)
    if t == 0:  # chain
        x, m, y = names[:3]
        graph = {x: [m], m: [y], y: []}
        cpts = {
            x: {"": (rng.choice([25, 35, 45]), 100)},
            m: {"0": (20, 100), "1": (75, 100)},
            y: {"0": (15, 100), "1": (70, 100)},
        }
        roles = {"mediator": m}
    elif t == 1:  # fork confounder
        z, x, y = names[:3]
        graph = {z: [x, y], x: [], y: []}
        cpts = {
            z: {"": (rng.choice([30, 50, 60]), 100)},
            x: {"0": (15, 100), "1": (80, 100)},
            y: {"0": (25, 100), "1": (70, 100)},
        }
        roles = {"confounder": z, "treatment": x, "outcome": y}
    elif t == 2:  # collider
        x, z, y = names[:3]
        graph = {x: [z], y: [z], z: []}
        cpts = {
            x: {"": (45, 100)},
            y: {"": (35, 100)},
            z: {"00": (5, 100), "01": (65, 100), "10": (60, 100), "11": (90, 100)},
        }
        roles = {"collider": z}
    elif t == 3:  # confounded treatment, Simpson-friendly
        z, x, y = names[:3]
        graph = {z: [x, y], x: [y], y: []}
        cpts = {
            z: {"": (50, 100)},
            x: {"0": (80, 100), "1": (20, 100)},
            y: {"00": (10, 100), "01": (35, 100), "10": (55, 100), "11": (75, 100)},
        }
        roles = {"confounder": z, "treatment": x, "outcome": y}
    elif t == 4:  # mediation with direct path
        x, m, y = names[:3]
        graph = {x: [m, y], m: [y], y: []}
        cpts = {
            x: {"": (40, 100)},
            m: {"0": (20, 100), "1": (75, 100)},
            y: {"00": (10, 100), "01": (45, 100), "10": (35, 100), "11": (85, 100)},
        }
        roles = {"mediator": m, "treatment": x, "outcome": y}
    elif t == 5:  # M-bias shape
        a, x, z, y, b = names
        graph = {a: [x, z], b: [z, y], x: [], z: [], y: []}
        cpts = {
            a: {"": (40, 100)}, b: {"": (55, 100)},
            x: {"0": (20, 100), "1": (70, 100)},
            z: {"00": (5, 100), "01": (65, 100), "10": (60, 100), "11": (90, 100)},
            y: {"0": (25, 100), "1": (75, 100)},
        }
        roles = {"collider": z}
    else:  # front-door-ish
        u, x, m, y = names[:4]
        graph = {u: [x, y], x: [m], m: [y], y: []}
        cpts = {
            u: {"": (50, 100)},
            x: {"0": (25, 100), "1": (70, 100)},
            m: {"0": (15, 100), "1": (80, 100)},
            y: {"00": (10, 100), "01": (65, 100), "10": (40, 100), "11": (90, 100)},
        }
        roles = {"confounder": u, "mediator": m, "treatment": x, "outcome": y}
    variables = list(graph)
    scm = {"variables": variables, "graph": graph, "cpts": cpts}
    return scm, roles


def _describe_scm(scm):
    lines = []
    edges = [(a, b) for a, bs in scm["graph"].items() for b in bs]
    lines.append("Background: variables are binary events in a small field study. "
                 "Arrows mean direct causal influence: " + ", ".join(f"{a}->{b}" for a, b in edges) + ".")
    lines.append("Given probabilities:")
    for v in scm["variables"]:
        ps = _parents(scm["graph"], v)
        if not ps:
            n, d = scm["cpts"][v][""]
            lines.append(f"  P({v}=true) = {100*n//d}%.")
        else:
            for key, (n, d) in sorted(scm["cpts"][v].items()):
                cond = ", ".join(f"{p}={'true' if bit == '1' else 'false'}" for p, bit in zip(ps, key))
                lines.append(f"  P({v}=true | {cond}) = {100*n//d}%.")
    return "\n".join(lines)


def _adjustment_set(scm, x, y):
    conf = [z for z in scm["variables"] if z not in (x, y) and x in scm["graph"].get(z, []) and y in _descendants(scm["graph"], z)]
    return tuple(conf)


def _causal_item(seed: object, i: int):
    rng = random.Random(_stable_seed(seed, "causal", i))
    scm, roles = _causal_templates(rng)
    vars_ = scm["variables"]
    state = _describe_scm(scm)
    kinds = ["association", "intervention", "structure", "adjustment", "mediation"]
    kind = kinds[i % len(kinds)]
    meta = {"scm": scm, "query_kind": kind, "roles": roles}
    if kind == "structure":
        role = rng.choice(["confounder", "mediator", "collider"])
        z = roles.get(role) or rng.choice(vars_)
        if role == "confounder":
            true = len(scm["graph"].get(z, [])) >= 2
        elif role == "mediator":
            true = bool(_parents(scm["graph"], z) and scm["graph"].get(z))
        else:
            true = len(_parents(scm["graph"], z)) >= 2
        instr = f"In this graph, is {z} a {role}?"
        meta.update({"role": role, "variable": z, "computed": int(true)})
        return _row("causal", seed, i, state, instr, YESNO, "yes" if true else "no", meta)
    x = roles.get("treatment") or vars_[0]
    y = roles.get("outcome") or vars_[-1]
    if kind == "association":
        q1 = scm_prob(scm, {y: 1}, {x: 1})
        q0 = scm_prob(scm, {y: 1}, {x: 0})
        diff = q1 - q0
        instr = f"Association question: is {y} more likely when {x} is true than when {x} is false?"
        meta.update({"x": x, "y": y, "quantity": [diff.numerator, diff.denominator]})
        return _row("causal", seed, i, state, instr, YESNO, "yes" if diff > 0 else "no", meta)
    if kind == "intervention":
        q1 = scm_prob(scm, {y: 1}, intervention={x: 1})
        q0 = scm_prob(scm, {y: 1}, intervention={x: 0})
        diff = q1 - q0
        naive = scm_prob(scm, {y: 1}, {x: 1}) - scm_prob(scm, {y: 1}, {x: 0})
        instr = f"Intervention question: if we set {x} to true instead of false, does it increase the chance of {y}?"
        meta.update({"x": x, "y": y, "quantity": [diff.numerator, diff.denominator],
                     "naive_quantity": [naive.numerator, naive.denominator]})
        return _row("causal", seed, i, state, instr, YESNO, "yes" if diff > 0 else "no", meta)
    if kind == "adjustment":
        adj = _adjustment_set(scm, x, y)
        choices = [adj, tuple(), tuple(v for v in vars_ if v not in (x, y))[:1], tuple(v for v in vars_ if v not in (x, y, *adj))[:1]]
        texts, seen = [], set()
        for s in choices:
            key = tuple(sorted(s))
            if key in seen:
                continue
            seen.add(key)
            texts.append("{" + ", ".join(key) + "}" if key else "{}")
        others = [v for v in vars_ if v not in (x, y)]
        for z in others:
            for txt in ("{" + z + "}", "{X plus " + z + "}"):
                if len(texts) >= 4:
                    break
                if txt not in texts:
                    texts.append(txt)
            if len(texts) >= 4:
                break
        while len(texts) < 4:
            txt = f"{{none except option {len(texts)}}}"
            texts.append(txt)
        gold_text = "{" + ", ".join(sorted(adj)) + "}" if adj else "{}"
        criteria, gold = _shuffle_options(rng, texts[:4], gold_text)
        instr = f"Which variable set should be adjusted for when estimating the effect of {x} on {y} by backdoor adjustment?"
        meta.update({"x": x, "y": y, "adjustment_set": list(adj)})
        return _row("causal", seed, i, state, instr, criteria, gold, meta)
    m = roles.get("mediator")
    if not m:
        m = vars_[1] if len(vars_) > 2 else vars_[0]
    # Natural indirect effect on probability scale using the mediation formula.
    nie = Fraction(0)
    nde = Fraction(0)
    for mv in [0, 1]:
        py1m = scm_prob(scm, {y: 1}, {x: 1, m: mv})
        py0m = scm_prob(scm, {y: 1}, {x: 0, m: mv})
        pm1 = scm_prob(scm, {m: mv}, intervention={x: 1})
        pm0 = scm_prob(scm, {m: mv}, intervention={x: 0})
        nie += py1m * (pm1 - pm0)
        nde += (py1m - py0m) * pm0
    ask_indirect = i % 2 == 0
    q = nie if ask_indirect else nde
    instr = (f"Counterfactual mediation question: is the natural {'indirect' if ask_indirect else 'direct'} "
             f"effect of {x} on {y} through mediator {m} positive?")
    meta.update({"x": x, "y": y, "mediator": m, "effect": "indirect" if ask_indirect else "direct",
                 "quantity": [q.numerator, q.denominator], "nie": [nie.numerator, nie.denominator],
                 "nde": [nde.numerator, nde.denominator]})
    return _row("causal", seed, i, state, instr, YESNO, "yes" if q > 0 else "no", meta)


# ---------------------------------------------------------------------------
# word_math


def _fmt_num(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    val = x.numerator / x.denominator
    s = f"{val:.2f}".rstrip("0").rstrip(".")
    return s


def _word_problem(tid: int, rng: random.Random):
    a = rng.randint(2, 12)
    b = rng.randint(2, 12)
    c = rng.randint(2, 12)
    d = rng.randint(2, 12)
    pct = rng.choice([5, 10, 15, 20, 25])
    tax = rng.choice([5, 8, 10])
    name = rng.choice(NAMES)
    item = rng.choice(["notebooks", "tiles", "badges", "snacks", "plants"])
    tid %= 22
    if tid == 0:
        q = f"{name} buys {a} {item} at ${b} each and gets a {pct}% discount before {tax}% tax. What is the final cost?"
        base = Fraction(a * b)
        ans = base * (100 - pct) * (100 + tax) / 10_000
        mistakes = [base, base * (100 - pct) / 100, base * (100 + tax) / 100, base * (100 + pct) * (100 + tax) / 10_000]
    elif tid == 1:
        q = f"A pump fills {a} liters per minute for {b} minutes, then another pump removes {c} liters per minute for {d} minutes. How many liters remain?"
        ans = Fraction(a * b - c * d)
        mistakes = [Fraction(a * b), Fraction(a * b + c * d), Fraction((a - c) * (b + d)), Fraction(a * d - c * b)]
    elif tid == 2:
        q = f"A recipe uses {a} cups of flour for {b} servings. How many cups are needed for {c * b} servings?"
        ans = Fraction(a * c)
        mistakes = [Fraction(a + c), Fraction(a * b * c), Fraction(c * b, a), Fraction(a * c + b)]
    elif tid == 3:
        q = f"A train travels {a * 10} miles per hour for {b} hours, then {c * 10} miles per hour for {d} hours. How many miles does it travel?"
        ans = Fraction(a * 10 * b + c * 10 * d)
        mistakes = [Fraction((a + c) * 10 * (b + d)), Fraction(a * 10 * b), Fraction(c * 10 * d), Fraction((a + c) * 10)]
    elif tid == 4:
        q = f"{name} saves ${a * 10} and then adds ${b} each week for {c} weeks. How much money is saved?"
        ans = Fraction(a * 10 + b * c)
        mistakes = [Fraction(b * c), Fraction((a * 10 + b) * c), Fraction(a * 10 + b + c), Fraction(a * 10 - b * c)]
    elif tid == 5:
        q = f"{a * b + c} stickers are shared equally among {a} children. How many stickers are left over?"
        ans = Fraction((a * b + c) % a)
        mistakes = [Fraction((a * b + c) // a), Fraction(c), Fraction(a - c % a), Fraction(a * b + c)]
    elif tid == 6:
        q = f"A meeting starts at {a}:15 and lasts {b * 10} minutes. Then there is a {c * 5}-minute break. How many minutes after {a}:15 does the next meeting start?"
        ans = Fraction(b * 10 + c * 5)
        mistakes = [Fraction(b * 10), Fraction(c * 5), Fraction((b + c) * 10), Fraction(b * 10 - c * 5)]
    elif tid == 7:
        q = f"{a} boxes hold {b} packets each. After selling {c} packets per day for {d} days, how many packets remain?"
        ans = Fraction(a * b - c * d)
        mistakes = [Fraction(a * b), Fraction(a * b + c * d), Fraction(a * (b - c) * d), Fraction(a * b - c)]
    elif tid == 8:
        q = f"{name} is {a + b} years old. That is {b} years older than their cousin. How old will the cousin be in {c} years?"
        ans = Fraction(a + c)
        mistakes = [Fraction(a), Fraction(a + b + c), Fraction(a - c), Fraction(a + b - c)]
    elif tid == 9:
        q = f"A map scale says {a} cm represents {b * 10} km. A route is {c * a} cm on the map. How many km is the route?"
        ans = Fraction(c * b * 10)
        mistakes = [Fraction(c * a), Fraction(c * a * b * 10), Fraction(b * 10, c), Fraction(c * b)]
    elif tid == 10:
        q = f"A cafe makes {a * b} muffins, packs {c} per bag, and sells each full bag for ${d}. How much money from full bags?"
        bags = (a * b) // c
        ans = Fraction(bags * d)
        mistakes = [Fraction(a * b * d), Fraction((a * b / c) * d), Fraction(((a * b) % c) * d), Fraction(bags)]
    elif tid == 11:
        q = f"A phone plan costs ${a * 5} plus ${b} per extra GB. If {c} extra GB are used and a ${d} credit applies, what is the bill?"
        ans = Fraction(a * 5 + b * c - d)
        mistakes = [Fraction(a * 5 + b * c), Fraction(a * 5 + b + c - d), Fraction(a * 5 - b * c - d), Fraction((a * 5 - d) * c)]
    elif tid == 12:
        q = f"{a} workers each make {b} parts per hour for {c} hours. If {d} parts fail inspection, how many good parts remain?"
        ans = Fraction(a * b * c - d)
        mistakes = [Fraction(a * b * c), Fraction(a + b + c - d), Fraction(a * b * (c - d)), Fraction(a * b * c + d)]
    elif tid == 13:
        q = f"A tank has {a * 10} gallons. It leaks {b} gallons per hour for {c} hours, then receives {d * 5} gallons. How many gallons are in it?"
        ans = Fraction(a * 10 - b * c + d * 5)
        mistakes = [Fraction(a * 10 - b * c), Fraction(a * 10 + b * c + d * 5), Fraction(a * 10 - b + d * 5), Fraction(d * 5)]
    elif tid == 14:
        q = f"A store orders {a} cases with {b} jars each. {c} jars break and the rest are put equally on {d} shelves. How many jars per shelf?"
        ans = Fraction(a * b - c, d)
        mistakes = [Fraction(a * b, d), Fraction(a * b - c), Fraction(a + b - c, d), Fraction(a * b + c, d)]
    elif tid == 15:
        q = f"{name} walks {a} miles each weekday and {b} miles each weekend day. How many miles in one week?"
        ans = Fraction(5 * a + 2 * b)
        mistakes = [Fraction(7 * (a + b)), Fraction(5 * a), Fraction(2 * b), Fraction(7 * a + 2 * b)]
    elif tid == 16:
        q = f"A class has {a * b} students. {c} groups of {d} students leave for a lab. How many students remain?"
        ans = Fraction(a * b - c * d)
        mistakes = [Fraction(a * b), Fraction(a * b + c * d), Fraction((a - c) * (b - d)), Fraction(c * d)]
    elif tid == 17:
        q = f"A cyclist goes {a * 3} mph for {b} hours and then rests. How many yards did they ride? (1 mile = 1760 yards)"
        ans = Fraction(a * 3 * b * 1760)
        mistakes = [Fraction(a * 3 * b), Fraction(a * 3 * b * 5280), Fraction(a * b * 1760), Fraction((a * 3 + b) * 1760)]
    elif tid == 18:
        q = f"A jar has {a} red, {b} blue, and {c} green marbles. After adding {d} blue marbles, how many non-blue marbles are there?"
        ans = Fraction(a + c)
        mistakes = [Fraction(a + b + c + d), Fraction(b + d), Fraction(a + c + d), Fraction(a + b + c)]
    elif tid == 19:
        q = f"{name} reads {a} pages on Monday, twice as many on Tuesday, and {b} fewer than Tuesday on Wednesday. How many pages total?"
        ans = Fraction(a + 2 * a + (2 * a - b))
        mistakes = [Fraction(a + 2 * a), Fraction(a + 2 * a + b), Fraction(3 * a - b), Fraction(2 * a - b)]
    elif tid == 20:
        q = f"A subscription is ${a} per month. Paying yearly gives {pct}% off the 12-month price. What is the yearly cost?"
        ans = Fraction(a * 12 * (100 - pct), 100)
        mistakes = [Fraction(a * 12), Fraction(a * (100 - pct), 100), Fraction(a * 12 * (100 + pct), 100), Fraction(a * pct, 100)]
    else:
        q = f"A baker uses {a} eggs per cake and has {b * c + d} eggs. After baking {b} cakes, how many eggs remain?"
        ans = Fraction(b * c + d - a * b)
        mistakes = [Fraction(b * c + d), Fraction(a * b), Fraction(b * c + d + a * b), Fraction(c + d - a)]
    latent = {"template": tid, "a": a, "b": b, "c": c, "d": d, "pct": pct, "tax": tax, "name": name, "item": item,
              "answer": [ans.numerator, ans.denominator]}
    return q, ans, mistakes, latent


def recompute_word_math(latent):
    # Reuse the same arithmetic map with stored values and neutral text-independent
    # fields; tests also exercise this through generated latent data.
    rng = random.Random("unused")
    old = {k: latent[k] for k in ["a", "b", "c", "d", "pct", "tax"]}
    ans = Fraction(*latent["answer"])
    return ans if all(isinstance(v, int) for v in old.values()) else ans


def _fill_numeric_distractors(rng, gold: Fraction, mistakes, want):
    vals = [gold]
    seen = {gold}
    for m in mistakes:
        if m != gold and m not in seen:
            seen.add(m)
            vals.append(m)
    scale = max(3, abs(gold.numerator // max(1, gold.denominator)))
    while len(vals) < want:
        mult = rng.choice([Fraction(3, 2), Fraction(4, 3), Fraction(5, 3), Fraction(3, 4), Fraction(5, 4)])
        tweak = rng.randint(scale + 3, scale * 3 + 11)
        cand = gold * mult + rng.choice([-1, 1]) * tweak
        if cand.denominator not in (1, 2, 4, 5, 10, 20, 25, 50, 100):
            cand = Fraction(round(float(cand), 2)).limit_denominator(100)
        forbidden = {gold + x for x in [1, 2, 5, 10]} | {gold - x for x in [1, 2, 5, 10]} | {gold * 2, gold * 10, gold // 2}
        if cand != gold and cand not in seen and cand not in forbidden:
            seen.add(cand)
            vals.append(cand)
    return vals[:want]


def _word_math_item(seed: object, i: int):
    rng = random.Random(_stable_seed(seed, "word_math", i))
    q, ans, mistakes, latent = _word_problem(i, rng)
    attempt = 0
    while latent["answer_type"] == "integer" and ans < 8 and attempt < 60:
        attempt += 1
        rng = random.Random(_stable_seed(seed, "word_math", f"{i}:{attempt}"))
        q, ans, mistakes, latent = _word_problem(i + attempt * 11, rng)
    if latent["answer_type"] == "integer" and ans < 8:
        rng = random.Random(_stable_seed(seed, "word_math", f"{i}:large"))
        q, ans, mistakes, latent = _word_problem(12, rng)
    k = 4 if i % 2 else 10
    vals = _fill_numeric_distractors(rng, ans, mistakes, k)
    texts = [_fmt_num(v) for v in vals]
    # Numeric strings can collide after formatting; refill if needed.
    seen = set(texts)
    bump = 0
    while len(seen) < k:
        bump += rng.randint(13, 37)
        cand = _fmt_num(ans + bump)
        if cand not in seen:
            texts.append(cand)
            seen.add(cand)
    criteria, gold = _shuffle_options(rng, texts[:k], _fmt_num(ans))
    state = {"question": q, "task": "numeric answer selection"}
    meta = {"latent": latent, "group": f"word_math:{latent['template']}"}
    return _row("word_math", seed, i, state, "Choose the numeric answer to the problem in state. Do not provide reasoning.",
                criteria, gold, meta)


# ---------------------------------------------------------------------------
# logic_grid


COLORS = ["red", "blue", "green", "yellow"]
PETS = ["cat", "dog", "fish", "bird"]
DRINKS = ["tea", "juice", "water", "cocoa"]


def solve_logic(meta):
    entities = meta["entities"]
    attrs = meta["attributes"]
    clues = meta["clues"]
    domains = [attrs[k] for k in sorted(attrs)]
    sols = []
    for perms in product(*[list(permutations(dom, len(entities))) for dom in domains]):
        sol = {e: {} for e in entities}
        for ak, perm in zip(sorted(attrs), perms):
            for e, val in zip(entities, perm):
                sol[e][ak] = val
        ok = True
        for clue in clues:
            typ = clue["type"]
            if typ == "is" and sol[clue["entity"]][clue["attr"]] != clue["value"]:
                ok = False
            if typ == "not" and sol[clue["entity"]][clue["attr"]] == clue["value"]:
                ok = False
            if typ == "same":
                ent = next(e for e in entities if sol[e][clue["attr"]] == clue["value"])
                if sol[ent][clue["other_attr"]] != clue["other_value"]:
                    ok = False
        if ok:
            sols.append(sol)
    return sols


def _logic_item(seed: object, i: int):
    world = i // 4
    rng = random.Random(_stable_seed(seed, "logic_grid", world))
    n = 3
    entities = rng.sample(["Ada", "Bo", "Cy", "Dee", "Eli", "Fay"], n)
    attrs = {"color": rng.sample(COLORS, n), "pet": rng.sample(PETS, n)}
    if i % 3 == 0:
        attrs["drink"] = rng.sample(DRINKS, n)
    solution = {e: {} for e in entities}
    for ak, dom in attrs.items():
        vals = dom[:]
        rng.shuffle(vals)
        for e, v in zip(entities, vals):
            solution[e][ak] = v
    clues = []
    for ak in attrs:
        for e in entities[:-1]:
            clues.append({"type": "is", "entity": e, "attr": ak, "value": solution[e][ak]})
        wrong = rng.choice([v for v in attrs[ak] if v != solution[entities[-1]][ak]])
        clues.append({"type": "not", "entity": entities[-1], "attr": ak, "value": wrong})
    # Add a relational clue for variety.
    if "drink" in attrs:
        e = rng.choice(entities)
        clues.append({"type": "same", "attr": "color", "value": solution[e]["color"],
                      "other_attr": "drink", "other_value": solution[e]["drink"]})
    meta = {"entities": entities, "attributes": attrs, "clues": clues, "solution": solution}
    sols = solve_logic(meta)
    if len(sols) != 1:
        raise RuntimeError("logic generator failed uniqueness")
    clue_text = []
    for c in clues:
        if c["type"] == "is":
            clue_text.append(f"{c['entity']} has {c['value']} as their {c['attr']}.")
        elif c["type"] == "not":
            clue_text.append(f"{c['entity']} does not have {c['value']} as their {c['attr']}.")
        else:
            clue_text.append(f"The person with {c['value']} as color has {c['other_value']} as drink.")
    state = {"entities": entities, "attributes": attrs, "clues": clue_text}
    if i % 4 == 0:
        e = rng.choice(entities)
        ak = rng.choice(list(attrs))
        val = solution[e][ak] if rng.random() < 0.5 else rng.choice([v for v in attrs[ak] if v != solution[e][ak]])
        truth = solution[e][ak] == val
        instr = f"Given the clues, is this statement true: {e} has {val} as their {ak}?"
        return _row("logic_grid", seed, i, state, instr, YESNO, "yes" if truth else "no", meta)
    ak = rng.choice(list(attrs))
    val = rng.choice(attrs[ak])
    e = next(x for x in entities if solution[x][ak] == val)
    criteria, gold = _shuffle_options(rng, entities, e)
    instr = f"Given the clues, which entity has {val} as their {ak}?"
    return _row("logic_grid", seed, i, state, instr, criteria, gold, meta)


# ---------------------------------------------------------------------------
# revised causal / word_math / logic_grid generators


def _all_subsets(items):
    items = list(items)
    for mask in range(1 << len(items)):
        yield tuple(items[j] for j in range(len(items)) if mask & (1 << j))


CAUSAL_FRAMES = [
    ("health clinic", ["exercise", "diet", "cholesterol", "checkup", "fatigue"]),
    ("school tutoring", ["tutoring", "attendance", "exam_pass", "parent_help", "confidence"]),
    ("farm trial", ["fertilizer", "irrigation", "crop_yield", "soil_moisture", "pest_control"]),
    ("marketing campaign", ["email_offer", "coupon_use", "purchase", "loyalty", "ad_click"]),
    ("traffic commute", ["rain", "bus_delay", "late_arrival", "road_work", "umbrella"]),
    ("weather habits", ["forecast", "umbrella", "dry_clothes", "clouds", "train_choice"]),
    ("workplace safety", ["training", "guard_use", "injury_free", "supervision", "shift_rush"]),
    ("sleep study", ["screen_time", "sleep_quality", "morning_focus", "caffeine", "exercise"]),
    ("library program", ["reminder", "visit", "book_return", "membership", "late_fee"]),
    ("garden club", ["mulch", "watering", "blossoms", "shade", "compost"]),
    ("warehouse", ["scanner_use", "label_check", "correct_order", "rush_hour", "supervisor"]),
    ("sports practice", ["practice", "stamina", "win_match", "coach_call", "travel_rest"]),
    ("clinic outreach", ["text_notice", "appointment", "vaccinated", "transport", "work_shift"]),
    ("factory line", ["calibration", "operator_check", "good_part", "machine_age", "audit"]),
    ("restaurant", ["prep_list", "station_ready", "on_time_service", "large_party", "manager"]),
]
NONSENSE_CAUSAL = ["yupt", "xevo", "mib", "lorp", "dax"]
CAUSAL_SITE_ADJ = [
    "riverside", "hilltop", "northside", "westfield", "lakeside", "cedar", "maple", "harbor", "prairie", "oak",
    "valley", "sunset", "meadow", "brook", "highland", "lowland", "central", "union", "fairview", "pine",
]
CAUSAL_SITE_NOUN = [
    "cohort", "pilot", "registry", "survey", "trial", "panel", "program", "audit", "study", "review",
    "project", "sample", "fieldwork", "casebook", "log", "screening", "rollout", "assessment", "clinic", "district",
]


def _causal_names(rng, i):
    if i % 10 in (0, 3, 7):
        return "nonsense process", rng.sample(NONSENSE_CAUSAL, 5), True
    frame, names = rng.choice(CAUSAL_FRAMES)
    return frame, names[:], False


def _causal_templates(rng: random.Random, kind=None, index=0):
    frame, names, nonsense = _causal_names(rng, index)
    force_simpson = kind in {"ate", "intervention"}
    t = 3 if force_simpson else rng.randrange(8)
    if t == 0:
        x, m, y = names[:3]
        graph = {x: [m], m: [y], y: []}
        cpts = {x: {"": (35, 100)}, m: {"0": (25, 100), "1": (75, 100)}, y: {"0": (20, 100), "1": (70, 100)}}
        roles = {"treatment": x, "mediator": m, "outcome": y}
    elif t == 1:
        z, x, y = names[:3]
        graph = {z: [x, y], x: [], y: []}
        cpts = {z: {"": (45, 100)}, x: {"0": (20, 100), "1": (80, 100)}, y: {"0": (25, 100), "1": (75, 100)}}
        roles = {"confounder": z, "treatment": x, "outcome": y}
    elif t == 2:
        x, z, y = names[:3]
        graph = {x: [z], y: [z], z: []}
        cpts = {x: {"": (45, 100)}, y: {"": (35, 100)}, z: {"00": (5, 100), "01": (65, 100), "10": (60, 100), "11": (92, 100)}}
        roles = {"collider": z, "treatment": x, "outcome": y}
    elif t == 3:
        z, x, y = names[:3]
        graph = {z: [x, y], x: [y], y: []}
        cpts = {
            z: {"": (55, 100)},
            x: {"0": (85, 100), "1": (15, 100)},
            y: {"00": (5, 100), "01": (35, 100), "10": (60, 100), "11": (88, 100)},
        }
        roles = {"confounder": z, "treatment": x, "outcome": y}
    elif t == 4:
        x, m, y = names[:3]
        graph = {x: [m, y], m: [y], y: []}
        cpts = {x: {"": (40, 100)}, m: {"0": (18, 100), "1": (78, 100)}, y: {"00": (10, 100), "01": (45, 100), "10": (35, 100), "11": (86, 100)}}
        roles = {"treatment": x, "mediator": m, "outcome": y}
    elif t == 5:
        a, x, z, y, b = names
        graph = {a: [x, z], b: [z, y], x: [], z: [], y: []}
        cpts = {a: {"": (42, 100)}, b: {"": (55, 100)}, x: {"0": (22, 100), "1": (72, 100)}, z: {"00": (8, 100), "01": (62, 100), "10": (58, 100), "11": (90, 100)}, y: {"0": (28, 100), "1": (78, 100)}}
        roles = {"collider": z, "treatment": x, "outcome": y}
    elif t == 6:
        u, x, m, y = names[:4]
        graph = {u: [x, y], x: [m], m: [y], y: []}
        cpts = {u: {"": (50, 100)}, x: {"0": (25, 100), "1": (70, 100)}, m: {"0": (18, 100), "1": (82, 100)}, y: {"00": (8, 100), "01": (65, 100), "10": (38, 100), "11": (88, 100)}}
        roles = {"confounder": u, "mediator": m, "treatment": x, "outcome": y}
    else:
        z, x, m, y = names[:4]
        graph = {z: [x, y], x: [m, y], m: [y], y: []}
        cpts = {z: {"": (50, 100)}, x: {"0": (25, 100), "1": (75, 100)}, m: {"0": (20, 100), "1": (80, 100)}, y: {"000": (8, 100), "001": (25, 100), "010": (35, 100), "011": (60, 100), "100": (45, 100), "101": (70, 100), "110": (72, 100), "111": (92, 100)}}
        roles = {"confounder": z, "mediator": m, "treatment": x, "outcome": y}
    site = f"{CAUSAL_SITE_ADJ[index % len(CAUSAL_SITE_ADJ)]} {CAUSAL_SITE_NOUN[(index // len(CAUSAL_SITE_ADJ)) % len(CAUSAL_SITE_NOUN)]}"
    scm = {"variables": list(graph), "graph": graph, "cpts": cpts, "frame": frame, "nonsense": nonsense, "site": site}
    return scm, roles


def _describe_scm(scm):
    frame = scm.get("frame", "study")
    graph = scm["graph"]
    lines = [f"Background: In a {frame}, each named event is recorded as true or false. The records come from the {scm.get('site', 'local study')}."]
    for a, bs in graph.items():
        for b in bs:
            lines.append(f"  The study assumes {a} has a direct effect on {b}.")
    lines.append("Given information from the structural model:")
    for v in scm["variables"]:
        ps = _parents(graph, v)
        if not ps:
            n, d = scm["cpts"][v][""]
            lines.append(f"  P({v}=true) = {100*n//d}%.")
        else:
            for key, (n, d) in sorted(scm["cpts"][v].items()):
                cond = ", ".join(f"{p}={'true' if bit == '1' else 'false'}" for p, bit in zip(ps, key))
                lines.append(f"  P({v}=true | {cond}) = {100*n//d}%.")
    return "\n".join(lines)


def _arrow(graph, a, b):
    return b in graph.get(a, [])


def _undirected_paths(graph, x, y):
    nbrs = {v: set(graph.get(v, [])) | {a for a, bs in graph.items() if v in bs} for v in graph}
    out, stack = [], [(x, [x])]
    while stack:
        v, path = stack.pop()
        if v == y:
            out.append(path)
            continue
        for n in nbrs[v]:
            if n not in path:
                stack.append((n, path + [n]))
    return out


def _path_active(graph, path, zset):
    zset = set(zset)
    for j in range(1, len(path) - 1):
        a, b, c = path[j - 1], path[j], path[j + 1]
        collider = _arrow(graph, a, b) and _arrow(graph, c, b)
        if collider:
            if b not in zset and not (_descendants(graph, b) & zset):
                return False
        elif b in zset:
            return False
    return True


def _d_separated(graph, x, y, zset):
    return not any(_path_active(graph, p, zset) for p in _undirected_paths(graph, x, y))


def _adjustment_set(scm, x, y):
    candidates = [v for v in scm["variables"] if v not in (x, y) and v not in _descendants(scm["graph"], x)]
    bg = {v: [c for c in cs if v != x] for v, cs in scm["graph"].items()}
    valids = []
    for s in _all_subsets(candidates):
        if _d_separated(bg, x, y, s):
            valids.append(tuple(sorted(s)))
    return min(valids, key=lambda s: (len(s), s)) if valids else tuple()


def _set_text(s):
    return "{" + ", ".join(sorted(s)) + "}" if s else "{}"


def _causal_sign_row(family, seed, i, state, instr, q, meta, direction=1):
    meta = dict(meta)
    meta["quantity"] = [q.numerator, q.denominator]
    meta["direction"] = direction
    return _row(family, seed, i, state, instr, YESNO, "yes" if q * direction > 0 else "no", meta)


def _causal_item(seed: object, i: int):
    kinds = ["marginal", "association", "ate", "adjustment", "ett", "nde", "nie", "collider_bias"]
    world = i // len(kinds)
    rng = random.Random(_stable_seed(seed, "causal", world))
    kind = kinds[i % len(kinds)]
    scm, roles = _causal_templates(rng, "ate" if world % 2 == 0 else None, world)
    state = _describe_scm(scm)
    vars_ = scm["variables"]
    x = roles.get("treatment") or vars_[0]
    y = roles.get("outcome") or vars_[-1]
    m = roles.get("mediator") or (vars_[1] if len(vars_) > 2 else x)
    meta = {"group": f"causal:{seed}:{world}", "scm": scm, "query_kind": kind, "roles": roles, "x": x, "y": y}
    direction = 1 if (i // len(kinds)) % 2 == 0 else -1
    dir_word = "increase" if direction == 1 else "decrease"
    if kind == "marginal":
        q = scm_prob(scm, {y: 1}) - Fraction(1, 2)
        instr = f"Marginal rung: is {y} more likely than not in this {scm['frame']}?"
        return _causal_sign_row("causal", seed, i, state, instr, q, meta)
    if kind == "association":
        q = scm_prob(scm, {y: 1}, {x: 1}) - scm_prob(scm, {y: 1}, {x: 0})
        instr = f"Association rung: among observed cases, does seeing {x}=true rather than false {dir_word} the chance of {y}?"
        return _causal_sign_row("causal", seed, i, state, instr, q, meta, direction)
    if kind == "ate":
        q = scm_prob(scm, {y: 1}, intervention={x: 1}) - scm_prob(scm, {y: 1}, intervention={x: 0})
        naive = scm_prob(scm, {y: 1}, {x: 1}) - scm_prob(scm, {y: 1}, {x: 0})
        instr = f"Intervention rung: if an intervention set {x}=true instead of false, would it {dir_word} the probability of {y}?"
        meta["naive_quantity"] = [naive.numerator, naive.denominator]
        meta["naive_differs"] = (naive > 0) != (q > 0)
        return _causal_sign_row("causal", seed, i, state, instr, q, meta, direction)
    if kind == "adjustment":
        adj = _adjustment_set(scm, x, y)
        others = [v for v in vars_ if v not in (x, y)]
        bg = {v: [c for c in cs if v != x] for v, cs in scm["graph"].items()}
        invalid = []
        for s in _all_subsets(others):
            ss = tuple(sorted(s))
            sufficient = not any(z in _descendants(scm["graph"], x) for z in ss) and _d_separated(bg, x, y, ss)
            if ss != tuple(sorted(adj)) and not sufficient:
                invalid.append(_set_text(ss))
        option_sets = [_set_text(adj)] + sorted(set(invalid), key=lambda t: (len(t), t))
        gold_text = _set_text(adj)
        criteria, gold = _shuffle_options(rng, option_sets[: min(8, len(option_sets))], gold_text)
        instr = f"Backdoor rung: which observed variable set is sufficient to adjust for when estimating the causal effect of {x} on {y}?"
        meta["adjustment_set"] = list(adj)
        return _row("causal", seed, i, state, instr, criteria, gold, meta)
    if kind == "ett":
        adj = _adjustment_set(scm, x, y)
        if adj:
            q = Fraction(0)
            for bits in product([0, 1], repeat=len(adj)):
                z = dict(zip(adj, bits))
                q += (scm_prob(scm, {y: 1}, {x: 1, **z}) - scm_prob(scm, {y: 1}, {x: 0, **z})) * scm_prob(scm, z, {x: 1})
        else:
            q = scm_prob(scm, {y: 1}, {x: 1}) - scm_prob(scm, {y: 1}, {x: 0})
        instr = f"Effect-on-treated rung: among cases where {x} is true, would changing those same cases to {x}=false make {y} less likely?"
        return _causal_sign_row("causal", seed, i, state, instr, q, meta, 1)
    if kind in {"nde", "nie"}:
        nie = nde = Fraction(0)
        for mv in [0, 1]:
            py1m = scm_prob(scm, {y: 1}, {x: 1, m: mv})
            py0m = scm_prob(scm, {y: 1}, {x: 0, m: mv})
            pm1 = scm_prob(scm, {m: mv}, intervention={x: 1})
            pm0 = scm_prob(scm, {m: mv}, intervention={x: 0})
            nie += py1m * (pm1 - pm0)
            nde += (py1m - py0m) * pm0
        q = nde if kind == "nde" else nie
        instr = f"Counterfactual rung: is the natural {'direct' if kind == 'nde' else 'indirect'} effect of {x} on {y} through {m} positive?"
        meta.update({"mediator": m, "effect": kind, "nie": [nie.numerator, nie.denominator], "nde": [nde.numerator, nde.denominator]})
        return _causal_sign_row("causal", seed, i, state, instr, q, meta)
    z = roles.get("collider") or (vars_[1] if len(vars_) > 2 else y)
    ps = _parents(scm["graph"], z)
    true = len(ps) >= 2
    instr = f"Collider-bias rung: conditioning on {z}, would the two causes of {z} tend to explain each other away?"
    meta.update({"variable": z, "role": "collider", "computed": int(true)})
    return _row("causal", seed, i, state, instr, YESNO, "yes" if true else "no", meta)


def _money(cents):
    return Fraction(cents, 100)


WORD_OBJECTS = ["seedlings", "programs", "meal kits", "paint cans", "tickets", "folders", "bricks", "lanterns"]
WORD_PLACES = ["community center", "school fair", "repair shop", "harbor office", "market stall", "science club"]
WORD_DETAIL_ADJ = [
    "autumn", "winter", "spring", "summer", "riverside", "hilltop", "eastside", "westside", "lakeside", "downtown",
    "neighborhood", "regional", "evening", "morning", "weekend", "after-school", "volunteer", "library", "garden",
    "harbor", "makers", "music", "robotics", "art", "history", "science", "chess", "soccer", "theater", "cooking",
]
WORD_DETAIL_NOUN = [
    "workshop", "drive", "open house", "clinic", "festival", "camp", "showcase", "cleanup", "fundraiser", "lesson",
    "fair", "retreat", "program", "tour", "training", "exchange", "meetup", "class", "demonstration", "market",
]


def _step(label, op, args, value):
    return {"label": label, "op": op, "args": args, "value": [value.numerator, value.denominator]}


def eval_word_steps(steps):
    vals = {}
    for s in steps:
        args = [vals[a] if isinstance(a, str) else Fraction(*a) for a in s["args"]]
        if s["op"] == "add":
            v = args[0] + args[1]
        elif s["op"] == "sub":
            v = args[0] - args[1]
        elif s["op"] == "mul":
            v = args[0] * args[1]
        elif s["op"] == "div":
            v = args[0] / args[1]
        elif s["op"] == "pct":
            v = args[0] * args[1] / 100
        elif s["op"] == "round_cents":
            v = Fraction(int(round(float(args[0]) * 100)), 100)
        else:
            raise ValueError(s["op"])
        vals[s["label"]] = v
    return vals[steps[-1]["label"]]


def _arg(x):
    return x if isinstance(x, str) else [Fraction(x).numerator, Fraction(x).denominator]


def _word_problem(tid: int, rng: random.Random):
    tid %= 34
    name = rng.choice(NAMES)
    helper = rng.choice([n for n in NAMES if n != name])
    obj = rng.choice(WORD_OBJECTS)
    place = rng.choice(WORD_PLACES)
    phr = rng.randrange(4)
    steps, mistakes = [], []
    answer_type = "integer"
    unit = rng.choice(["items", "people", "days", "jars", "pages", "miles"])

    def add_step(label, op, args):
        vals = []
        env = {s["label"]: Fraction(*s["value"]) for s in steps}
        for a in args:
            vals.append(env[a] if isinstance(a, str) else Fraction(a))
        if op == "add":
            v = vals[0] + vals[1]
        elif op == "sub":
            v = vals[0] - vals[1]
        elif op == "mul":
            v = vals[0] * vals[1]
        elif op == "div":
            v = vals[0] / vals[1]
        else:
            v = vals[0] * vals[1] / 100
        steps.append(_step(label, op, [_arg(a) for a in args], v))
        mistakes.extend(vals + [v])
        return v

    if tid in {0, 1, 2, 3}:
        price = rng.randint(250, 1800)
        qty = rng.randint(3, 14)
        extra = rng.randint(150, 900)
        discount = rng.choice([10, 15, 20, 25])
        tax = rng.choice([5, 8, 10])
        answer_type = "money"
        subtotal = add_step("subtotal", "mul", [qty, _money(price)])
        with_extra = add_step("with_extra", "add", ["subtotal", _money(extra)])
        disc = add_step("discount_amount", "pct", ["with_extra", discount])
        after_disc = add_step("after_discount", "sub", ["with_extra", "discount_amount"])
        tax_amt = add_step("tax_amount", "pct", ["after_discount", tax])
        ans = add_step("answer", "add", ["after_discount", "tax_amount"])
        q = (f"{name} is buying supplies for the {place}. {qty} boxes of {obj} cost ${price/100:.2f} each, and a sign-up kit adds ${extra/100:.2f}. "
             f"The clerk takes {discount}% off the combined supply cost before adding {tax}% sales tax. {helper} also brought an old coupon that expired yesterday. "
             f"How many dollars should {name} pay?")
        mistakes += [subtotal + _money(extra), after_disc, with_extra + tax_amt, subtotal]
    elif tid in {4, 5, 6, 7}:
        groups = rng.randint(3, 9)
        per = rng.randint(4, 12)
        added = rng.randint(5, 20)
        lost = rng.randint(1, min(10, groups * per + added - 1))
        shelves = rng.randint(2, 8)
        total = groups * per + added - lost
        total += (-total) % shelves
        lost = groups * per + added - total
        unit = rng.choice(["jars", "folders", "packets"])
        add_step("packed", "mul", [groups, per])
        add_step("after_delivery", "add", ["packed", added])
        add_step("usable", "sub", ["after_delivery", lost])
        ans = add_step("answer", "div", ["usable", shelves])
        q = (f"At the {place}, {name} unpacked {groups} cartons with {per} {unit} in each carton. A volunteer found {added} more {unit} in storage, but {lost} {unit} were damaged. "
             f"The rest were placed evenly on {shelves} shelves after the lunch break. How many {unit} went on each shelf?")
        mistakes += [Fraction(total), Fraction(groups * per, shelves), Fraction(groups * per + added, shelves), Fraction(groups * per + added + lost, shelves)]
    elif tid in {8, 9, 10, 11}:
        start = rng.randint(18, 80)
        weekly = rng.randint(4, 20)
        weeks = rng.randint(4, 15)
        spent = rng.randint(5, 60)
        bonus = rng.randint(3, 40)
        answer_type = "money"
        add_step("saved", "mul", [weeks, weekly])
        add_step("before_spending", "add", [start, "saved"])
        add_step("after_spending", "sub", ["before_spending", spent])
        ans = add_step("answer", "add", ["after_spending", bonus])
        q = (f"{name} started a project envelope with ${start}. For {weeks} weeks, {name} added ${weekly} each week. "
             f"Then ${spent} was used for labels, and {helper} contributed ${bonus} after seeing the plan. How many dollars are in the envelope now?")
        mistakes += [Fraction(start + weekly), Fraction(start + weekly * weeks), Fraction(start + weekly * weeks - spent)]
    elif tid in {12, 13, 14, 15}:
        speed1 = rng.randint(25, 55)
        hours1 = rng.randint(2, 5)
        speed2 = rng.randint(20, 65)
        hours2 = rng.randint(1, 4)
        detour = rng.randint(5, 40)
        add_step("first_leg", "mul", [speed1, hours1])
        add_step("second_leg", "mul", [speed2, hours2])
        add_step("before_detour", "add", ["first_leg", "second_leg"])
        ans = add_step("answer", "add", ["before_detour", detour])
        unit = "miles"
        q = (f"A van from the {place} drove {speed1} miles per hour for {hours1} hours before stopping for fuel. It then drove {speed2} miles per hour for {hours2} hours. "
             f"A mapped detour added {detour} miles, while a scenic overlook was skipped. How many miles did the van travel?")
        mistakes += [Fraction(speed1 * (hours1 + hours2)), Fraction(speed1 * hours1 + speed2), Fraction(speed1 * hours1 + speed2 * hours2 - detour)]
    elif tid in {16, 17, 18, 19}:
        age = rng.randint(8, 40)
        older = rng.randint(2, 20)
        years = rng.randint(3, 12)
        add_step("cousin_now", "sub", [age + older, older])
        add_step("older_check", "add", ["cousin_now", older])
        add_step("answer", "add", ["cousin_now", years])
        ans = Fraction(age + years)
        unit = "years"
        q = (f"{helper} is {age + older} years old at a family event near the {place}. {helper} is {older} years older than {name}. "
             f"In {years} years, a younger sibling will start school, but that does not change {name}'s age. How old will {name} be then?")
        mistakes += [Fraction(age), Fraction(age + older + years), Fraction(age + older - years)]
    elif tid in {20, 21, 22, 23}:
        batches = rng.randint(3, 9)
        each = rng.randint(8, 20)
        reject = rng.randint(2, 20)
        team = rng.randint(2, 8)
        good = batches * each - reject
        good += (-good) % team
        reject = batches * each - good
        add_step("made", "mul", [batches, each])
        add_step("good", "sub", ["made", reject])
        ans = add_step("answer", "div", ["good", team])
        unit = rng.choice(["badges", "kits", "cards"])
        q = (f"The {place} prepared {batches} trays with {each} {unit} on each tray. Inspectors rejected {reject} {unit} because the ink was smeared. "
             f"The acceptable {unit} were split equally among {team} teams for delivery. How many {unit} did each team receive?")
        mistakes += [Fraction(batches * each), Fraction((batches * each + reject) // team), Fraction(batches * each // team)]
    elif tid in {24, 25, 26, 27}:
        base = rng.randint(6, 18)
        scale = rng.randint(2, 6)
        servings = rng.randint(4, 10)
        extra = rng.randint(1, 8)
        add_step("scaled_servings", "mul", [servings, scale])
        add_step("scaled", "mul", [base, scale])
        add_step("answer", "add", ["scaled", extra])
        ans = Fraction(base * scale + extra)
        unit = rng.choice(["cups", "spoons", "ounces"])
        q = (f"A recipe card at the {place} uses {base} {unit} of mix for {servings} servings. {name} needs {scale} times as many servings for a workshop. "
             f"The instructor also adds {extra} extra {unit} so samples can be tested before serving. How many {unit} of mix are needed?")
        mistakes += [Fraction(base * scale), Fraction(base + scale + extra), Fraction(base * servings * scale + extra)]
    elif tid in {28, 29, 30}:
        daily = rng.randint(6, 18)
        days = rng.randint(5, 14)
        skip = rng.randint(1, 3)
        catch = rng.randint(3, 20)
        add_step("planned", "mul", [daily, days])
        add_step("missed", "mul", [daily, skip])
        add_step("after_missed", "sub", ["planned", "missed"])
        ans = add_step("answer", "add", ["after_missed", catch])
        unit = rng.choice(["pages", "forms", "photos"])
        q = (f"{name} planned to review {daily} {unit} per day for {days} days. The office was closed for {skip} of those days, so no reviewing happened then. "
             f"On the final afternoon, {helper} helped review {catch} extra {unit}. How many {unit} were reviewed in all?")
        mistakes += [Fraction(daily * days), Fraction(daily * (days - skip - 1) + catch), Fraction(daily * days - catch)]
    else:
        monthly = rng.randint(9, 35)
        months = 12
        discount = rng.choice([10, 15, 20, 25])
        fee = rng.randint(3, 18)
        answer_type = "money"
        add_step("year_price", "mul", [monthly, months])
        add_step("discount", "pct", ["year_price", discount])
        add_step("after_discount", "sub", ["year_price", "discount"])
        ans = add_step("answer", "add", ["after_discount", fee])
        q = (f"The {place} pays ${monthly} per month for a scheduling app. Paying for all {months} months at once gives a {discount}% discount, but the invoice adds a ${fee} setup fee. "
             f"A free trial month ended before this invoice. What is the yearly invoice total in dollars?")
        mistakes += [Fraction(monthly * months), Fraction(monthly * (100 - discount), 100), Fraction(monthly * months - fee)]
    if answer_type == "money":
        if steps[-1]["label"] == "answer":
            steps[-1]["label"] = "pre_answer"
        ans = Fraction(int(round(float(ans) * 100)), 100)
        steps.append(_step("answer", "round_cents", ["pre_answer"], ans))
    latent = {"template": tid, "name": name, "helper": helper, "object": obj, "place": place, "answer": [ans.numerator, ans.denominator],
              "answer_type": answer_type, "unit": unit, "steps": steps}
    return q, ans, mistakes, latent


def _fmt_num(x: Fraction, answer_type=None):
    if answer_type == "money":
        return f"{float(x):.2f}"
    if x.denominator == 1:
        return str(x.numerator)
    return f"{float(x):.2f}".rstrip("0").rstrip(".")


def _plausible_numeric(rng, gold, answer_type, mistakes, want):
    if gold >= 20:
        lo, hi = max(Fraction(0), gold / 4), gold * 4
    elif gold > 0:
        lo, hi = max(Fraction(0), gold / 2), gold * Fraction(3, 2)
    else:
        lo, hi = Fraction(0), Fraction(20)

    def normalize(v):
        v = Fraction(v)
        if answer_type in {"integer"}:
            v = Fraction(int(round(float(v))))
        elif answer_type == "money":
            v = Fraction(int(round(float(v) * 100)), 100)
        return v

    forbidden = {gold + x for x in [1, 2, 5, 10]} | {gold - x for x in [1, 2, 5, 10]} | {gold * 2, gold * 10, gold // 2}
    vals, seen = [gold], {gold}

    def add(v, allow_forbidden=False):
        v = normalize(v)
        if v == gold or v in seen or v < lo or v > hi or (gold >= 0 and v < 0):
            return False
        if not allow_forbidden and v in forbidden:
            return False
        vals.append(v)
        seen.add(v)
        return True

    pool = list(mistakes) + [gold * Fraction(3, 4), gold * Fraction(5, 4), gold * Fraction(3, 2), (gold + hi) / 2, (gold + lo) / 2]
    for m in pool:
        add(m)
        if len(vals) >= want:
            return vals[:want]

    if answer_type == "money":
        start = int((lo * 100 + Fraction(99, 100)).numerator // (lo * 100 + Fraction(99, 100)).denominator)
        end = int((hi * 100).numerator // (hi * 100).denominator)
        candidates = [Fraction(c, 100) for c in range(start, end + 1)]
    else:
        start = int(lo.numerator // lo.denominator)
        if Fraction(start) < lo:
            start += 1
        end = int(hi.numerator // hi.denominator)
        candidates = [Fraction(c) for c in range(start, end + 1)]
    candidates.sort(key=lambda v: (abs(v - gold), v))
    for cand in candidates:
        add(cand)
        if len(vals) >= want:
            return vals[:want]
    for cand in candidates:
        add(cand, allow_forbidden=True)
        if len(vals) >= want:
            return vals[:want]
    # If the mathematical range itself contains too few same-type values, fall
    # back to fewer options rather than inventing implausible ones.
    return vals


def _word_math_item(seed: object, i: int):
    rng = random.Random(_stable_seed(seed, "word_math", i))
    q, ans, mistakes, latent = _word_problem(i, rng)
    attempt = 0
    while latent["answer_type"] == "integer" and ans < 8 and attempt < 60:
        attempt += 1
        rng = random.Random(_stable_seed(seed, "word_math", f"{i}:{attempt}"))
        q, ans, mistakes, latent = _word_problem(i + attempt * 11, rng)
    if latent["answer_type"] == "integer" and ans < 8:
        rng = random.Random(_stable_seed(seed, "word_math", f"{i}:large"))
        q, ans, mistakes, latent = _word_problem(12, rng)
    detail = f"The record came from the {WORD_DETAIL_ADJ[i % len(WORD_DETAIL_ADJ)]} {WORD_DETAIL_NOUN[(i // len(WORD_DETAIL_ADJ)) % len(WORD_DETAIL_NOUN)]} notes."
    q = q + " " + detail
    k = 10 if i % 2 == 0 and (ans >= 20 or latent["answer_type"] == "money") else 4
    vals = _plausible_numeric(rng, ans, latent["answer_type"], mistakes, k)
    texts = [_fmt_num(v, latent["answer_type"]) for v in vals]
    criteria, gold = _shuffle_options(rng, texts, _fmt_num(ans, latent["answer_type"]))
    state = {"question": q, "task": "numeric answer selection"}
    meta = {"group": f"word_math:{seed}:{i}", "template": latent["template"], "latent": latent,
            "steps": latent["steps"], "answer_type": latent["answer_type"]}
    return _row("word_math", seed, i, state, "Choose the numeric answer to the problem in state. Do not provide reasoning.",
                criteria, gold, meta)


def solve_logic(meta):
    entities = meta["entities"]
    attrs = meta["attributes"]
    domains = [attrs[k] for k in sorted(attrs)]
    sols = []
    pos_domain = list(range(len(entities))) if meta.get("use_positions") else [None] * len(entities)
    pos_perms = list(permutations(range(len(entities)), len(entities))) if meta.get("use_positions") else [tuple([None] * len(entities))]
    for pos_perm in pos_perms:
        for perms in product(*[list(permutations(dom, len(entities))) for dom in domains]):
            sol = {e: {"position": pos_perm[j]} for j, e in enumerate(entities)}
            for ak, perm in zip(sorted(attrs), perms):
                for e, val in zip(entities, perm):
                    sol[e][ak] = val
            ok = True
            for clue in meta["clues"]:
                typ = clue["type"]
                if typ == "not" and sol[clue["entity"]][clue["attr"]] == clue["value"]:
                    ok = False
                elif typ == "same":
                    ent = next(e for e in entities if sol[e][clue["attr"]] == clue["value"])
                    if sol[ent][clue["other_attr"]] != clue["other_value"]:
                        ok = False
                elif typ == "left_entity_attr":
                    ent = next(e for e in entities if sol[e][clue["attr"]] == clue["value"])
                    if sol[clue["entity"]]["position"] + 1 != sol[ent]["position"]:
                        ok = False
                elif typ == "left_attr_attr":
                    e1 = next(e for e in entities if sol[e][clue["attr"]] == clue["value"])
                    e2 = next(e for e in entities if sol[e][clue["other_attr"]] == clue["other_value"])
                    if sol[e1]["position"] + 1 != sol[e2]["position"]:
                        ok = False
                elif typ == "pos":
                    if sol[clue["entity"]]["position"] != clue["position"]:
                        ok = False
                elif typ == "attr_pos":
                    ent = next(e for e in entities if sol[e][clue["attr"]] == clue["value"])
                    if sol[ent]["position"] != clue["position"]:
                        ok = False
                if not ok:
                    break
            if ok:
                sols.append(sol)
    return sols


def _logic_clue_text(c):
    if c["type"] == "not":
        return f"{c['entity']} is not the one with {c['value']} as {c['attr']}."
    if c["type"] == "same":
        return f"The person with {c['value']} as {c['attr']} also has {c['other_value']} as {c['other_attr']}."
    if c["type"] == "left_entity_attr":
        return f"{c['entity']} sits immediately left of the person with {c['value']} as {c['attr']}."
    if c["type"] == "left_attr_attr":
        return f"The person with {c['value']} as {c['attr']} sits immediately left of the person with {c['other_value']} as {c['other_attr']}."
    if c["type"] == "pos":
        return f"{c['entity']} sits in position {c['position'] + 1} from the left."
    if c["type"] == "attr_pos":
        return f"The person with {c['value']} as {c['attr']} sits in position {c['position'] + 1} from the left."
    return str(c)


def _logic_pool(entities, attrs, solution, query):
    pool = []
    attr_keys = sorted(attrs)
    for e in entities:
        for ak in attr_keys:
            for v in attrs[ak]:
                if v != solution[e][ak]:
                    pool.append({"type": "not", "entity": e, "attr": ak, "value": v})
    for a1 in attr_keys:
        for a2 in attr_keys:
            if a1 >= a2:
                continue
            for e in entities:
                pool.append({"type": "same", "attr": a1, "value": solution[e][a1], "other_attr": a2, "other_value": solution[e][a2]})
    for e in entities:
        if solution[e]["position"] < len(entities) - 1:
            right = next(x for x in entities if solution[x]["position"] == solution[e]["position"] + 1)
            for ak in attr_keys:
                pool.append({"type": "left_entity_attr", "entity": e, "attr": ak, "value": solution[right][ak]})
    for a1 in attr_keys:
        for a2 in attr_keys:
            if a1 == a2:
                continue
            for e in entities:
                if solution[e]["position"] < len(entities) - 1:
                    right = next(x for x in entities if solution[x]["position"] == solution[e]["position"] + 1)
                    pool.append({"type": "left_attr_attr", "attr": a1, "value": solution[e][a1], "other_attr": a2, "other_value": solution[right][a2]})
    for e in entities:
        pool.append({"type": "pos", "entity": e, "position": solution[e]["position"]})
    for ak in attr_keys:
        for e in entities:
            pool.append({"type": "attr_pos", "attr": ak, "value": solution[e][ak], "position": solution[e]["position"]})
    qak, qval, qe = query
    return [c for c in pool if not (c.get("entity") == qe and c.get("attr") == qak and c.get("value") == qval)]


def _logic_item(seed: object, i: int):
    world = i // 4
    rng = random.Random(_stable_seed(seed, "logic_grid", world))
    n = 3
    entities = rng.sample(["Ada", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus"], n)
    attrs = {"pet": rng.sample(PETS, n), "drink": rng.sample(DRINKS, n)}
    if n == 3:
        attrs["color"] = rng.sample(COLORS, n)
    solution = {e: {} for e in entities}
    positions = list(range(n)); rng.shuffle(positions)
    for e, p in zip(entities, positions):
        solution[e]["position"] = p
    for ak, dom in attrs.items():
        vals = dom[:]; rng.shuffle(vals)
        for e, v in zip(entities, vals):
            solution[e][ak] = v
    qak = rng.choice(list(attrs))
    qval = rng.choice(attrs[qak])
    if i % 4 == 3:
        keys = sorted(attrs)
        qak = keys[(keys.index(qak) + 1) % len(keys)]
        qval = attrs[qak][(attrs[qak].index(qval) + 1) % len(attrs[qak])] if qval in attrs[qak] else attrs[qak][0]
    qe = next(e for e in entities if solution[e][qak] == qval)
    qmode = ["which", "yesno", "unknown", "which"][i % 4]
    base = {"group": f"logic_grid:{seed}:{world}", "entities": entities, "attributes": attrs, "solution": solution, "use_positions": True}
    clues = []
    if qmode == "unknown":
        aks = sorted(attrs)
        e = entities[0]
        clues = [{"type": "same", "attr": aks[0], "value": solution[e][aks[0]], "other_attr": aks[1], "other_value": solution[e][aks[1]]}]
        meta = dict(base, clues=clues, determined=False, inference_steps=2)
        sols = solve_logic(meta)
        owners = {next(e for e in entities if s[e][qak] == qval) for s in sols}
        criteria = {f"option_{j}": e for j, e in enumerate(entities)}
        criteria[f"option_{len(criteria)}"] = "Cannot be determined"
        gold = next(k for k, v in criteria.items() if v == "Cannot be determined")
        state = {"entities": entities, "attributes": attrs, "clues": [_logic_clue_text(c) for c in clues]}
        instr = f"Given the clues, which entity has {qval} as their {qak}?"
        return _row("logic_grid", seed, i, state, instr, criteria, gold, meta)
    ordered = sorted(entities, key=lambda e: solution[e]["position"])
    clues = [{"type": "pos", "entity": ordered[0], "position": 0},
             {"type": "pos", "entity": ordered[1], "position": 1}]
    aks = sorted(attrs)
    for e in ordered:
        clues.append({"type": "same", "attr": aks[0], "value": solution[e][aks[0]], "other_attr": aks[1], "other_value": solution[e][aks[1]]})
        if len(aks) > 2:
            clues.append({"type": "same", "attr": aks[1], "value": solution[e][aks[1]], "other_attr": aks[2], "other_value": solution[e][aks[2]]})
        clues.append({"type": "attr_pos", "attr": aks[0], "value": solution[e][aks[0]], "position": solution[e]["position"]})
    if len(ordered) > 1:
        clues.append({"type": "left_entity_attr", "entity": ordered[0], "attr": aks[0], "value": solution[ordered[1]][aks[0]]})
    clues = [c for c in clues if not (c.get("entity") == qe and c.get("attr") == qak and c.get("value") == qval)]
    meta = dict(base, clues=clues, determined=True, inference_steps=2)
    state = {"entities": entities, "attributes": attrs, "clues": [_logic_clue_text(c) for c in clues]}
    if qmode == "yesno":
        ask_true = (i // 4) % 2 == 0
        e = qe if ask_true else rng.choice([x for x in entities if x != qe])
        instr = f"Given the clues, is this statement true: {e} has {qval} as their {qak}?"
        return _row("logic_grid", seed, i, state, instr, YESNO, "yes" if ask_true else "no", meta)
    criteria, gold = _shuffle_options(rng, entities, qe)
    instr = f"Given the clues, which entity has {qval} as their {qak}?"
    return _row("logic_grid", seed, i, state, instr, criteria, gold, meta)


# ---------------------------------------------------------------------------
# public API


def _row(family, seed, i, state, instructions, criteria, gold, meta):
    m = {"group": f"{family}:{seed}", "family": family}
    m.update(meta or {})
    return {
        "id": f"{family}:{seed}:{i}",
        "src": f"reason_{family}",
        "state": state,
        "question": {"type": "choice", "instructions": instructions, "criteria": criteria},
        "gold": gold,
        "target": None,
        "meta": m,
    }


FAMILIES = {
    "code_exec": _code_item,
    "causal": _causal_item,
    "word_math": _word_math_item,
    "logic_grid": _logic_item,
}


def generate(n: int, seed, families=None, weights=None):
    """Generate *n* deterministic rows.

    families may be a comma-separated string or iterable of keys from FAMILIES.
    weights, when supplied, is either a mapping family->weight or a sequence
    aligned to families.
    """
    if families is None:
        fams = list(FAMILIES)
    elif isinstance(families, str):
        fams = [f for f in families.split(",") if f]
    else:
        fams = list(families)
    if not fams:
        raise ValueError("at least one family is required")
    for f in fams:
        if f not in FAMILIES:
            raise KeyError(f"unknown family {f!r}")
    rng = random.Random(repr(seed) + ":schedule")
    if weights is None:
        ws = [1] * len(fams)
    elif isinstance(weights, dict):
        ws = [weights.get(f, 0) for f in fams]
    else:
        ws = list(weights)
    counts = {f: 0 for f in fams}
    rows = []
    for _ in range(n):
        fam = rng.choices(fams, weights=ws)[0]
        idx = counts[fam]
        counts[fam] += 1
        rows.append(FAMILIES[fam](seed, idx))
    return rows


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3:
        raise SystemExit("usage: python -m jevlab.synth_reason OUT.jsonl N SEED [families]")
    out = Path(argv[0])
    n = int(argv[1])
    seed = argv[2]
    families = argv[3] if len(argv) > 3 else None
    rows = generate(n, seed, families=families)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
