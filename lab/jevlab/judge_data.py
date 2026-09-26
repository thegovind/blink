"""Judge-tier style training items (JevBench v1.0 judge tier = math-answer correctness + task-prompt routing).

 * math_judge: GSM8K TRAIN problems with their worked solution; half kept correct, half with one corrupted
   arithmetic result propagated to the final answer (or only the final answer changed). noul: is the saved
   answer correct? Also from our programmatic word_math rows (answer-only form).
 * task_route: Dolly-15k instructions routed to their task category (8 categories + 'other' from a second source).
  python -m jevlab.judge_data OUT_DIR
"""
import json
import random
import re
import sys
from pathlib import Path

RNG = random.Random("judge-data-v1")
Q_JUDGE = ["Is the saved answer to this problem correct?", "Does the saved solution reach the correct final answer?",
           "Should the saved answer be marked correct?", "Is the recorded final answer right?"]


def gsm_items():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for i, r in enumerate(ds):
        sol, ans = r["answer"].split("####")
        ans = ans.strip().replace(",", "")
        steps = sol.strip()
        correct = RNG.random() < 0.5
        shown_sol, shown_ans = steps, ans
        try:
            float(ans)
        except ValueError:
            continue
        if not correct:
            nums = list(re.finditer(r"<<([^=<>]+)=(-?\d+(?:\.\d+)?)>>(-?\d+(?:\.\d+)?)", steps))
            try:
                gold = float(ans)
            except ValueError:
                continue
            delta = RNG.choice([1, 2, 3, 5, 10, -1, -2, -5])
            if nums and RNG.random() < 0.6:
                m = RNG.choice(nums)
                bad = str(int(float(m.group(3)) + delta)) if float(m.group(3)).is_integer() else str(round(float(m.group(3)) + delta, 2))
                shown_sol = steps[: m.start()] + f"<<{m.group(1)}={bad}>>{bad}" + steps[m.end():]
                shown_ans = str(int(gold + delta)) if gold.is_integer() else str(round(gold + delta, 2))
            else:
                shown_ans = str(int(gold + delta)) if gold.is_integer() else str(round(gold + delta, 2))
            if shown_ans == ans:
                continue
        form = RNG.random()
        if form < 0.5:
            state = f"Problem: {r['question']}\n\nSaved solution:\n{shown_sol}\nFinal answer: {shown_ans}"
        else:
            state = f"Problem: {r['question']}\n\nSaved answer: {shown_ans}"
        out.append({"id": f"gsmjudge:{i}", "src": "judge_math_gsm", "state": state,
                    "question": {"type": "noul", "instructions": RNG.choice(Q_JUDGE),
                                 "criteria": {"true": "The saved answer is correct.", "false": "The saved answer is wrong."}},
                    "gold": "yes" if correct else "no", "target": None, "meta": {"group": f"gsm:{i}"}})
    return out


def wordmath_items(path, n=4000):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["meta"].get("family") == "word_math"]
    RNG.shuffle(rows)
    out = []
    for r in rows[:n]:
        crit = r["question"]["criteria"]
        correct = RNG.random() < 0.5
        val = crit[r["gold"]] if correct else crit[RNG.choice([k for k in crit if k != r["gold"]])]
        out.append({"id": f"wmjudge:{r['id']}", "src": "judge_math_synth", "state": f"Problem: {r['state']['question']}\n\nSaved answer: {val}",
                    "question": {"type": "noul", "instructions": RNG.choice(Q_JUDGE),
                                 "criteria": {"true": "The saved answer is correct.", "false": "The saved answer is wrong."}},
                    "gold": "yes" if correct else "no", "target": None, "meta": {"group": r["meta"]["group"]}})
    return out


CAT = {"open_qa": "Open question answering: a factual or general question with no reference text",
       "general_qa": "General advice or explanation request",
       "classification": "Classify or categorise the given items",
       "closed_qa": "Answer a question using only a supplied reference passage",
       "brainstorming": "Generate ideas or a list of options",
       "information_extraction": "Extract specific facts or fields from a supplied passage",
       "summarization": "Summarise a supplied passage",
       "creative_writing": "Write a story, poem or other creative text"}


def dolly_items(n=6000):
    from datasets import load_dataset
    ds = list(load_dataset("databricks/databricks-dolly-15k", split="train"))
    RNG.shuffle(ds)
    out = []
    for i, r in enumerate(ds[:n]):
        if r["category"] not in CAT:
            continue
        prompt = r["instruction"] + (("\n\n" + r["context"][:1500]) if r["context"] else "")
        keys = list(CAT)
        RNG.shuffle(keys)
        out.append({"id": f"dolly:{i}", "src": "judge_route_dolly", "state": prompt,
                    "question": {"type": "choice", "instructions": "Route this task prompt to the category of work it asks for.",
                                 "criteria": {k: CAT[k] for k in keys}},
                    "gold": r["category"], "target": None, "meta": {"group": f"dolly:{i}"}})
    return out


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("judge_gsm", gsm_items()), ("judge_wm", wordmath_items("data/syn2-dd/reason.jsonl")),
                       ("judge_route", dolly_items())):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows))
