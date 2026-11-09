"""A few-shot evaluator over a tiny task with a stub model (Section 2, Section 2.4).

The task asks for a made-up operation, written "blip", that the stub model can
learn only from the demonstrations in its prompt. Here "blip" means
multiplication. The stub tries +, -, and * on each demonstration and applies the
first operator that matches all of them. With no demonstration it has nothing to
infer from and answers 0. No weights change between calls, so accuracy that
rises with K is in-context learning in the paper's sense. The task and the stub
are invented; they are not from the paper.

Run: python3 scripts/fewshot_eval.py
"""

import re

import numpy as np

OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


def format_prompt(demos, query, description=None):
    head = f"{description}\n\n" if description else ""
    return head + "".join(f"Q: {q}\nA: {a}\n\n" for q, a in demos) + f"Q: {query}\nA:"


def stub_model(prompt):
    pairs = re.findall(r"Q: What is (\d+) blip (\d+)\?\nA: (-?\d+)\n", prompt)
    final = re.findall(r"Q: What is (\d+) blip (\d+)\?\nA:$", prompt)[0]
    a, b = int(final[0]), int(final[1])
    for sym, f in OPS.items():
        if pairs and all(f(int(x), int(y)) == int(z) for x, y, z in pairs):
            return " " + str(f(a, b))
    return " 0"


def make(n, rng):
    out = []
    for _ in range(n):
        a, b = rng.integers(3, 20, size=2)
        out.append((f"What is {a} blip {b}?", str(a * b)))
    return out


def evaluate(model, dataset, pool, k, seed=0):
    rng = np.random.default_rng(seed)
    right = 0
    for q, a in dataset:
        ok = [i for i, (pq, _) in enumerate(pool) if pq != q]
        idx = rng.choice(ok, size=k, replace=False)
        prompt = format_prompt([pool[i] for i in idx], q)
        right += model(prompt).strip() == a
    return right / len(dataset)


rng = np.random.default_rng(1)
pool, test = make(30, rng), make(50, rng)
print(format_prompt(pool[:2], test[0][0]))
print()
for k in (0, 1, 2, 5, 10):
    print(f"K = {k:>2}: exact-match accuracy {evaluate(stub_model, test, pool, k):.2f}")
