"""Prints the numbers for "Intermediate tokens as extra computation".

A toy model, not a language model: one forward pass applies at most `budget`
integer operations and slips (adds 1) with probability eps. Answer-only uses one
pass. Stepwise uses ceil(k / budget) passes, each writing its value. The closed
form for stepwise is (1 - eps) ** ceil(k / budget). Mirrors exercise 2.

Run: python3 content/papers/cot/scripts/budget_curve.py
"""

import math

import numpy as np

NAMES = ("add", "sub", "mul")
BUDGET, EPS, N, SEED = 1, 0.05, 20000, 0


def apply_op(v, op):
    name, n = op
    return v + n if name == "add" else v - n if name == "sub" else v * n


def make_problem(rng, k):
    start = int(rng.integers(1, 10))
    ops = []
    for _ in range(k):
        name = NAMES[int(rng.integers(0, 3))]
        ops.append((name, int(rng.integers(2, 5)) if name == "mul" else int(rng.integers(1, 10))))
    return start, ops


def answer_only(start, ops, budget, eps, rng):
    v = start
    for op in ops[:budget]:
        v = apply_op(v, op)
    return v + (1 if rng.random() < eps else 0)


def stepwise(start, ops, budget, eps, rng):
    v = start
    for i in range(0, len(ops), budget):
        for op in ops[i : i + budget]:
            v = apply_op(v, op)
        if rng.random() < eps:
            v += 1
    return v


rng = np.random.default_rng(SEED)
print(f"budget {BUDGET} operation per pass, slip probability {EPS}, {N} problems per k, seed {SEED}")
print(f"{'k':>2} {'answer-only':>12} {'stepwise':>9} {'closed form':>12} {'passes':>7}")
for k in range(1, 9):
    direct = chain = 0
    for _ in range(N):
        start, ops = make_problem(rng, k)
        truth = start
        for op in ops:
            truth = apply_op(truth, op)
        direct += answer_only(start, ops, BUDGET, EPS, rng) == truth
        chain += stepwise(start, ops, BUDGET, EPS, rng) == truth
    passes = math.ceil(k / BUDGET)
    print(f"{k:>2} {direct / N:>12.3f} {chain / N:>9.3f} {(1 - EPS) ** passes:>12.3f} {passes:>7}")
