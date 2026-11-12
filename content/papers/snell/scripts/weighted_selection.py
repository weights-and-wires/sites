"""Step-wise and inter-answer aggregation with a PRM (Section 5.1).

Part 1 turns per-step PRM scores into one score per solution: last step,
minimum, and product. Part 2 picks a final answer from several scored
solutions three ways: best-of-N, majority vote, and best-of-N weighted.

Run: python3 scripts/weighted_selection.py
"""

import numpy as np


def pick_best_of_n(answers, scores):
    return answers[int(np.argmax(scores))]


def pick_majority(answers):
    counts = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    return max(counts, key=counts.get)


def pick_weighted(answers, scores):
    totals = {}
    for a, s in zip(answers, scores):
        totals[a] = totals.get(a, 0.0) + float(s)
    return max(totals, key=totals.get), totals


print("Part 1: step-wise aggregation of one solution")
for steps in ([0.90, 0.80, 0.70], [0.90, 0.20, 0.80]):
    print(f"  steps {steps}: last={steps[-1]:.2f} min={min(steps):.2f} prod={np.prod(steps):.3f}")

problems = [
    ("Problem A (correct answer 12)", 12, [7, 12, 12, 12, 9], [0.90, 0.55, 0.50, 0.45, 0.60]),
    ("Problem B (correct answer 8)", 8, [5, 5, 5, 8, 8], [0.20, 0.30, 0.20, 0.80, 0.70]),
]
print()
print("Part 2: inter-answer aggregation")
for name, correct, answers, scores in problems:
    bon = pick_best_of_n(answers, scores)
    maj = pick_majority(answers)
    wtd, totals = pick_weighted(answers, scores)
    print(f"  {name}")
    print(f"    answers {answers}")
    print(f"    scores  {scores}")
    print(f"    best-of-N picks {bon} (correct={bon == correct})")
    print(f"    majority picks {maj} (correct={maj == correct})")
    totals_text = ", ".join(f"{a}: {t:.2f}" for a, t in totals.items())
    print(f"    weighted totals {totals_text} -> picks {wtd} (correct={wtd == correct})")
