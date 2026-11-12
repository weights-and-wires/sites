"""Best-of-N on one problem: ORM against PRM against voting (Sections 2.1, 2.5, 2.6).

Five sampled solutions to one problem. The correct final answer is 12. The ORM
score is its prediction at the final token. The PRM gives one probability per step.

Run: python3 scripts/best_of_n_toy.py
"""

from collections import Counter

import numpy as np

CORRECT = 12
# (final answer, ORM score, PRM step probabilities)
candidates = [
    (12, 0.80, [0.95, 0.90, 0.92]),         # c0: right answer, sound reasoning
    (15, 0.85, [0.97, 0.95, 0.20, 0.90]),   # c1: wrong, fluent, one bad step
    (15, 0.60, [0.90, 0.93, 0.30]),         # c2: wrong
    (12, 0.70, [0.90, 0.30, 0.90]),         # c3: right answer, flawed reasoning
    (15, 0.82, [0.96, 0.90, 0.40, 0.90]),   # c4: wrong
]
answers = [c[0] for c in candidates]
orm = np.array([c[1] for c in candidates])
prm = np.array([np.prod(c[2]) for c in candidates])

print("idx answer  ORM    PRM product")
for i, (a, o, steps) in enumerate(candidates):
    print(f"c{i}  {a:5d}  {o:.2f}   {prm[i]:.4f}")

print()
orm_pick = int(np.argmax(orm))
prm_pick = int(np.argmax(prm))
vote = Counter(answers).most_common(1)[0][0]
print(f"ORM best-of-5 picks c{orm_pick}, answer {answers[orm_pick]}, "
      f"correct={answers[orm_pick] == CORRECT}")
print(f"PRM best-of-5 picks c{prm_pick}, answer {answers[prm_pick]}, "
      f"correct={answers[prm_pick] == CORRECT}")
print(f"Majority vote picks answer {vote}, correct={vote == CORRECT}")

print()
for name, scores in (("ORM", orm), ("PRM", prm)):
    totals = {}
    for a, s in zip(answers, scores):
        totals[a] = totals.get(a, 0.0) + float(s)
    best = max(totals, key=totals.get)
    text = ", ".join(f"{a}: {t:.4f}" for a, t in totals.items())
    print(f"{name}-weighted vote totals: {text} -> answer {best}, correct={best == CORRECT}")

print()
print("c3 has the right answer and a step at 0.30; the PRM gives it "
      f"{prm[3]:.4f}, so the PRM ranks it {int((prm > prm[3]).sum()) + 1} of 5.")
