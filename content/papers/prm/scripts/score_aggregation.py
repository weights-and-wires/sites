"""Solution scores from step scores: product against minimum (Section 2.6, Appendix F.2).

Run: python3 scripts/score_aggregation.py
"""

import numpy as np


def solution_score(step_scores, reduction):
    return float(np.prod(step_scores) if reduction == "product" else np.min(step_scores))


print("Solutions as lists of per-step probabilities of being correct")
solutions = {
    "A (3 steps)": [0.95, 0.90, 0.92],
    "B (error in step 3)": [0.99, 0.99, 0.40, 0.99],
    "short (2 steps)": [0.90, 0.95],
    "long (10 steps)": [0.98] * 10,
}
for name, steps in solutions.items():
    print(
        f"{name:22s} product={solution_score(steps, 'product'):.4f}  "
        f"min={solution_score(steps, 'min'):.4f}"
    )

print()
print("Length bias: which of short and long does each reduction prefer?")
for reduction in ("product", "min"):
    s_short = solution_score(solutions["short (2 steps)"], reduction)
    s_long = solution_score(solutions["long (10 steps)"], reduction)
    pick = "short" if s_short > s_long else "long"
    print(f"{reduction:8s} short={s_short:.4f} long={s_long:.4f} -> picks {pick}")

print()
print("Neutral labels: p_pos=0.50, p_neu=0.30, p_neg=0.20 for one step")
p_pos, p_neu = 0.50, 0.30
print(f"neutral counted as positive: step score {p_pos + p_neu:.2f}")
print(f"neutral counted as negative: step score {p_pos:.2f}")
print("Over four identical steps, the product is "
      f"{(p_pos + p_neu) ** 4:.4f} against {p_pos ** 4:.4f}.")
