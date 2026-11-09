"""Equations 1 to 3 on one group of G = 8 completions with Equation 4 rewards,
and the clip range for the epsilon that Section 3.2.1 reports."""

import numpy as np

from r1_lib import group_advantages

rewards = np.array([2, 2, 1, 1, 0, 0, 2, 1], dtype=float)
adv = group_advantages(rewards)
print("rewards   ", rewards.tolist())
print("mean, std ", rewards.mean(), round(rewards.std(), 4))
print("advantages", np.round(adv, 4).tolist())
print("sum of advantages", round(adv.sum(), 6))

print()
# Once every completion follows the format, the format reward is a constant
# shift of the group, and the advantages equal those of accuracy alone.
acc = np.array([1, 1, 0, 0, 0, 0, 1, 0], dtype=float)
print("accuracy only      ", np.round(group_advantages(acc), 4).tolist())
print("accuracy + 1 format", np.round(group_advantages(acc + 1), 4).tolist())

print()
# Clip interval of Equation 1 for epsilon = 0.2 and for epsilon = 10 (Section 3.2.1).
for eps in (0.2, 10):
    print(f"epsilon = {eps}: clip(ratio, {1 - eps}, {1 + eps})")
ratio = 1.5
for eps in (0.2, 10):
    clipped = float(np.clip(ratio, 1 - eps, 1 + eps))
    A = 1.0
    print(f"epsilon = {eps}, ratio = {ratio}, A = +1: min(ratio*A, clip*A) = {min(ratio * A, clipped * A)}")
