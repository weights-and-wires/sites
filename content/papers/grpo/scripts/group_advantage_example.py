"""Group-relative advantages (Section 4.1.2) on a group of G = 8 samples."""

import numpy as np

from grpo_lib import group_advantages

r = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
adv = group_advantages(r)
print("rewards        ", r)
print("mean, std      ", r.mean(), round(float(r.std()), 4))
print("advantages     ", np.round(adv, 4))
print("adv mean, std  ", round(float(adv.mean()), 6), round(float(adv.std()), 4))
print("correct sample advantage", round(float(adv[0]), 4), "| wrong sample advantage", round(float(adv[1]), 4))

print()
print("Scale and shift invariance:")
print("advantages of 10 * r + 3 equal advantages of r:", bool(np.allclose(group_advantages(10 * r + 3), adv)))

print()
print("A group where every sample earns the same reward:")
print("rewards        ", np.ones(8))
print("advantages     ", group_advantages(np.ones(8)))

print()
print("Probability that all G = 8 samples get the same 0/1 reward, by success rate p:")
G = 8
for p in [0.02, 0.05, 0.1, 0.5, 0.9, 0.98]:
    dead = p**G + (1 - p) ** G
    print(f"p = {p:<5} P(no signal) = {dead:.4f}")
