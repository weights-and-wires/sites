"""Show zero advantages in uniform groups and the cost of filtering them
(Section 3.2, Equations 9 and 11; Section 4.1 for G = 16 and a batch of 512).

Run: python3 scripts/dynamic_sampling_example.py
"""

import numpy as np


def advantages(r):
    r = np.asarray(r, dtype=float)
    s = r.std()
    return np.zeros_like(r) if s == 0 else (r - r.mean()) / s


print("Group advantages with rewards in {+1, -1} (Equation 7), G = 4")
for r in ([1, 1, 1, 1], [-1, -1, -1, -1], [1, -1, -1, -1], [1, 1, -1, -1]):
    print(f"rewards {r}: advantages {np.round(advantages(r), 4).tolist()}")

G = 16
print()
print(f"Chance that a group of G = {G} samples is uniform (all correct or all wrong)")
print("pass rate p   P(uniform) = p^G + (1-p)^G")
for p in (0.01, 0.1, 0.5, 0.9, 0.99):
    print(f"{p:<12}  {p**G + (1 - p) ** G:.5f}")

print()
print("Draws needed to fill a batch of 512 kept prompts (seeded pool)")
rng = np.random.default_rng(0)
batch, kept, drawn = 512, 0, 0
rates = rng.beta(0.4, 0.4, size=100000)  # many prompts are nearly always right or wrong
for p in rates:
    drawn += 1
    c = rng.binomial(G, p)
    if 0 < c < G:
        kept += 1
        if kept == batch:
            break
print(f"kept {kept} groups after drawing {drawn} prompts, kept fraction {kept / drawn:.4f}")
print(f"extra sampling factor {drawn / kept:.2f}x")
