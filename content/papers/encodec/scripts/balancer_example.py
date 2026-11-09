"""The loss balancer of Section 3.4, Equation 5, on a small example.

Run: python3 scripts/balancer_example.py
"""

import numpy as np

# Weights from Section 4.4 for the 24 kHz models.
names = ["time L1", "freq", "adv", "feat"]
lam = np.array([0.1, 1.0, 3.0, 3.0])
print(f"sum of weights = {lam.sum():.1f}")
print("fraction of the gradient each loss gets:", {n: round(float(f), 4) for n, f in zip(names, lam / lam.sum())})

# Two losses whose gradients with respect to x_hat have very different scales.
g = {"rec": np.array([3.0, 4.0]), "adv": np.array([0.0, 0.02])}
w = {"rec": 1.0, "adv": 1.0}
R = 1.0

plain = sum(w[k] * g[k] for k in g)
print(f"\nplain weighted sum of gradients: {plain}, norm {np.linalg.norm(plain):.4f}")
print(f"  share of the norm from rec: {np.linalg.norm(g['rec']) / (np.linalg.norm(g['rec']) + np.linalg.norm(g['adv'])):.4f}")

tot = sum(w.values())
tilde = {k: R * w[k] / tot * g[k] / np.linalg.norm(g[k]) for k in g}
print("balanced pieces:")
for k in g:
    print(f"  {k}: {tilde[k]}, norm {np.linalg.norm(tilde[k]):.4f}")
bal = sum(tilde.values())
print(f"balanced sum: {bal}")

# A loss rescaled by 1000 leaves the balanced gradient unchanged.
g2 = {"rec": g["rec"] * 1000, "adv": g["adv"]}
tilde2 = {k: R * w[k] / tot * g2[k] / np.linalg.norm(g2[k]) for k in g2}
print(f"after multiplying the rec loss by 1000, balanced sum is {sum(tilde2.values())} (same)")

# EMA of the norm with beta = 0.999: a sudden jump in scale is tracked slowly.
beta = 0.999
ema = 1.0
for step in range(1, 1001):
    ema = beta * ema + (1 - beta) * 10.0
    if step in (1, 100, 1000):
        print(f"norm jumps from 1 to 10: EMA after {step} steps = {ema:.4f}")
