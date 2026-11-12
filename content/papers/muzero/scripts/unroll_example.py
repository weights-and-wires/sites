#!/usr/bin/env python3
"""Unrolling a tiny linear h, g, f model for K = 2 steps (Equation 1)."""

import numpy as np

from _load import load

m = load("4-unrolled-loss")

params = {
    "H": np.eye(2),
    "G": np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 2.0]]),
    "Wp": np.eye(2),
    "wv": np.array([1.0, 1.0]),
}
obs = np.array([1.0, 2.0])
actions = [0, 1]
pi = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
z = [3.0, 5.0, 7.0]
u = [99.0, 0.0, 1.0]  # u[0] is never used

r, p, v = m.unroll(params, obs, actions)
print("rewards r_1, r_2:", r)
print("values v_0..v_2:", v)
print("policies:", [np.round(x, 4).tolist() for x in p])

total = 0.0
for k in range(3):
    lr = (u[k] - r[k - 1]) ** 2 if k > 0 else 0.0
    lv = (z[k] - v[k]) ** 2
    lp = -float(pi[k] @ np.log(p[k]))
    print(f"k = {k}: reward {lr:.4f}, value {lv:.4f}, policy {lp:.4f}")
    total += lr + lv + lp
print(f"sum = {total:.4f}")
print(f"unrolled_loss = {m.unrolled_loss(params, obs, actions, pi, z, u):.4f}")
print(f"scaled by 1/K (K = 2) as in Appendix G: {total / 2:.4f}")
