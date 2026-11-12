"""Derived numbers from Table 1 (support spacing, discounts, schedules).

Run: python3 scripts/hyperparameters.py
"""

import numpy as np

v_min, v_max, atoms, n, gamma = -10.0, 10.0, 51, 3, 0.99
dz = (v_max - v_min) / (atoms - 1)
z = v_min + np.arange(atoms) * dz
print(f"atom spacing = {dz:.2f}, first atoms = {z[:3]}, last = {z[-1]}")
print(f"distribution outputs per state with 18 actions = {atoms * 18}")
print(f"gamma^3 at gamma = 0.99 = {gamma**n:.6f}")
# Largest discounted return a clipped reward stream can reach: 1 / (1 - gamma).
print(f"max discounted return with rewards in [-1, 1] = {1 / (1 - gamma):.1f} (the support stops at {v_max})")
# beta schedule over 200M frames
for frac in (0.0, 0.5, 1.0):
    print(f"beta at {frac:.0%} of training = {0.4 + 0.6 * frac:.2f}")
