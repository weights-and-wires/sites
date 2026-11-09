"""Compute the numbers for the classifier-free guidance example.

Run: python3 scripts/cfg_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Section 3.1: eps_hat = eps(x, null) + s * (eps(x, c) - eps(x, null)).
eps_u = np.array([0.2, -0.4, 1.0, 0.5])
eps_c = np.array([0.5, -0.1, 0.6, 0.9])
for s in (1.0, 1.5, 4.0):
    print(f"s = {s}:", eps_u + s * (eps_c - eps_u))

# Appendix A: guidance on the first three of four channels only.
# The model output has 2C channels: C for noise, C for covariance.
C = 4
out_c = np.arange(1, 2 * C + 1, dtype=float)          # conditional output
out_u = np.arange(1, 2 * C + 1, dtype=float) * 0.5    # unconditional output
s = 1.5
eps_c, eps_u = out_c[:C], out_u[:C]
guided = eps_c.copy()
guided[:3] = eps_u[:3] + s * (eps_c[:3] - eps_u[:3])
print()
print("conditional eps  =", eps_c)
print("unconditional    =", eps_u)
print("guided on 3 chan =", guided)
print("guided on all 4  =", eps_u + s * (eps_c - eps_u))
print("covariance half comes from the conditional pass:", out_c[C:])
