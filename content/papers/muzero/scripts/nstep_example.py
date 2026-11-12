#!/usr/bin/env python3
"""n-step value targets (Section 3, the z_t formula; Appendix G)."""

import numpy as np

from _load import load

m = load("2-nstep-target")

u = [1.0, 0.0, 2.0, 0.0, 1.0]
v = [5.0, 4.0, 3.0, 2.0, 1.0]
print("u =", u)
print("v =", v)
for n, gamma in [(2, 0.5), (3, 0.9), (10, 0.997)]:
    z = m.nstep_targets(u, v, n, gamma)
    print(f"n = {n}, gamma = {gamma}: z = {np.round(z, 4).tolist()}")

print("\nz_1 for n = 2, gamma = 0.5, term by term:")
print("  u[1] =", u[1], " 0.5 * u[2] =", 0.5 * u[2], " 0.25 * v[3] =", 0.25 * v[3])
print("  z_1 =", m.nstep_target(u, v, 1, 2, 0.5))

print("\nWeight of the bootstrap term, gamma^n")
for n, gamma in [(10, 0.997), (5, 0.997), (50, 0.997)]:
    print(f"  n = {n:>2}, gamma = {gamma}: {gamma ** n:.4f}")
