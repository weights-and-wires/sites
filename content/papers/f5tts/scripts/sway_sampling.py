"""Sway Sampling of flow steps (Section 3.2, Equation 7).

Run: python3 scripts/sway_sampling.py
"""

import numpy as np


def sway(u, s):
    return u + s * (np.cos(np.pi / 2 * u) - 1 + u)


s_max = 2 / (np.pi - 2)
print(f"allowed range: s in [-1, 2/(pi-2)] = [-1, {s_max:.4f}]")

print()
u = np.linspace(0, 1, 9)  # 8 steps (NFE = 8)
print("u          ", np.round(u, 4))
for s in (0.0, -0.8, -1.0, s_max):
    print(f"s = {s:+.4f}", np.round(sway(u, s), 4))

print()
print("s = -1 closed form: t = 1 - cos(pi/2 * u)")
print("u = 0.5 ->", round(1 - np.cos(np.pi / 4), 4), "; sway gives", round(float(sway(0.5, -1.0)), 4))

print()
print("Monotonic on the boundary values of s (minimum slope over a fine grid):")
fine = np.linspace(0, 1, 100001)
for s in (-1.0, 0.0, s_max):
    slope = np.diff(sway(fine, s)).min() / (fine[1] - fine[0])
    print(f"s = {s:+.4f}  min slope = {slope:.6f}")
for s in (-1.2, 2.0):
    t = sway(fine, s)
    print(f"s = {s:+.4f}  monotonic: {bool(np.all(np.diff(t) >= 0))}")

print()
print("Steps below t = 0.5 with 16 steps (NFE = 16):")
u16 = np.linspace(0, 1, 17)
for s in (0.0, -0.4, -0.8, -1.0, 0.5):
    t = sway(u16, s)
    print(f"s = {s:+.1f}  steps with t < 0.5: {int((t[:-1] < 0.5).sum())} of 16   first step size: {t[1] - t[0]:.4f}   last: {t[-1] - t[-2]:.4f}")

print()
print("Step sizes for s = -1, NFE = 8:", np.round(np.diff(sway(u, -1.0)), 4))
