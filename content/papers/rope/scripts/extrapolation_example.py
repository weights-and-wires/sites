"""Compute the numbers for the context-extension example.

The paper has no extrapolation method. This script computes the two common
rescalings of the angles: position interpolation (Chen et al., 2023) and
NTK-aware base scaling (a community method, later folded into YaRN).

Run: python3 scripts/extrapolation_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

d, base = 8, 10000.0
train_len, new_len = 2048, 8192
s = new_len / train_len


def thetas(base_):
    return base_ ** (-2.0 * np.arange(d // 2) / d)


th = thetas(base)
print("original thetas:       ", th)
print("scale factor s =", s)

# Position interpolation: m -> m / s, so the largest new position maps to the largest trained one.
print("position 8191 -> ", 8191 / s)
print("angle of pair 1 at 8191 (original):    ", 8191 * th[0])
print("angle of pair 1 at 8191 (interpolated):", 8191 / s * th[0])
print("adjacent-position angle gap, pair 1 (interpolated):", th[0] / s)

# NTK-aware: base' = base * s^(d/(d-2)).
base2 = base * s ** (d / (d - 2))
th2 = thetas(base2)
print()
print("NTK-aware base:", base2)
print("NTK-aware thetas:      ", th2)
print("ratio th2/th (pair 1..4):", th2 / th)
print("lowest frequency ratio equals 1/s:", th2[-1] / th[-1], 1 / s)

# Wavelengths against the trained length for d = 128.
d2 = 128
t = base ** (-2.0 * np.arange(d2 // 2) / d2)
wl = 2 * np.pi / t
print()
print("d = 128: pairs with wavelength shorter than 2048 tokens:", int((wl < 2048).sum()), "of", d2 // 2)
print("d = 128: longest wavelength:", wl[-1])
