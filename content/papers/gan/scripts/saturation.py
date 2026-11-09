"""Compare the saturating and non-saturating generator losses (Section 3).

Run: python3 scripts/saturation.py
"""

import numpy as np


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


print("logit a of D(G(z)) | D(G(z)) | d/da log(1-D) | d/da (-log D)")
for a in [-8.0, -4.6, -2.0, 0.0, 2.0]:
    d = sigmoid(a)
    sat = -d  # d/da log(1 - sigmoid(a))
    ns = -(1 - d)  # d/da -log(sigmoid(a))
    print(f"{a:18.1f} | {d:.5f} | {sat:14.5f} | {ns:.5f}")

print()
a = -4.6
d = sigmoid(a)
print("D(G(z)) at a = -4.6     =", round(float(d), 4))
print("saturating gradient     =", round(float(-d), 4))
print("non-saturating gradient =", round(float(-(1 - d)), 4))
print("ratio                   =", round(float((1 - d) / d), 1))
