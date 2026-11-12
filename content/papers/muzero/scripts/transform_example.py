#!/usr/bin/env python3
"""The invertible value transform h and the categorical support (Appendix F)."""

import numpy as np

from _load import load

m = load("3-value-transform")

print("== h and its inverse")
for x in [-100.0, -3.7, 0.0, 3.7, 100.0, 1000.0, 20000.0]:
    y = float(m.h(x))
    print(f"x = {x:>9.1f}: h(x) = {y:>9.4f}, h_inv(h(x)) = {float(m.h_inv(y)):>9.1f}")

print("\n== Raw values the 601-entry support can represent")
print(f"h_inv(300) = {float(m.h_inv(300.0)):.1f}")
print(f"h_inv(-300) = {float(m.h_inv(-300.0)):.1f}")

print("\n== Support encoding of 3.7 without the transform (the paper's example)")
print(np.round(m.phi(3.7, 0, 5), 4).tolist())

print("\n== Target for x = 100 on the full support")
t = m.to_target(100.0)
idx = np.nonzero(t)[0]
print("h(100) =", round(float(m.h(100.0)), 4))
print("nonzero supports:", [int(i - 300) for i in idx], "weights:", np.round(t[idx], 4).tolist())
print("decoded back:", round(m.from_logits(np.log(t + 1e-30)), 4))

print("\n== Mean before inverting (support 0 and 10, equal mass)")
logits = np.full(11, -50.0)
logits[0] = logits[10] = 0.0
print("h_inv(mean) =", round(m.from_logits(logits, lo=0), 4))
print("mean of h_inv =", round(0.5 * (float(m.h_inv(0.0)) + float(m.h_inv(10.0))), 4))
