"""Trace Algorithm 3 (online normalizer) on a four-element vector.

Run: python3 scripts/trace_example.py
"""

import math

x = [2.0, 1.0, 4.0, 3.0]

m, d = -math.inf, 0.0
print("j  x_j   m_j   d_j (step by step)")
for j, xj in enumerate(x, start=1):
    m_new = max(m, xj)
    scale = math.exp(m - m_new)  # exp(-inf) is 0.0 on the first step
    d = d * scale + math.exp(xj - m_new)
    print(f"{j}  {xj:<4}  {m_new:<4}  d_{j} = d_{j-1} * e^({m} - {m_new}) + e^({xj} - {m_new})"
          f" = {d:.4f}   (rescale factor {scale:.4f})")
    m = m_new

# Check against the safe two-pass definition (Algorithm 2).
m_safe = max(x)
d_safe = sum(math.exp(v - m_safe) for v in x)
print()
print(f"online:  m_V = {m}, d_V = {d:.4f}")
print(f"safe:    m_V = {m_safe}, d_V = {d_safe:.4f}")
y = [math.exp(v - m) / d for v in x]
print("y =", [round(v, 4) for v in y], "sum =", round(sum(y), 6))

# The paper's bound 1 <= d_j <= j (Section 3), checked on random data.
import random

random.seed(0)
ok = True
for _ in range(200):
    xs = [random.gauss(0, 5) for _ in range(50)]
    m, d = -math.inf, 0.0
    for j, xj in enumerate(xs, start=1):
        m_new = max(m, xj)
        d = d * math.exp(m - m_new) + math.exp(xj - m_new)
        m = m_new
        ok = ok and (1.0 - 1e-12 <= d <= j + 1e-12)
print("1 <= d_j <= j held on 200 random vectors of length 50:", ok)
