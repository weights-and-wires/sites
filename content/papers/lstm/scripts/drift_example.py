"""Compute the numbers for the internal-state drift example (Section 4.8).

Run: python3 scripts/drift_example.py
"""

import numpy as np


def f(x):
    return 1.0 / (1.0 + np.exp(-x))


def h_prime(x):
    # h(x) = 2 f(x) - 1, so h'(x) = 2 f(x) (1 - f(x)).
    return 2.0 * f(x) * (1.0 - f(x))


print("h'(s) as the internal state drifts away from 0:")
for s in (0.0, 2.0, 5.0, 10.0):
    print(f"  s = {s:4.1f}: h'(s) = {h_prime(s):.3e}")

# Drift: a cell whose input is always about +0.5 (an input gate at 1 and g(net) = 0.5).
s = 0.0
for t in range(1, 101):
    s += 0.5
    if t in (10, 50, 100):
        print(f"after {t:3d} steps of +0.5: s = {s:5.1f}, h'(s) = {h_prime(s):.3e}")

# The remedy in Section 4.8 and Section 5.4.3: bias the input gate toward 0.
for bias in (0.0, -3.0, -6.0):
    i_gate = f(bias)
    s = 0.0
    for t in range(100):
        s += i_gate * 0.5
    print(f"input gate bias {bias:5.1f}: i_in = {i_gate:.5f}; s after 100 steps = {s:.3f}, h'(s) = {h_prime(s):.3e}")
