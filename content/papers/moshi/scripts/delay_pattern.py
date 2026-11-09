"""Compute the acoustic-delay examples on the Moshi page.

Run: python3 scripts/delay_pattern.py
"""

import numpy as np

Q, T, tau = 3, 5, 1
# Codes are in 1..N, so 0 is free to act as the initial token of Equation 4.
A = np.arange(1, Q * T + 1).reshape(Q, T) + 10 * np.arange(Q)[:, None]
print("A (rows q = 1..Q, columns t = 1..T):")
print(A)

V = np.zeros_like(A)
V[0] = A[0]
for q in range(1, Q):
    V[q, tau:] = A[q, : T - tau]
print(f"V with tau = {tau} (Equation 4):")
print(V)

frame_ms = 80
for name, delays in [
    ("[0]*8", [0] * 8),
    ("[0, 1, 1, 1, 1, 1, 1, 1]", [0] + [1] * 7),
    ("[0, 2, 2, 2, 2, 2, 2, 2]", [0] + [2] * 7),
    ("[0, 1, 2, 3, 4, 5, 6, 7]", list(range(8))),
]:
    print(f"{name:28s} latency = {(max(delays) + 1) * frame_ms} ms")
