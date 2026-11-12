"""Work through row convolution (Section 3.7, Equation 11) and count its parameters.

Run: python3 scripts/row_conv_example.py
"""

import numpy as np


def row_convolution(h, W):
    T, d = h.shape
    tau = W.shape[1] - 1
    padded = np.vstack([h, np.zeros((tau, d))])
    return sum(W[:, j] * padded[j : j + T] for j in range(tau + 1))


# Part 1: d = 1, tau = 2, T = 4.
h = np.array([[1.0], [2.0], [3.0], [4.0]])
W = np.array([[1.0, 10.0, 100.0]])
r = row_convolution(h, W).ravel()
print(f"h = {h.ravel().tolist()}, W = {W.ravel().tolist()} (tau = 2)")
print(f"r = {r.tolist()}")
print("r_0 = 1*1 + 10*2 + 100*3 = 321;  r_3 = 1*4 + 0 + 0 = 4 (zero padding)")

# Part 2: an impulse at t = 3 shows which outputs depend on it.
h = np.zeros((6, 1))
h[3, 0] = 1.0
W = np.array([[1.0, 2.0, 3.0]])
print()
print(f"impulse at t = 3, W = {W.ravel().tolist()}: r = {row_convolution(h, W).ravel().tolist()}")
print("outputs at t = 1, 2, 3 see the impulse; t = 0 and t >= 4 do not")

# Part 3: parameter count of the deployed model (Section 7.4: d = 2560, tau = 19).
d, tau = 2560, 19
row = d * (tau + 1)
fc = d * d
print()
print(f"row convolution with d = {d}, tau = {tau}: {row:,} weights")
print(f"one {d} x {d} fully connected layer: {fc:,} weights")
print(f"ratio = {fc / row:.1f}")
step_ms = 40
print(f"future context if one step is {step_ms} ms: {tau} steps = {tau * step_ms} ms")
