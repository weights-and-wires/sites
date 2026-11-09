"""Compute the numbers for the SSM recurrence-versus-convolution example.

One channel, N = 1, A = -1, B = 1, C = 1, and a step size of ln 2.

Run: python3 scripts/ssm_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

A, B, C, delta = -1.0, 1.0, 1.0, np.log(2.0)
x = np.array([1.0, 2.0, 3.0, 4.0])

# Equation 4 (zero-order hold) for scalar A.
A_bar = np.exp(delta * A)
B_bar = (A_bar - 1.0) / A * B
print(f"delta = ln 2 = {delta:.4f}")
print(f"A_bar = exp(delta*A) = {A_bar:.4f}")
print(f"B_bar = (A_bar - 1)/A * B = {B_bar:.4f}")

# Equation 2: the recurrence.
h = 0.0
ys = []
hs = []
for xt in x:
    h = A_bar * h + B_bar * xt
    hs.append(h)
    ys.append(C * h)
print("recurrence h_t =", [float(v) for v in hs])
print("recurrence y_t =", [float(v) for v in ys])

# Equation 3: the convolution kernel K = (C B_bar, C A_bar B_bar, C A_bar^2 B_bar, ...).
K = np.array([C * A_bar**k * B_bar for k in range(len(x))])
print("kernel K       =", K.tolist())
y_conv = [sum(K[k] * x[t - k] for k in range(t + 1)) for t in range(len(x))]
print("convolution y_t =", [float(v) for v in y_conv])
print("match:", np.allclose(ys, y_conv))

# Cost of the two views for a longer sequence (Section 3.3.2).
print()
print("Per-channel state size is N = 1 here. With N = 16 and D = 4096 channels,")
print("the recurrence carries", 16 * 4096, "numbers between steps.")
