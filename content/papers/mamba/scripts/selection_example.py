"""Compute the numbers for the selection and gating example (Theorem 1).

Run: python3 scripts/selection_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softplus(z):
    return np.log1p(np.exp(z))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Theorem 1: N = 1, A = -1, B = 1, delta = softplus(z), z = Linear(x_t).
print("z      delta=softplus(z)  A_bar    B_bar    sigmoid(z)  1-sigmoid(z)")
for z in [-4.0, 0.0, 4.0]:
    delta = softplus(z)
    A_bar = np.exp(delta * -1.0)
    B_bar = (A_bar - 1.0) / -1.0
    print(f"{z:5.1f}  {delta:17.4f}  {A_bar:.4f}  {B_bar:.4f}  {sigmoid(z):.4f}      {1 - sigmoid(z):.4f}")

# A toy selective-copying stream: the value to remember is 4.0 at step 1.
# The other three tokens are noise with value 9.0.
x = np.array([9.0, 4.0, 9.0, 9.0])
z_sel = np.array([-8.0, 8.0, -8.0, -8.0])  # the model "sees" which token matters
z_fix = np.full(4, 0.0)  # a time-invariant model: the same gate for every step


def run(z):
    h = 0.0
    trace = []
    for xt, zt in zip(x, z):
        g = sigmoid(zt)
        h = (1.0 - g) * h + g * xt
        trace.append(h)
    return np.array(trace)


print()
print("x                  =", x.tolist())
print("selective gate g_t =", sigmoid(z_sel).round(4).tolist())
print("selective h_t      =", run(z_sel).round(4).tolist())
print("fixed gate g = 0.5 h_t =", run(z_fix).round(4).tolist())

# Large delta resets the state, small delta keeps it (Section 3.5.2).
print()
for delta in [0.01, 1.0, 10.0]:
    print(f"delta = {delta:5.2f}  A_bar = exp(-delta) = {np.exp(-delta):.4f}")
