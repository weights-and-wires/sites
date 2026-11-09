"""Compute the numbers for the Transformer block example.

Run: python3 scripts/block_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# LayerNorm of one vector (gamma = 1, beta = 0, eps = 0 to keep the arithmetic exact).
x = np.array([1.0, 2.0, 3.0, 4.0])
mu, var = x.mean(), x.var()
print(f"x = {x.tolist()}, mean = {mu}, variance = {var}")
print("LayerNorm(x) =", ((x - mu) / np.sqrt(var)).tolist())
y = np.array([10.0, 20.0, 30.0, 40.0])
print("LayerNorm(10x) =", ((y - y.mean()) / y.std()).tolist(), "(same: scale does not matter)")

# Position-wise FFN, Equation 2, with d_model = 2, d_ff = 3.
W1 = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]])
b1 = np.array([0.0, 0.0, -1.0])
W2 = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]])
b2 = np.array([0.5, 0.0])
t = np.array([1.0, 2.0])
hidden = t @ W1 + b1
act = np.maximum(0, hidden)
print()
print("FFN input        =", t.tolist())
print("x W1 + b1        =", hidden.tolist())
print("after ReLU       =", act.tolist())
print("output           =", (act @ W2 + b2).tolist())

# Post-LN versus pre-LN on one sublayer with Sublayer(x) = 2x (a stand-in).
def ln(v):
    return (v - v.mean()) / v.std()

sub = lambda v: 2 * v
big = np.array([100.0, 0.0, 0.0, 0.0])
print()
print("residual stream x =", big.tolist())
print("post-LN: LayerNorm(x + Sublayer(x)) =", ln(big + sub(big)).tolist())
print("pre-LN:  x + Sublayer(LayerNorm(x)) =", (big + sub(ln(big))).tolist())
print("pre-LN keeps the raw x term on the identity path; post-LN rescales it every layer.")

# Base-model FFN size (Section 3.3: d_model = 512, d_ff = 2048).
d_model, d_ff = 512, 2048
print()
print("FFN weights:", d_model * d_ff + d_ff * d_model, "; with biases:", d_model * d_ff + d_ff + d_ff * d_model + d_model)
