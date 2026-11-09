"""Compute the numbers for the adaLN-Zero example.

Run: python3 scripts/adaln_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def layer_norm(x, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


# One token with d = 4, condition c = embedding of (t, y).
x = np.array([[1.0, 2.0, 3.0, 4.0]])
print("LayerNorm(x) =", layer_norm(x)[0])

gamma = np.array([0.5, 0.0, -0.5, 1.0])
beta = np.array([1.0, 1.0, 0.0, 0.0])
mod = layer_norm(x) * (1 + gamma) + beta
print("adaLN: LN(x) * (1 + gamma) + beta =", mod[0])
print("gamma = 0, beta = 0 leaves LN(x) unchanged:", np.allclose(layer_norm(x) * 1 + 0, layer_norm(x)))

# Gated residual: x + alpha * F(.). Use F(h) = 2h as a stand-in sublayer.
F = lambda h: 2 * h
for alpha in (0.0, 0.1, 1.0):
    out = x + alpha * F(layer_norm(x))
    print(f"alpha = {alpha}: x + alpha * F(LN(x)) =", out[0])

# Zero init of the Linear(d, 6d) that regresses (gamma1, beta1, alpha1, gamma2, beta2, alpha2).
d = 4
rng = np.random.default_rng(0)
c = rng.normal(size=(d,))
W = np.zeros((d, 6 * d))
b = np.zeros(6 * d)
chunks = np.split(c @ W + b, 6)
print()
print("zero-init regression, alpha1 =", chunks[2], "(all zero for any c)")
print("a c with norm", round(float(np.linalg.norm(c)), 4), "still gives zero parameters")

# Parameter count of the adaLN-Zero regression layer per block.
for name, d in (("S", 384), ("B", 768), ("L", 1024), ("XL", 1152)):
    n = d * 6 * d + 6 * d
    print(f"DiT-{name}: Linear(d, 6d) holds {n:,} parameters per block ({n / 1e6:.2f}M)")
print("adaLN (no gate): Linear(d, 4d) for XL =", 1152 * 4 * 1152 + 4 * 1152)
print("XL: 28 blocks of adaLN-Zero regression =", 28 * (1152 * 6 * 1152 + 6 * 1152))
