"""Compute the numbers for the RMSNorm example.

Run: python3 scripts/rmsnorm_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def rmsnorm(x, g=None, eps=0.0):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    y = x / rms
    return y if g is None else y * g


def layernorm(x, eps=0.0):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


x = np.array([1.0, 2.0, 3.0, 4.0])
print("x =", x.tolist())
print("mean square =", np.mean(x**2), " RMS =", round(float(np.sqrt(np.mean(x**2))), 4))
print("RMSNorm(x)   =", rmsnorm(x).tolist())
print("LayerNorm(x) =", layernorm(x).tolist())
print("RMSNorm(10x) =", rmsnorm(10 * x).tolist(), "(same as RMSNorm(x))")
print("mean of RMSNorm(x) =", round(float(rmsnorm(x).mean()), 4), "; mean square of RMSNorm(x) =", round(float(np.mean(rmsnorm(x) ** 2)), 4))

s = x + 10
print()
print("shifted x + 10 =", s.tolist())
print("LayerNorm(x + 10) =", layernorm(s).tolist(), "(same as LayerNorm(x))")
print("RMSNorm(x + 10)   =", rmsnorm(s).tolist(), "(changes)")

# For a vector with zero mean the two normalizers agree.
z = np.array([-3.0, -1.0, 1.0, 3.0])
print()
print("zero-mean z =", z.tolist())
print("RMSNorm(z)   =", rmsnorm(z).tolist())
print("LayerNorm(z) =", layernorm(z).tolist(), "(equal)")

# Pre-norm: the branch sees the same input no matter how large the stream is.
big = 100 * x
print()
print("pre-norm branch input, stream x     =", rmsnorm(x).tolist())
print("pre-norm branch input, stream 100 x =", rmsnorm(big).tolist())
branch = rmsnorm(x)  # stand-in: the branch output equals its normalized input
print("update size / stream size, stream x     =", round(float(np.linalg.norm(branch) / np.linalg.norm(x)), 4))
print("update size / stream size, stream 100 x =", round(float(np.linalg.norm(branch) / np.linalg.norm(big)), 4))

# Parameter counts per norm layer and for LLaMA-7B (Table 2: d = 4096, 32 layers).
d, layers = 4096, 32
norms = 2 * layers + 1
print()
print("RMSNorm params per layer: d =", d, "; LayerNorm with bias: 2d =", 2 * d)
print("norm layers in a 32-layer model (2 per block + 1 final):", norms)
print("RMSNorm total:", norms * d, "; LayerNorm total:", norms * 2 * d)
