"""A feature-level drafter step (Section 3.1, Figure 6) on toy dimensions, and
why the advanced token matters (Section 1, Figure 3).

Run: python3 scripts/drafter_step.py
"""

import numpy as np

rng = np.random.default_rng(0)
d, V, n = 3, 5, 4
E = rng.normal(size=(V, d))  # frozen embedding table
tokens = np.array([2, 4, 1, 3])  # t_1 ... t_4
F = rng.normal(size=(n, d))  # f_1 ... f_4 from the target

# Position i of the drafter sees f_i and the embedding of t_{i+1}.
inputs = np.concatenate([F[:-1], E[tokens[1:]]], axis=1)
print("feature rows used:", F[:-1].shape, " token ids used:", tokens[1:].tolist())
print("drafter input shape:", inputs.shape, "= (n - 1, 2 d)")

W = rng.normal(size=(2 * d, d)) * 0.3
b = np.zeros(d)
f_hat = inputs @ W + b
print("predicted features shape:", f_hat.shape)

def smooth_l1(a, b, beta=1.0):
    x = np.abs(a - b)
    return float(np.where(x < beta, 0.5 * x**2 / beta, x - 0.5 * beta).mean())

print(f"smooth L1 vs the target's f_2..f_4: {smooth_l1(f_hat, F[1:]):.4f}")
print("smooth L1 at |x| = 0.5 and 3 (beta 1):", smooth_l1(np.array([0.5]), np.array([0.0])), smooth_l1(np.array([3.0]), np.array([0.0])))

# Feature uncertainty: after the same prefix "I", the next token is "am" or
# "always". The next feature depends on that token.
N = 4000
d = 4
E2 = rng.normal(size=(2, d))  # embeddings of the two candidates
A = rng.normal(size=(d, d)) * 0.5
B = rng.normal(size=(d, d)) * 0.8
f_i = rng.normal(size=(N, d))
tok = rng.integers(0, 2, size=N)
f_next = f_i @ A + E2[tok] @ B

def fit_mse(X, Y):
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return float(((X @ W - Y) ** 2).mean())

print(f"feature variance (mean square):         {float((f_next ** 2).mean()):.4f}")
print(f"MSE, predict f_(i+1) from f_i only:     {fit_mse(f_i, f_next):.4f}")
print(f"MSE, from f_i and the next token:       {fit_mse(np.concatenate([f_i, E2[tok]], axis=1), f_next):.6f}")
