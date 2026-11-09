"""Check weight absorption and show why RoPE on the content key breaks it.

Part 1: one head, d_h = 2, d_c = 2. Part 2: random multi-head check with
Equations 9-11 and the absorbed form of Section 2.1.2.
Part 3: a rotation between W_UK and the query makes the absorbed matrix
depend on the relative position.

Run: python3 scripts/absorb_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Part 1: hand-checkable single score
W_UK = np.array([[1.0, 1.0], [0.0, 1.0]])
q = np.array([2.0, 1.0])
c = np.array([1.0, 3.0])
k = W_UK @ c
print("k = W_UK c            =", k.tolist())
print("q . k                 =", float(q @ k))
q_abs = W_UK.T @ q
print("W_UK^T q              =", q_abs.tolist())
print("(W_UK^T q) . c        =", float(q_abs @ c))

# Part 2: random multi-head causal attention, naive vs absorbed
rng = np.random.default_rng(0)
T, d, n_h, d_h, d_c = 6, 8, 2, 4, 3
H = rng.normal(size=(T, d))
WQ = rng.normal(size=(n_h * d_h, d))
WDKV = rng.normal(size=(d_c, d))
WUK = rng.normal(size=(n_h * d_h, d_c))
WUV = rng.normal(size=(n_h * d_h, d_c))
mask = np.tril(np.ones((T, T), dtype=bool))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


C = H @ WDKV.T
Q = (H @ WQ.T).reshape(T, n_h, d_h)
K = (C @ WUK.T).reshape(T, n_h, d_h)
V = (C @ WUV.T).reshape(T, n_h, d_h)
naive = np.zeros((T, n_h, d_h))
absorbed = np.zeros((T, n_h, d_h))
for i in range(n_h):
    s = Q[:, i] @ K[:, i].T / np.sqrt(d_h)
    naive[:, i] = softmax(np.where(mask, s, -1e9)) @ V[:, i]
    WUK_i = WUK[i * d_h:(i + 1) * d_h]
    WUV_i = WUV[i * d_h:(i + 1) * d_h]
    s2 = (Q[:, i] @ WUK_i) @ C.T / np.sqrt(d_h)
    absorbed[:, i] = (softmax(np.where(mask, s2, -1e9)) @ C) @ WUV_i.T
print("max |naive - absorbed| =", float(np.abs(naive - absorbed).max()))
print("floats cached per token: naive keys+values", 2 * n_h * d_h, "latent", d_c)

# Part 3: RoPE between the query and W_UK
def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


print("absorbed matrix W_UK^T R(delta), theta = 1:")
for delta in (0, 1, 2):
    print(f"  delta={delta}:", (W_UK.T @ rot(1.0 * delta)).round(4).tolist())
