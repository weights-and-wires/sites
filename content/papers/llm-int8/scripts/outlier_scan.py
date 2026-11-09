"""Synthetic hidden states with systematic outlier dimensions, and what they do to int8 error.

The data is synthetic: it mimics the pattern in Section 4 (a few feature
dimensions with large values in most tokens) and is not the paper's data.

Run: python3 scripts/outlier_scan.py
"""

import numpy as np

rng = np.random.default_rng(0)
s, h, o = 512, 1024, 256
outlier_dims = [37, 411, 900]


def make_hidden(magnitude):
    X = rng.normal(0, 1.0, size=(s, h))
    for d in outlier_dims:
        X[:, d] = magnitude * (1 + 0.1 * rng.normal(size=s))  # same sign in every token
    return X


def vectorwise_matmul(X, W):
    cx = 127.0 / np.max(np.abs(X), axis=1)
    cw = 127.0 / np.max(np.abs(W), axis=0)
    Xq = np.rint(X * cx[:, None]).astype(np.int8)
    Wq = np.rint(W * cw[None, :]).astype(np.int8)
    return (Xq.astype(np.int32) @ Wq.astype(np.int32)) / np.outer(cx, cw)


def llm_int8_matmul(X, W, alpha=6.0):
    O = np.where(np.max(np.abs(X), axis=0) >= alpha)[0]
    reg = np.setdiff1d(np.arange(X.shape[1]), O)
    return X[:, O] @ W[O, :] + vectorwise_matmul(X[:, reg], W[reg, :])


W = rng.normal(0, 0.05, size=(h, o))


def rel_err(approx, exact):
    return float(np.linalg.norm(approx - exact) / np.linalg.norm(exact))


print(f"X: {s} tokens x {h} features; outlier dims {outlier_dims}; W: {h} x {o}")
print(f"{'magnitude':>9} {'vector-wise':>12} {'decomposed':>11} {'dims >= 6':>10} {'values flushed to 0':>20}")
for mag in (0.0, 10.0, 20.0, 60.0):
    X = make_hidden(mag) if mag else rng.normal(0, 1.0, size=(s, h))
    exact = X @ W
    ev = rel_err(vectorwise_matmul(X, W), exact)
    ed = rel_err(llm_int8_matmul(X, W), exact)
    n_out = int((np.abs(X).max(axis=0) >= 6.0).sum())
    cx = 127.0 / np.abs(X).max(axis=1)
    codes = np.rint(X * cx[:, None])
    regular = np.setdiff1d(np.arange(h), outlier_dims)
    flushed = float((codes[:, regular] == 0).mean())
    print(f"{mag:>9.1f} {ev:>12.4f} {ed:>11.4f} {n_out:>10d} {flushed:>20.3f}")

X = make_hidden(60.0)
per_channel = np.abs(X).max(axis=0)
top = np.argsort(per_channel)[::-1][:5]
print("top channels by max |x|:", [(int(i), round(float(per_channel[i]), 1)) for i in top])
print("median channel max |x| :", round(float(np.median(per_channel)), 2))
print("share of dims in O     :", round(len(outlier_dims) / h * 100, 3), "%")
