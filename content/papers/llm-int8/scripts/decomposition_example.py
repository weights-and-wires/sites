"""Mixed-precision decomposition on a small matrix with an outlier feature (Equation 8).

Run: python3 scripts/decomposition_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def vectorwise_matmul(X, W):
    cx = 127.0 / np.max(np.abs(X), axis=1)
    cw = 127.0 / np.max(np.abs(W), axis=0)
    Xq = np.rint(X * cx[:, None]).astype(np.int8)
    Wq = np.rint(W * cw[None, :]).astype(np.int8)
    return (Xq.astype(np.int32) @ Wq.astype(np.int32)) / np.outer(cx, cw)


def llm_int8_matmul(X, W, alpha=6.0):
    O = np.where(np.max(np.abs(X), axis=0) >= alpha)[0]
    regular = np.setdiff1d(np.arange(X.shape[1]), O)
    return X[:, O] @ W[O, :] + vectorwise_matmul(X[:, regular], W[regular, :]), O


X = np.array([[0.30, 40.0, -0.50, 0.20],
              [-0.40, 55.0, 0.10, 0.60],
              [0.25, -48.0, 0.35, -0.15]])
W = np.array([[0.5, -0.2], [0.3, 0.4], [-0.6, 0.1], [0.2, 0.7]])
exact = X @ W
plain = vectorwise_matmul(X, W)
mixed, O = llm_int8_matmul(X, W)

print("outlier dimensions O   =", O.tolist())
print("exact X W              =", exact.tolist())
print("vector-wise only       =", plain.tolist())
print("decomposed (Eq. 8)     =", mixed.tolist())
print("vector-wise max error  =", round(float(np.abs(plain - exact).max()), 4))
print("decomposed max error   =", round(float(np.abs(mixed - exact).max()), 4))
print("outlier columns of X   =", X[:, O].ravel().tolist())
