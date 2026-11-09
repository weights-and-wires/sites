"""Tensor-wise and vector-wise int8 matmul on a 2x3 by 3x2 example (Equations 6 and 7).

Run: python3 scripts/vectorwise_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

X = np.array([[0.2, -0.4, 0.1], [20.0, 30.0, -10.0]])
W = np.array([[0.5, -1.0], [1.0, 0.3], [-0.2, 0.8]])
exact = X @ W


def quantize_rows(A):
    c = 127.0 / np.max(np.abs(A), axis=1)
    return np.rint(A * c[:, None]).astype(np.int8), c


# Equation 6 with one constant per tensor.
cx = 127.0 / np.abs(X).max()
cw = 127.0 / np.abs(W).max()
Xq = np.rint(cx * X).astype(np.int8)
Wq = np.rint(cw * W).astype(np.int8)
tensor = (Xq.astype(np.int32) @ Wq.astype(np.int32)) / (cx * cw)

# Equation 7: c_x per row of X, c_w per column of W.
Xv, cxv = quantize_rows(X)
Wt, cwv = quantize_rows(W.T)
Wv = Wt.T
Ci32 = Xv.astype(np.int32) @ Wv.astype(np.int32)
vector = Ci32 / np.outer(cxv, cwv)

print("exact X W              =", exact.tolist())
print("c_x (rows of X)        =", cxv.tolist())
print("c_w (columns of W)     =", cwv.tolist())
print("X int8 rows            =", Xv.tolist())
print("W int8 columns (as W)  =", Wv.tolist())
print("C int32                =", Ci32.tolist())
print("vector-wise result     =", vector.tolist())
print("tensor-wise result     =", tensor.tolist())
print("vector-wise abs error  =", np.abs(vector - exact).round(4).tolist())
print("tensor-wise abs error  =", np.abs(tensor - exact).round(4).tolist())
