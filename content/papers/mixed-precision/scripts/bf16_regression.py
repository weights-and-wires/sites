"""bfloat16 by mantissa truncation. bfloat16 is not in the paper; it is a later format."""
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # the overflow to inf is the point

def to_bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) & np.uint32(0xFFFF0000)).view(np.float32)

for v in [1 + 2.0**-8, 1 + 3 * 2.0**-8, 1 + 2.0**-7, 3.14159265, 1e-30, 1e38]:
    print(f"bf16({v!r}) = {float(to_bf16(np.float32(v)))!r}   fp16 = {float(np.float16(v))!r}")

rng = np.random.default_rng(0)
X = rng.normal(size=(256, 4)).astype(np.float32)
w_true = np.array([1.5, -2.0, 0.5, 3.0], dtype=np.float32)
y = X @ w_true

def train(quant, master=True, lr=0.05, steps=200):
    w = np.zeros(4, np.float32)
    for _ in range(steps):
        wq = quant(w) if master else w
        pred = quant(X @ wq)
        grad = quant(X.T @ (pred - y) / len(X))
        w = quant(w - lr * grad) if not master else w - lr * grad
    return w, float(np.mean((X @ w - y) ** 2))

ident = lambda a: np.asarray(a, np.float32)
for name, q, m in [("fp32", ident, True), ("bf16 + fp32 master", to_bf16, True), ("bf16 weights only", to_bf16, False)]:
    w, loss = train(q, m)
    print(f"{name:20s} loss {loss:.3e}  w {np.round(w, 3)}")
