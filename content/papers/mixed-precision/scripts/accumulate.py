"""Accumulate FP16 products in FP32 (Section 3.3)."""
import numpy as np

def dot(a, b, acc):
    s = acc(0)
    for x, y in zip(a, b):
        s = acc(s + acc(x) * acc(y))
    return s

n = 4096
a = np.ones(n, dtype=np.float16)
print("dot of 4096 ones, FP16 accumulator:", float(dot(a, a, np.float16)))
print("dot of 4096 ones, FP32 accumulator:", float(dot(a, a, np.float32)))

rng = np.random.default_rng(0)
a = rng.uniform(0, 1, 4096).astype(np.float16)
b = rng.uniform(0, 1, 4096).astype(np.float16)
exact = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
f16 = float(dot(a, b, np.float16))
f32 = float(dot(a, b, np.float32))
print(f"random dot, exact {exact:.2f}, FP16 acc {f16:.2f}, FP32 acc {f32:.2f}")
print(f"relative error: FP16 acc {abs(f16-exact)/exact:.4f}, FP32 acc {abs(f32-exact)/exact:.2e}")
