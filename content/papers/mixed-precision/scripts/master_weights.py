"""Why an FP32 master copy matters (Section 3.1)."""
import numpy as np

# Case 1: the weight-to-update ratio exceeds 2048.
w16 = np.float16(1.0)
print("1 + 2^-10 in FP16:", float(np.float16(1.0) + np.float16(2.0**-10)))
print("1 + 2^-11 in FP16:", float(np.float16(1.0) + np.float16(2.0**-11)), "(tie rounds to even)")
print("1 + 2^-12 in FP16:", float(np.float16(1.0) + np.float16(2.0**-12)))

# 1000 steps of w <- w - lr * g with lr * g = 2e-4 and w = 1.0.
lr, g, steps = np.float32(1e-3), np.float32(0.2), 1000
w_fp16 = np.float16(1.0)
master = np.float32(1.0)
for _ in range(steps):
    upd = np.float16(lr * g)
    w_fp16 = np.float16(w_fp16 - upd)
    master = np.float32(master - lr * g)
print(f"update lr*g = {float(lr*g):.1e}, representable in FP16 as {float(np.float16(lr*g)):.6e}")
print("FP16-only weight after 1000 steps:", float(w_fp16))
print("FP32 master after 1000 steps     :", round(float(master), 6))
print("FP16 copy of master              :", float(np.float16(master)))

# Case 2: the update itself underflows.
lr, g = np.float32(1e-4), np.float32(1e-4)
print(f"update lr*g = {float(lr*g):.1e}; in FP16 it becomes {float(np.float16(lr*g))}")
