# Per-row int8 quantization: W ~= q * s[row], with q in [-127, 127].
import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(0, 0.05, (64, 64)).astype(np.float32)
W[7] *= 12  # one row with much larger weights


def quantize_rows(W):
    s = np.abs(W).max(axis=1) / 127.0
    q = np.clip(np.round(W / s[:, None]), -127, 127).astype(np.int8)
    return q, s.astype(np.float32)


def dequantize_rows(q, s):
    return q.astype(np.float32) * s[:, None]  # same shape of rule as lin_w in runtime.py


def quantize_tensor(W):
    s = np.abs(W).max() / 127.0
    return np.clip(np.round(W / s), -127, 127).astype(np.int8), np.float32(s)


q, s = quantize_rows(W)
Wr = dequantize_rows(q, s)
qt, st = quantize_tensor(W)
Wt = qt.astype(np.float32) * st

print("worked example: ", end="")
w = np.array([[0.9, -0.3, 0.05]], np.float32)
qw, sw = quantize_rows(w)
print("s =", round(float(sw[0]), 5), "q =", qw[0].tolist(), "back =", [round(float(v), 4) for v in dequantize_rows(qw, sw)[0]])

err_r, err_t = np.abs(W - Wr), np.abs(W - Wt)
print("max |error| per row <= s/2 :", bool((err_r.max(axis=1) <= s / 2 + 1e-9).all()))
print("mean |error|, per-row   :", round(float(err_r.mean()), 6))
print("mean |error|, per-tensor:", round(float(err_t.mean()), 6))
quiet = np.arange(64) != 7
print("mean |error| on the 63 quiet rows, per-row   :", round(float(err_r[quiet].mean()), 6))
print("mean |error| on the 63 quiet rows, per-tensor:", round(float(err_t[quiet].mean()), 6))
print("mean s/4 over quiet rows (rule of thumb):", round(float((s[quiet] / 4).mean()), 6))

x = rng.normal(0, 1, 64).astype(np.float32)
y = W @ x
for name, M in (("per-row", Wr), ("per-tensor", Wt)):
    print(f"matmul relative error, {name}:", round(float(np.linalg.norm(M @ x - y) / np.linalg.norm(y)), 5))

fp32_bytes = W.size * 4
int8_bytes = q.size + s.size * 4
print(f"bytes: fp32 {fp32_bytes}, int8 + fp32 row scales {int8_bytes}, ratio {fp32_bytes / int8_bytes:.2f}x")
result = {"row_err": err_r.mean(axis=1).tolist(), "tensor_err": err_t.mean(axis=1).tolist()}
