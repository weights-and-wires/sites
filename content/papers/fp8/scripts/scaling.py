"""Overflow, underflow, and per-tensor scaling (Section 2, Section 4.2, Section 4.3)."""
import numpy as np
from fp8_common import FORMATS, quantize


def roundtrip(x, fmt, scale):
    return quantize(x * scale, fmt) / scale


def rel_rmse(q, x):
    return float(np.sqrt(np.mean((q - x) ** 2)) / np.sqrt(np.mean(x**2)))


rng = np.random.default_rng(0)
print("saturation:", float(quantize(1000.0, "e4m3")), float(quantize(1e6, "e5m2")))

# Activations with a few large outliers, like the ones that need per-tensor scales.
act = rng.normal(size=10_000)
act[:10] *= 1000
print(f"activations: amax {np.abs(act).max():.1f}")
for scale, label in ((1.0, "no scaling"), (448.0 / np.abs(act).max(), "amax scaling")):
    q = roundtrip(act, "e4m3", scale)
    clipped = int(np.sum(np.abs(act * scale) > 448))
    print(f"  E4M3 {label}: scale {scale:.4g}, values beyond 448 after scaling {clipped}, relative RMSE {rel_rmse(q, act):.4f}")

# Small-magnitude gradients, around 1e-5.
grad = rng.normal(size=10_000) * 1e-5
print(f"gradients: std 1e-5, amax {np.abs(grad).max():.2e}")
for fmt in ("e4m3", "e5m2"):
    for scale, label in ((1.0, "no scaling"), (FORMATS[fmt]["max"] / np.abs(grad).max(), "amax scaling")):
        q = roundtrip(grad, fmt, scale)
        print(f"  {fmt} {label}: zeros {np.mean(q == 0):.1%}, relative RMSE {rel_rmse(q, grad):.4f}")

# Which format for which tensor: a Gaussian tensor versus a wide-range gradient tensor.
wide = rng.normal(size=10_000) * 10.0 ** rng.uniform(-6, 0, size=10_000)
tensors = {"Gaussian (weights-like)": rng.normal(size=10_000), "wide-range (gradient-like)": wide}
for name, t in tensors.items():
    print(name, f"amax {np.abs(t).max():.3g}, log10 spread of |t| (1st to 99th percentile) "
          f"{np.log10(np.percentile(np.abs(t), 99) / np.percentile(np.abs(t), 1)):.1f}")
    for fmt in ("e4m3", "e5m2"):
        s = FORMATS[fmt]["max"] / np.abs(t).max()
        q = roundtrip(t, fmt, s)
        elem = np.abs(q - t) / np.abs(t)
        print(f"  {fmt}: zeros {np.mean(q == 0):.1%}, relative RMSE {rel_rmse(q, t):.4f}, "
              f"median per-element error {np.median(elem):.4f}")
