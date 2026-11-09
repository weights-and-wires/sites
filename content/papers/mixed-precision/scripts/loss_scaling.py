"""Loss scaling keeps small gradients from underflowing (Section 3.2)."""
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # the overflow to inf is the point

rng = np.random.default_rng(0)
# Gradient magnitudes spread over 2^-30 .. 2^-10, as in the shape of Figure 3.
exps = rng.uniform(-30, -10, size=100_000)
g = (2.0**exps).astype(np.float32)

def roundtrip(g, scale):
    return (g * np.float32(scale)).astype(np.float16).astype(np.float32) / np.float32(scale)

for scale in [1, 8, 128, 1024, 32768]:
    r = roundtrip(g, scale)
    zero = float(np.mean(r == 0)) * 100
    print(f"scale {scale:6d}: {zero:5.2f}% of gradients are zero")

# One gradient by hand.
x = np.float32(1e-8)
print("1e-8 stored in FP16:", float(np.float16(x)))
print("1e-8 * 1024 stored in FP16:", float(np.float16(x * 1024)))
print("unscaled in FP32:", float(np.float32(np.float16(x * 1024)) / 1024))

# Overflow side: choose a scale so scale * max|g| <= 65504.
mx = 2.0
print("max|g| = 2: scale 2^15 gives", float(np.float16(mx * 2**15)), "; scale 2^16 gives", float(np.float16(mx * 2**16)))
best = 2 ** int(np.floor(np.log2(65504 / mx)))
print("largest power-of-two scale for max|g| = 2:", best)
