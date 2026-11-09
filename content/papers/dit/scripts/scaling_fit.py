"""Fit FID against Gflops for the 12 DiT models, using Table 4 of the paper.

Run: python3 scripts/scaling_fit.py
"""

import numpy as np

# (Gflops, FID-50K at 400K steps, no guidance), Table 4.
ROWS = {
    "S/8": (0.36, 153.60), "S/4": (1.41, 100.41), "S/2": (6.06, 68.40),
    "B/8": (1.42, 122.74), "B/4": (5.56, 68.38), "B/2": (23.01, 43.47),
    "L/8": (5.01, 118.87), "L/4": (19.70, 45.64), "L/2": (80.71, 23.33),
    "XL/8": (7.39, 106.41), "XL/4": (29.05, 43.01), "XL/2": (118.64, 19.47),
}
g = np.array([v[0] for v in ROWS.values()])
fid = np.array([v[1] for v in ROWS.values()])
print("corr(Gflops, FID)       =", round(float(np.corrcoef(g, fid)[0, 1]), 3))
print("corr(log Gflops, FID)   =", round(float(np.corrcoef(np.log(g), fid)[0, 1]), 3))
slope, intercept = np.polyfit(np.log10(g), fid, 1)
print(f"fit: FID = {slope:.1f} * log10(Gflops) + {intercept:.1f}")

print()
print("Similar Gflops, different shape (the paper's example, Section 5):")
print("  S/2 :", ROWS["S/2"], " B/4 :", ROWS["B/4"])
print("Same width, patch size halved: Gflops x4, FID drops")
for m in ("S", "B", "L", "XL"):
    a, b = ROWS[f"{m}/4"], ROWS[f"{m}/2"]
    print(f"  {m}/4 -> {m}/2: Gflops {a[0]} -> {b[0]} (x{b[0] / a[0]:.2f}), FID {a[1]} -> {b[1]}")

print()
print("Training compute estimate = Gflops * batch * steps * 3 (Section 5):")
xl2 = 118.64 * 256 * 7_000_000 * 3
print(f"  XL/2, 7M steps, batch 256: {xl2:.3e} Gflops")
print("Sampling compute (Section 5.2): L/2, 1000 steps vs XL/2, 128 steps")
print(f"  L/2: {80.71 * 1000 / 1000:.1f} Tflops (paper: 80.7); XL/2: {118.64 * 128 / 1000:.1f} Tflops (paper: 15.2)")
