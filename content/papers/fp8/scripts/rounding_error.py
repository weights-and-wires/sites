"""Relative rounding error of E4M3 and E5M2 on Gaussian data (Section 3)."""
import numpy as np
from fp8_common import quantize

print("tie cases (round to nearest even):")
for x in (1.0625, 1.1875, 1.3125):
    print(f"  E4M3({x}) = {float(quantize(x, 'e4m3'))}")
print("  E5M2(1.125) =", float(quantize(1.125, "e5m2")), " E5M2(1.375) =", float(quantize(1.375, "e5m2")))

rng = np.random.default_rng(0)
x = rng.normal(size=100_000)
for fmt in ("e4m3", "e5m2"):
    q = quantize(x, fmt)
    nz = np.abs(x) > 0.1
    rel = np.abs(q - x)[nz] / np.abs(x)[nz]
    print(f"{fmt}: mean rel err {rel.mean():.4f}, max rel err {rel.max():.4f}, "
          f"worst possible 2^-(m+1) = {2.0 ** -(4 if fmt == 'e4m3' else 3):.4f}")
