"""Decode every FP8 bit pattern and check Table 1 (Section 3, Section 3.1)."""
import numpy as np
from fp8_common import decode, grid

for fmt in ("e4m3", "e5m2"):
    g = grid(fmt)
    pos = g[g > 0]
    subn = pos[pos < 2.0 ** (-6 if fmt == "e4m3" else -14)]
    print(f"{fmt}: finite non-negative values {len(g)} (incl. zero), max {g[-1]:g}, "
          f"min subnormal {pos[0]:.6g}, max subnormal {subn[-1]:.6g}, min normal {pos[len(subn)]:.6g}")
    nans = sum(np.isnan(decode(c, fmt)) for c in range(256))
    infs = sum(np.isinf(decode(c, fmt)) for c in range(256))
    print(f"{fmt}: NaN codes {nans}, infinity codes {infs}, "
          f"binades {int(np.log2(g[-1] / pos[0])) + 1}")

print("E4M3 top binade (exponent field 1111):", [decode(0b01111000 | m, "e4m3") for m in range(8)])
print("E4M3 max without the reclaim (exponent field 1111 reserved):", decode(0b01110111, "e4m3"))
print("E5M2 max:", decode(0b01111011, "e5m2"))
print("distinct values E4M3 in [1, 2):", [decode(0b00111000 | m, "e4m3") for m in range(8)])
print("distinct values E5M2 in [1, 2):", [decode(0b00111100 | m, "e5m2") for m in range(4)])
