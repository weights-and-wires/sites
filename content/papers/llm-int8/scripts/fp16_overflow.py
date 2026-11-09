"""How soon a sum of squared outliers overflows FP16, and how int8 bins shrink.

Run: python3 scripts/fp16_overflow.py
"""

import numpy as np

fp16_max = float(np.finfo(np.float16).max)
print("FP16 largest finite value =", fp16_max)
term = 60.0 * 60.0
n = int(np.ceil(fp16_max / term))
print("terms of 60*60 =", term, "that overflow FP16 at n =", n, "; partial sum at n-1 =", term * (n - 1), "; at n =", term * n)
np.seterr(over="ignore")
acc = np.float16(0)
for k in range(1, 40):
    acc = np.float16(acc + np.float16(term))
    if np.isinf(acc):
        print("FP16 running sum reaches inf at term", k)
        break

for m in (1.0, 6.0, 60.0):
    print(f"absmax {m:>5}: one int8 step = {m / 127:.4f}, values below {m / 254:.4f} round to 0")
