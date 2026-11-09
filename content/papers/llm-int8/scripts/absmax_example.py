"""Absmax quantization of one tensor, with and without an outlier.

Run: python3 scripts/absmax_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def absmax_quantize(x):
    s = 127.0 / np.max(np.abs(x))  # the paper's s_x (Section 2.1)
    return np.rint(s * x).astype(np.int8), s


clean = np.array([0.30, -0.70, 0.45, 0.10, -0.20, 0.60])
dirty = np.append(clean, 60.0)

for name, x in (("without outlier", clean), ("with outlier 60.0", dirty)):
    q, s = absmax_quantize(x)
    back = q / s
    err = np.abs(back - x)
    print(f"--- {name}")
    print("scale s_x        =", round(float(s), 4))
    print("int8 codes       =", q.tolist())
    print("dequantized      =", back.tolist())
    print("max abs error    =", round(float(err[: len(clean)].max()), 4), "(over the six small values)")
    print("codes equal to 0 =", int((q[: len(clean)] == 0).sum()), "of", len(clean))
print("bin width 1/s    =", round(1 / (127 / 0.70), 4), "vs", round(1 / (127 / 60.0), 4))
