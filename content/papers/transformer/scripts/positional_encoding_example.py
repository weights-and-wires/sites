"""Compute the numbers for the positional encoding example.

Run: python3 scripts/positional_encoding_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def pe(pos, d_model):
    out = np.zeros(d_model)
    for i in range(d_model // 2):
        angle = pos / 10000 ** (2 * i / d_model)
        out[2 * i] = np.sin(angle)
        out[2 * i + 1] = np.cos(angle)
    return out


d = 4
for pos in range(4):
    print(f"PE({pos}) =", pe(pos, d).tolist())

print()
print("wavelengths for d_model = 4:", [2 * np.pi * 10000 ** (2 * i / d) for i in range(d // 2)])

# Section 3.5: PE(pos + k) is a linear function of PE(pos). For the first pair the
# angle is pos * 1, so a rotation by k = 1 maps PE(1)[:2] to PE(2)[:2].
k = 1.0
R = np.array([[np.cos(k), np.sin(k)], [-np.sin(k), np.cos(k)]])
print()
print("rotation by k=1 applied to PE(1)[:2] =", (R @ pe(1, d)[:2]).tolist())
print("PE(2)[:2]                           =", pe(2, d)[:2].tolist())
