"""Show where naive softmax fails in float32 and where the online version does not.

Run: python3 scripts/overflow_example.py
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def naive(x):
    e = np.exp(x)  # Algorithm 1
    return e / e.sum()


def online(x):
    m, d = np.float32(-np.inf), np.float32(0.0)
    for v in x:
        m_new = max(m, v)
        d = d * np.exp(m - m_new) + np.exp(v - m_new)
        m = m_new
    return np.exp(x - m) / d


print("float32 limits: exp(88) =", np.exp(np.float32(88)), " exp(89) =", np.exp(np.float32(89)))
for name, vals in [("overflow", [100.0, 101.0, 102.0]), ("underflow", [-200.0, -201.0, -202.0])]:
    x = np.array(vals, dtype=np.float32)
    print(f"{name}: x = {vals}")
    print("  naive  :", naive(x))
    print("  online :", np.round(online(x), 4))
