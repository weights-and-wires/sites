"""Show that additive sinusoidal encodings leak absolute position into the score.

Both methods use the same x_m, x_n (identical content at every position) and
W_q = W_k = I, to isolate the position effect.

Run: python3 scripts/additive_vs_rotary.py
"""

import numpy as np

d = 8
rng = np.random.default_rng(1)
x = rng.normal(size=d)  # the same token content at every position


def sin_pe(pos):
    out = np.zeros(d)
    for i in range(d // 2):
        a = pos / 10000 ** (2 * i / d)
        out[2 * i], out[2 * i + 1] = np.sin(a), np.cos(a)
    return out


def rope(v, m):
    th = 10000.0 ** (-2.0 * np.arange(d // 2) / d)
    out = np.empty_like(v)
    c, s = np.cos(m * th), np.sin(m * th)
    out[0::2], out[1::2] = v[0::2] * c - v[1::2] * s, v[0::2] * s + v[1::2] * c
    return out


print("offset m-n = 2; same token content at every position")
print(" m   n   additive (x+p_m).(x+p_n)   rotary")
for m in [2, 10, 50, 200]:
    n = m - 2
    add = (x + sin_pe(m)) @ (x + sin_pe(n))
    rot = rope(x, m) @ rope(x, n)
    print(f"{m:3d} {n:3d}   {add:22.4f}   {rot:8.4f}")
