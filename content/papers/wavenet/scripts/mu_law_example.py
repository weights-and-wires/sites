"""Compute the mu-law companding examples.

Run: python3 scripts/mu_law_example.py
"""

import numpy as np

MU = 255


def compress(x, mu=MU):
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def expand(f, mu=MU):
    return np.sign(f) * np.expm1(np.abs(f) * np.log1p(mu)) / mu


def encode(x, mu=MU):
    f = compress(x, mu)
    return np.floor((f + 1) / 2 * mu + 0.5).astype(int)


def decode(q, mu=MU):
    return expand(2.0 * q / mu - 1, mu)


print("16-bit outputs:", 2**16, " mu-law outputs:", MU + 1)
print("table: x, f(x), q, decoded, abs error")
for x in [0.001, 0.01, 0.1, 0.5, 0.9]:
    f = compress(x)
    q = int(encode(np.array(x)))
    xr = float(decode(np.array(q)))
    print(f"x={x} f={f:.4f} q={q} decoded={xr:.5f} err={abs(x - xr):.5f}")

# Width of one quantization bin, in amplitude units, at two amplitudes.
step_f = 2 / MU
for x in [0.0, 1.0]:
    dx_df = np.log1p(MU) * (1 + MU * x) / MU
    print(f"x={x}: bin width in amplitude = {dx_df * step_f:.5f}")
print(f"linear 8-bit bin width = {2 / MU:.5f}")

# Worst round-trip error over a grid.
grid = np.linspace(-0.999, 0.999, 200001)
err = np.abs(decode(encode(grid)) - grid)
small = np.abs(grid) < 0.01
print(f"max error overall = {err.max():.5f}; for |x| < 0.01 = {err[small].max():.5f}")
lin = np.abs(np.round((grid + 1) / 2 * MU) / MU * 2 - 1 - grid)
print(f"linear 8-bit max error = {lin.max():.5f}; for |x| < 0.01 = {lin[small].max():.5f}")
print("encode(-1.0), encode(0.0), encode(1.0) =", encode(np.array([-1.0, 0.0, 1.0])).tolist())
