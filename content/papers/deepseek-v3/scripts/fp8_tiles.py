"""Fine-grained FP8 scaling (Section 3.3.2, Figure 7a) on a made-up matrix.

Compares one scale per tensor, 1x128 tiles (the paper's activation grouping),
and 128x128 blocks (the paper's weight grouping). E4M3 only, as in the paper.
The data is synthetic, not the paper's.
"""
import numpy as np

E4M3_MAX, MBITS, BIAS = 448.0, 3, 7


def quantize(x):
    """Round to the nearest E4M3 value and saturate (see the FP8 page)."""
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    _, p = np.frexp(np.where(a > 0, a, 1.0))
    e = np.maximum(p - 1, 1 - BIAS)
    spacing = 2.0 ** (e - MBITS)
    return np.sign(x) * np.minimum(np.round(a / spacing) * spacing, E4M3_MAX)


def per_tensor(x):
    s = E4M3_MAX / np.abs(x).max()
    return quantize(x * s) / s


def tiles(x, n_c=128):
    """One scale per row per n_c columns (1 x n_c tiles)."""
    r, c = x.shape
    g = x.reshape(r, c // n_c, n_c)
    s = E4M3_MAX / np.abs(g).max(axis=2, keepdims=True)
    return (quantize(g * s) / s).reshape(r, c)


def blocks(x, n_c=128):
    """One scale per n_c x n_c block."""
    r, c = x.shape
    g = x.reshape(r // n_c, n_c, c // n_c, n_c)
    s = E4M3_MAX / np.abs(g).max(axis=(1, 3), keepdims=True)
    return (quantize(g * s) / s).reshape(r, c)


def rel_rmse(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(x**2)))


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (256, 512))
    x[3, 7] = 100000.0  # one outlier in one token
    print("matrix 256 x 512, Gaussian, one outlier 100000 at (3, 7); amax =", np.abs(x).max())
    for name, fn in (("per tensor", per_tensor), ("1x128 tiles", tiles), ("128x128 blocks", blocks)):
        y = fn(x)
        print(f"  {name:15s} rel RMSE, all = {rel_rmse(x, y):.4f}   "
              f"rel RMSE, rows 100-255 = {rel_rmse(x[100:], y[100:]):.4f}   "
              f"zeros in rows 100-255 = {(y[100:] == 0).mean() * 100:.2f}%")
    y = tiles(x)
    print("  tile holding the outlier: row 3, columns 0-127")
    print("    rel RMSE in that tile =", round(rel_rmse(x[3, :128], y[3, :128]), 4))
    print("    rel RMSE in row 3, columns 128-255 =", round(rel_rmse(x[3, 128:256], y[3, 128:256]), 4))
    # scale ratio between the outlier tile and a normal tile
    g = x.reshape(256, 4, 128)
    amax = np.abs(g).max(axis=2)
    print("  amax of the outlier tile = %.1f, of its neighbour tile = %.3f" % (amax[3, 0], amax[3, 1]))


if __name__ == "__main__":
    main()
