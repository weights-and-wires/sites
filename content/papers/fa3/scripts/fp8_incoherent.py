"""Block quantization and incoherent processing on a toy (Section 3.3).

Rounds to E4M3 with NumPy, scales per tensor or per block, optionally rotates
with a random-sign Hadamard transform first, and measures the error of the
score matrix S = Q K^T.

Run: python3 scripts/fp8_incoherent.py
"""

import numpy as np

E4M3_MAX = 448.0


def round_e4m3(x):
    """Round to the nearest E4M3 value (ties to even) and saturate at 448."""
    x = np.asarray(x, dtype=float)
    a = np.abs(x)
    e = np.maximum(np.floor(np.log2(np.where(a > 0, a, 1.0))), -6)
    step = 2.0 ** (e - 3)
    return np.sign(x) * np.minimum(np.round(a / step) * step, E4M3_MAX)


def quantize(x, block_rows=None):
    """Quantize and dequantize with one scale per tensor or per block of rows."""
    if block_rows is None:
        block_rows = x.shape[0]
    out = np.empty_like(x)
    for i in range(0, x.shape[0], block_rows):
        blk = x[i:i + block_rows]
        scale = np.abs(blk).max() / E4M3_MAX
        out[i:i + block_rows] = round_e4m3(blk / scale) * scale
    return out


def fwht(x):
    """Orthonormal fast Walsh-Hadamard transform over the last axis."""
    d = x.shape[-1]
    x = np.array(x, dtype=float)
    h = 1
    while h < d:
        x = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        x = np.stack([x[..., 0, :] + x[..., 1, :], x[..., 0, :] - x[..., 1, :]], axis=-2)
        x = x.reshape(*x.shape[:-3], d)
        h *= 2
    return x / np.sqrt(d)


def rel_err(S, Sq):
    return np.sqrt(np.mean((S - Sq) ** 2)) / np.sqrt(np.mean(S ** 2))


def measure(Q, K, signs, block_rows):
    S = Q @ K.T
    rot = lambda x: fwht(x * signs)
    return {
        "per-tensor": rel_err(S, quantize(Q) @ quantize(K).T),
        "per-block": rel_err(S, quantize(Q, block_rows) @ quantize(K, block_rows).T),
        "incoherent only": rel_err(S, quantize(rot(Q)) @ quantize(rot(K)).T),
        "both": rel_err(S, quantize(rot(Q), block_rows) @ quantize(rot(K), block_rows).T),
    }


if __name__ == "__main__":
    print("E4M3 grid facts")
    for x in (1.0, 1.0625, 1.1, 460.0, 2.0 ** -9, 2.0 ** -10, 0.3):
        print(f"  round_e4m3({x:<12.8g}) = {float(round_e4m3(x)):.8g}")
    row = np.array([[1.0, 2.0, 3.0, 4.0]])
    print(f"  block [1, 2, 3, 4]: scale = 4 / 448 = {4 / 448:.6f}; 3 / scale = {3 / (4 / 448):.0f}")
    print(f"  dequantized block = {np.round(quantize(row), 4)}")
    print()
    print("Hadamard check: rotation keeps Q K^T and the norm (d = 64)")
    rng = np.random.default_rng(1)
    d = 64
    signs = rng.choice([-1.0, 1.0], size=d)
    q, k = rng.normal(size=(2, 5, d))
    rot = lambda x: fwht(x * signs)
    print(f"  max |(QM)(KM)^T - QK^T| = {np.abs(rot(q) @ rot(k).T - q @ k.T).max():.2e}")
    print(f"  norm before {np.linalg.norm(q[0]):.4f}, after {np.linalg.norm(rot(q)[0]):.4f}")
    print()

    N, d, br = 512, 64, 64
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=d)

    def paper_like():
        f = lambda: rng.normal(size=(N, d)) + 10 * rng.normal(size=(N, d)) * (rng.random((N, d)) < 1e-3)
        return f(), f()

    def channel_outliers(mag):
        Q, K = rng.normal(size=(N, d)), rng.normal(size=(N, d))
        Q[:, :2] *= mag
        K[:, :2] *= mag
        return Q, K

    cases = (("paper-like iid outliers (0.1% of entries, sd 10)", paper_like()),
             ("2 outlier channels, 30x", channel_outliers(30)),
             ("2 outlier channels, 100x", channel_outliers(100)))
    print(f"Relative RMSE of S = Q K^T, N={N}, d={d}, blocks of {br} rows")
    for name, (Q, K) in cases:
        r = measure(Q, K, signs, br)
        print(f"  {name}")
        for key, val in r.items():
            print(f"    {key:<16} {val:.3e}")
        print(f"    per-tensor / both = {r['per-tensor'] / r['both']:.2f}x")
    print()
    print("Paper Table 3 ratios (RMSE of attention output vs FP64)")
    print(f"  baseline FP8 / FlashAttention-3 FP8 = 2.4e-2 / 9.1e-3 = {2.4e-2 / 9.1e-3:.2f}")
    print(f"  baseline FP16 / FlashAttention-3 FP16 = 3.2e-4 / 1.9e-4 = {3.2e-4 / 1.9e-4:.2f}")
