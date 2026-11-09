import numpy as np
from submission import quantize

SPEC = {"e4m3": (4, 3, 7), "e5m2": (5, 2, 15)}


def grid(fmt):
    """Every finite non-negative FP8 value, built from the bit layout."""
    eb, mb, bias = SPEC[fmt]
    vals = []
    for e in range(1 << eb):
        for m in range(1 << mb):
            if fmt == "e4m3" and e == 15 and m == 7:
                continue
            if fmt == "e5m2" and e == 31:
                continue
            vals.append(m / 2**mb * 2.0 ** (1 - bias) if e == 0 else (1 + m / 2**mb) * 2.0 ** (e - bias))
    return np.array(sorted(vals))


def test_returns_float_array_of_same_shape():
    x = np.ones((3, 4)) * 1.3
    q = quantize(x, "e4m3")
    assert q.shape == (3, 4), f"expected shape (3, 4), got {q.shape}"
    assert np.issubdtype(q.dtype, np.floating), f"expected a float array, got {q.dtype}"


def test_representable_values_are_unchanged():
    for fmt in ("e4m3", "e5m2"):
        g = grid(fmt)
        got = quantize(g, fmt)
        assert np.array_equal(got, g), f"{fmt}: representable values must not move; first diff at {g[got != g][:3]}"


def test_ties_go_to_even():
    # Between 1.0 (M=0) and 1.125 (M=1): the tie goes to 1.0. Between 1.125 (M=1) and 1.25 (M=2): to 1.25.
    got = quantize(np.array([1.0625, 1.1875, 1.3125]), "e4m3")
    assert np.array_equal(got, [1.0, 1.25, 1.25]), f"E4M3 ties: expected [1.0, 1.25, 1.25], got {got}"
    got = quantize(np.array([1.125, 1.375]), "e5m2")
    assert np.array_equal(got, [1.0, 1.5]), f"E5M2 ties: expected [1.0, 1.5], got {got}"


def test_rounds_to_nearest_not_down():
    got = quantize(np.array([1.07, 1.12, 1.9, 3.9]), "e4m3")
    assert np.array_equal(got, [1.125, 1.125, 1.875, 4.0]), f"expected [1.125, 1.125, 1.875, 4.0], got {got}"


def test_saturates_instead_of_overflowing():
    got = quantize(np.array([449.0, 470.0, 1000.0, -1e6]), "e4m3")
    assert np.array_equal(got, [448.0, 448.0, 448.0, -448.0]), f"E4M3 expected +-448, got {got}"
    got = quantize(np.array([60000.0, 1e9]), "e5m2")
    assert np.array_equal(got, [57344.0, 57344.0]), f"E5M2 expected 57344, got {got}"


def test_subnormals_and_underflow():
    # E4M3 subnormal spacing is 2^-9 = 0.001953125.
    got = quantize(np.array([0.0009, 0.001, 0.003, 0.0148]), "e4m3")
    want = [0.0, 2.0**-9, 2.0 * 2.0**-9, 0.015625]
    assert np.array_equal(got, want), f"E4M3 near zero: expected {want}, got {got}"
    got = quantize(np.array([1e-5, 2e-5]), "e5m2")
    want = [2.0**-16, 2.0**-16]
    assert np.array_equal(got, want), f"E5M2 near zero: expected {want}, got {got}"
    assert quantize(np.array([1e-9]), "e5m2")[0] == 0.0, "values below half the smallest subnormal must become 0"


def test_negative_values_mirror_positive():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500) * 3
    for fmt in ("e4m3", "e5m2"):
        assert np.array_equal(quantize(-x, fmt), -quantize(x, fmt)), f"{fmt}: quantize(-x) must equal -quantize(x)"


def test_matches_brute_force_nearest():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000) * 10.0 ** rng.uniform(-3, 2, size=2000)
    for fmt in ("e4m3", "e5m2"):
        g = grid(fmt)
        full = np.concatenate([-g[::-1], g])
        want = full[np.abs(x[:, None] - full[None, :]).argmin(axis=1)]
        got = quantize(x, fmt)
        bad = np.flatnonzero(got != want)
        assert bad.size == 0, f"{fmt}: {bad.size} values differ from the nearest grid point; x={x[bad[:2]]}, expected {want[bad[:2]]}, got {got[bad[:2]]}"


def test_nan_passes_through():
    got = quantize(np.array([np.nan, 1.0]), "e4m3")
    assert np.isnan(got[0]) and got[1] == 1.0, f"expected [nan, 1.0], got {got}"
