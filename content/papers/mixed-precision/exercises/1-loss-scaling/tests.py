import warnings

import numpy as np
from submission import choose_scale, scale_roundtrip

warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_small_gradient_underflows_without_scaling():
    g = np.array([1e-8], dtype=np.float32)
    got = scale_roundtrip(g, 1.0)
    assert got[0] == 0.0, f"1e-8 is below 2^-25 and must become 0 in FP16, got {got[0]}"


def test_scaling_preserves_small_gradient():
    g = np.array([1e-8], dtype=np.float32)
    got = scale_roundtrip(g, 1024.0)
    assert got[0] != 0.0, f"scale 1024 must keep 1e-8 alive, got {got[0]}"
    assert np.isclose(got[0], 1e-8, rtol=0.05), f"expected about 1e-8, got {got[0]}"


def test_output_is_unscaled_float32():
    g = np.array([0.5, 0.25], dtype=np.float32)
    got = scale_roundtrip(g, 256.0)
    assert got.dtype == np.float32, f"expected float32, got {got.dtype}"
    assert np.allclose(got, g), f"expected {g}, got {got}"


def test_unscaling_happens_in_float32():
    g = np.array([3e-5], dtype=np.float32)
    got = scale_roundtrip(g, 65536.0 / 8)
    want = np.float32(np.float16(np.float32(3e-5) * np.float32(8192.0))) / np.float32(8192.0)
    assert np.allclose(got, want, rtol=1e-6), f"expected {want}, got {got}"


def test_overflow_gives_inf():
    g = np.array([2.0], dtype=np.float32)
    got = scale_roundtrip(g, 65536.0)
    assert np.isinf(got[0]), f"2.0 * 65536 exceeds 65504 and must overflow to inf, got {got[0]}"


def test_fraction_zeroed_matches_the_paper_rule():
    exps = np.linspace(-30, -10, 2001)
    g = (2.0**exps).astype(np.float32)
    z1 = float(np.mean(scale_roundtrip(g, 1.0) == 0))
    z128 = float(np.mean(scale_roundtrip(g, 128.0) == 0))
    assert 0.2 < z1 < 0.3, f"about 25% of the gradients should vanish at scale 1, got {z1:.3f}"
    assert z128 == 0.0, f"scale 128 should keep every gradient, got {z128:.3f} zeros"


def test_choose_scale_values():
    assert choose_scale(2.0) == 16384.0, f"expected 16384.0, got {choose_scale(2.0)}"
    assert choose_scale(1.0) == 32768.0, f"expected 32768.0, got {choose_scale(1.0)}"
    assert choose_scale(1e-3) == 33554432.0, f"expected 33554432.0, got {choose_scale(1e-3)}"


def test_choose_scale_never_overflows_and_is_maximal():
    for m in [0.3, 1.0, 7.0, 123.4, 1e-4]:
        s = choose_scale(m)
        assert s * m <= 65504.0, f"scale {s} overflows for max {m}"
        assert 2 * s * m > 65504.0, f"scale {s} is not the largest power of two for max {m}"
        assert np.log2(s) == round(np.log2(s)), f"scale {s} is not a power of two"
