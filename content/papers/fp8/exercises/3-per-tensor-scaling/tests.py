import numpy as np
from submission import amax_scale, scaled_roundtrip, quantize


def rel_rmse(q, x):
    return float(np.sqrt(np.mean((q - x) ** 2)) / np.sqrt(np.mean(x**2)))


def test_scale_value():
    x = np.array([0.5, -2.0, 1.0])
    assert np.isclose(amax_scale(x, "e4m3"), 224.0), f"E4M3: expected 448 / 2 = 224, got {amax_scale(x, 'e4m3')}"
    assert np.isclose(amax_scale(x, "e5m2"), 28672.0), f"E5M2: expected 57344 / 2 = 28672, got {amax_scale(x, 'e5m2')}"


def test_scale_maps_amax_to_format_max():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000) * 7.3
    for fmt, top in (("e4m3", 448.0), ("e5m2", 57344.0)):
        got = np.max(np.abs(x * amax_scale(x, fmt)))
        assert np.isclose(got, top), f"{fmt}: scaled amax should be {top}, got {got}"


def test_scale_uses_magnitude_not_signed_max():
    x = np.array([1.0, -4.0, 2.0])
    assert np.isclose(amax_scale(x, "e4m3"), 112.0), f"expected 448 / 4 = 112, got {amax_scale(x, 'e4m3')}"


def test_zero_tensor_is_safe():
    z = np.zeros(8)
    assert amax_scale(z, "e4m3") == 1.0, f"all-zero tensor should give scale 1.0, got {amax_scale(z, 'e4m3')}"
    out = scaled_roundtrip(z, "e4m3")
    assert np.all(out == 0), f"expected zeros, got {out}"


def test_roundtrip_returns_original_units():
    x = np.array([0.5, -2.0, 1.0])
    got = scaled_roundtrip(x, "e4m3")
    assert np.allclose(got, x), f"powers of two survive exactly; expected {x}, got {got}"
    x = np.array([300.0, -250.0, 100.0])
    got = scaled_roundtrip(x, "e4m3")
    assert np.max(np.abs(got - x) / np.abs(x)) < 0.07, f"values must come back near {x}, got {got}"


def test_large_outliers_do_not_overflow():
    rng = np.random.default_rng(0)
    x = rng.normal(size=10_000)
    x[:10] *= 1000
    err_scaled = rel_rmse(scaled_roundtrip(x, "e4m3"), x)
    err_plain = rel_rmse(quantize(x, "e4m3"), x)
    assert err_scaled < 0.03, f"expected relative RMSE about 0.02 with scaling, got {err_scaled:.4f}"
    assert err_plain > 0.5, f"sanity check: plain casting saturates the outliers, expected error above 0.5, got {err_plain:.4f}"


def test_tiny_gradients_do_not_flush_to_zero():
    rng = np.random.default_rng(0)
    g = rng.normal(size=10_000) * 1e-5
    out = scaled_roundtrip(g, "e4m3")
    zeros = float(np.mean(out == 0))
    assert zeros < 0.01, f"scaling should rescue values near 1e-5; {zeros:.1%} became 0"
    assert rel_rmse(out, g) < 0.04, f"expected relative RMSE about 0.026, got {rel_rmse(out, g):.4f}"
    assert float(np.mean(quantize(g, "e4m3") == 0)) == 1.0, "sanity check: without scaling every value is below half the E4M3 subnormal spacing"
