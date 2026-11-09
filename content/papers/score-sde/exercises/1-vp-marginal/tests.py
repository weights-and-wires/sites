import numpy as np
from submission import euler_maruyama_vp, int_beta, vp_kernel


def test_int_beta_values():
    # beta(s) = 0.1 + 19.9 s, so the integral to t = 1 is 0.1 + 19.9 / 2 = 10.05.
    assert np.isclose(int_beta(1.0), 10.05), f"expected 10.05 at t = 1, got {int_beta(1.0)}"
    assert np.isclose(int_beta(0.5), 2.5375), f"expected 2.5375 at t = 0.5, got {int_beta(0.5)}"
    assert np.isclose(int_beta(0.0), 0.0), f"expected 0 at t = 0, got {int_beta(0.0)}"


def test_int_beta_custom_schedule():
    got = int_beta(2.0, bmin=1.0, bmax=3.0)  # beta(s) = 1 + 2 s, integral = 2 + 4
    assert np.isclose(got, 6.0), f"expected 6.0 for beta(s) = 1 + 2s on [0, 2], got {got}"


def test_kernel_values():
    m, v = vp_kernel(0.5)
    assert np.isclose(m, 0.281183, atol=1e-5), f"expected mean scale 0.281183 at t = 0.5, got {m}"
    assert np.isclose(v, 0.920936, atol=1e-5), f"expected variance 0.920936 at t = 0.5, got {v}"


def test_kernel_limits():
    m0, v0 = vp_kernel(0.0)
    assert np.isclose(m0, 1.0) and np.isclose(v0, 0.0), f"at t = 0 expected (1, 0), got ({m0}, {v0})"
    m1, v1 = vp_kernel(1.0)
    assert m1 < 0.01, f"expected the signal to be almost gone at t = 1 (0.0066), got {m1}"
    assert abs(v1 - 1.0) < 1e-4, f"expected variance close to 1 at t = 1, got {v1}"


def test_kernel_accepts_arrays():
    t = np.array([0.1, 0.5, 1.0])
    m, v = vp_kernel(t)
    assert m.shape == (3,) and v.shape == (3,), f"expected shape (3,), got {m.shape} and {v.shape}"
    assert np.all(np.diff(m) < 0) and np.all(np.diff(v) > 0), "the mean scale must fall and the variance must rise with t"


def test_euler_maruyama_matches_kernel():
    rng = np.random.default_rng(0)
    x = euler_maruyama_vp(np.full(40000, 2.0), 0.5, 500, rng)
    m, v = vp_kernel(0.5)
    assert abs(x.mean() - 2.0 * m) < 0.03, f"expected mean {2.0 * m:.4f}, got {x.mean():.4f}"
    assert abs(x.var() - v) < 0.03, f"expected variance {v:.4f}, got {x.var():.4f}"


def test_euler_maruyama_early_time():
    rng = np.random.default_rng(1)
    x = euler_maruyama_vp(np.full(40000, 2.0), 0.1, 200, rng)
    m, v = vp_kernel(0.1)
    assert abs(x.mean() - 2.0 * m) < 0.03, f"expected mean {2.0 * m:.4f}, got {x.mean():.4f}"
    assert abs(x.var() - v) < 0.02, f"expected variance {v:.4f}, got {x.var():.4f}"


def test_variance_is_preserved_for_unit_variance_data():
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal(40000)
    x = euler_maruyama_vp(x0, 0.5, 500, rng)
    assert abs(x.var() - 1.0) < 0.04, f"unit-variance data must keep variance 1 (Appendix B), got {x.var():.4f}"
    assert abs(x.mean()) < 0.03, f"expected mean 0, got {x.mean():.4f}"


def test_shape_and_input_untouched():
    rng = np.random.default_rng(3)
    x0 = np.ones((5, 3))
    out = euler_maruyama_vp(x0, 0.2, 20, rng)
    assert out.shape == (5, 3), f"expected shape (5, 3), got {out.shape}"
    assert np.all(x0 == 1.0), "the input array must not be modified in place"
