import numpy as np
from submission import alpha_sigma, corrupt, drop_condition, sample_lambda


def test_alpha_sigma_values():
    a, s = alpha_sigma(0.0)
    assert np.isclose(a, np.sqrt(0.5)) and np.isclose(s, np.sqrt(0.5)), f"lambda = 0 means signal equals noise; expected both 0.7071, got {a}, {s}"
    a, s = alpha_sigma(4.0)
    assert np.isclose(a**2, 0.982014, atol=1e-5), f"expected alpha^2 = 0.982014 at lambda = 4, got {a**2}"
    assert np.isclose(s**2, 0.017986, atol=1e-5), f"expected sigma^2 = 0.017986 at lambda = 4, got {s**2}"


def test_alpha_sigma_variance_preserving():
    lam = np.linspace(-20, 20, 41)
    a, s = alpha_sigma(lam)
    assert a.shape == (41,), f"expected an array of shape (41,), got {np.shape(a)}"
    assert np.allclose(a**2 + s**2, 1.0), "alpha^2 + sigma^2 must equal 1 at every lambda"
    assert np.all(np.diff(a) > 0) and np.all(np.diff(s) < 0), "signal must grow and noise must shrink as lambda increases"


def test_lambda_log_snr_definition():
    lam = np.array([-3.0, 0.5, 2.0])
    a, s = alpha_sigma(lam)
    assert np.allclose(np.log(a**2 / s**2), lam), f"lambda must be log(alpha^2 / sigma^2); expected {lam}, got {np.log(a**2 / s**2)}"


def test_sample_lambda_endpoints():
    assert np.isclose(sample_lambda(0.0), 20.0), f"u = 0 must give lambda_max = 20, got {sample_lambda(0.0)}"
    assert np.isclose(sample_lambda(1.0), -20.0), f"u = 1 must give lambda_min = -20, got {sample_lambda(1.0)}"
    assert np.isclose(sample_lambda(0.0, -10.0, 15.0), 15.0), f"expected lambda_max = 15 at u = 0, got {sample_lambda(0.0, -10.0, 15.0)}"
    assert np.isclose(sample_lambda(1.0, -10.0, 15.0), -10.0), f"expected lambda_min = -10 at u = 1, got {sample_lambda(1.0, -10.0, 15.0)}"


def test_sample_lambda_interior():
    got = sample_lambda(np.array([0.25, 0.5, 0.75]))
    want = np.array([1.7627, 0.0, -1.7627])
    assert np.allclose(got, want, atol=2e-3), f"expected {want} at u = 0.25, 0.5, 0.75, got {got}"


def test_sample_lambda_is_hyperbolic_secant_like():
    lam = sample_lambda(np.random.default_rng(0).random(100_000))
    assert abs(np.median(lam)) < 0.05, f"p(lambda) is symmetric, so the median should be near 0, got {np.median(lam):.3f}"
    frac = np.mean(np.abs(lam) < 2.0)
    assert 0.55 < frac < 0.62, f"expected about 59% of lambda within [-2, 2] (more mass near 0 than a uniform lambda), got {frac:.3f}"


def test_drop_condition_rate_and_values():
    rng = np.random.default_rng(1)
    c = rng.integers(0, 10, size=20_000)
    out = drop_condition(c, 0.2, rng, null=-1)
    rate = np.mean(out == -1)
    assert abs(rate - 0.2) < 0.02, f"expected about 20% of labels dropped, got {rate:.3f}"
    kept = out != -1
    assert np.array_equal(out[kept], c[kept]), "labels that are not dropped must keep their value"


def test_drop_condition_edge_cases_and_copy():
    rng = np.random.default_rng(2)
    c = np.arange(50)
    keep = c.copy()
    assert np.array_equal(drop_condition(c, 0.0, rng, null=-1), keep), "p_uncond = 0 must keep every label"
    assert np.all(drop_condition(c, 1.0, rng, null=-1) == -1), "p_uncond = 1 must drop every label"
    drop_condition(c, 0.5, rng, null=-1)
    assert np.array_equal(c, keep), "drop_condition must not change the caller's array"
    assert np.all(drop_condition(c, 1.0, rng, null=99) == 99), "the null token must be the one passed in"


def test_drop_condition_is_per_example():
    rng = np.random.default_rng(3)
    out = drop_condition(np.zeros(1000, dtype=int), 0.5, rng, null=-1)
    n_null = int(np.sum(out == -1))
    assert 300 < n_null < 700, f"each example needs its own coin flip; a batch should mix both kinds, got {n_null} of 1000 dropped"


def test_corrupt_limits_and_broadcast():
    x = np.array([[2.0, -1.0], [0.5, 1.0]])
    eps = np.array([[1.0, 1.0], [-1.0, 2.0]])
    a, s = alpha_sigma(0.0)
    got = corrupt(x, 0.0, eps)
    assert np.allclose(got, a * x + s * eps), f"expected alpha x + sigma eps = {a * x + s * eps}, got {got}"
    got = corrupt(x, np.array([4.0, -4.0]), np.zeros_like(x))
    want = np.sqrt([0.982014, 0.017986])[:, None] * x
    assert np.allclose(got, want, atol=1e-4), f"each row needs its own lambda; expected\n{want}\ngot\n{got}"


def test_corrupt_is_variance_preserving():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((100_000, 1))
    eps = rng.standard_normal((100_000, 1))
    for lam in (-3.0, 0.0, 3.0):
        v = corrupt(x, lam, eps).var()
        assert abs(v - 1.0) < 0.03, f"unit-variance data stays at unit variance; expected 1 at lambda = {lam}, got {v:.3f}"
