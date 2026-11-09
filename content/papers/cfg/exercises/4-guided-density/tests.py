import numpy as np
from submission import density_stats, guided_density

X = np.linspace(-12, 12, 4801)
DX = X[1] - X[0]


def log_gauss(m, s):
    return -0.5 * ((X - m) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))


def two_class():
    """x | c ~ N(+-2, 1) with equal class probabilities; return log p(x | c=1) and log p(x)."""
    lc = log_gauss(2.0, 1.0)
    return lc, np.logaddexp(lc, log_gauss(-2.0, 1.0)) + np.log(0.5)


def test_normalized():
    lc, lm = two_class()
    for w in (0.0, 1.0, 4.0):
        p = guided_density(lc, lm, w, DX)
        assert np.isclose(p.sum() * DX, 1.0), f"the density must integrate to 1 at w = {w}, got {p.sum() * DX}"
        assert np.all(p >= 0), "a density cannot be negative"


def test_w_zero_is_the_class_density():
    lc, lm = two_class()
    p = guided_density(lc, lm, 0.0, DX)
    want = np.exp(lc)
    assert np.allclose(p, want, atol=1e-6), f"w = 0 must return p(x|c); max difference {np.abs(p - want).max():.2e}"


def test_w_minus_one_is_the_marginal():
    lc, lm = two_class()
    p = guided_density(lc, lm, -1.0, DX)
    want = np.exp(lm)
    assert np.allclose(p, want, atol=1e-6), f"w = -1 must return the marginal p(x); max difference {np.abs(p - want).max():.2e}"


def test_gaussian_closed_form():
    lc, lm = log_gauss(1.0, 1.0), log_gauss(0.0, 2.0)
    p = guided_density(lc, lm, 1.0, DX)
    m, s = density_stats(X, p, DX)
    assert np.isclose(m, 8 / 7, atol=1e-4), f"expected mean 8/7 = 1.1429 (precision 1.75), got {m:.4f}"
    assert np.isclose(s, np.sqrt(1 / 1.75), atol=1e-4), f"expected std sqrt(1 / 1.75) = 0.7559, got {s:.4f}"


def test_two_class_reference_values():
    lc, lm = two_class()
    want = {1.0: (2.0710, 0.9366), 2.0: (2.1023, 0.9135), 4.0: (2.1394, 0.8889)}
    for w, (wm, ws) in want.items():
        m, s = density_stats(X, guided_density(lc, lm, w, DX), DX)
        assert abs(m - wm) < 2e-3 and abs(s - ws) < 2e-3, f"at w = {w} expected mean {wm}, std {ws}; got mean {m:.4f}, std {s:.4f}"


def test_guidance_concentrates_and_moves_away():
    lc, lm = two_class()
    stats = [density_stats(X, guided_density(lc, lm, w, DX), DX) for w in (0.0, 1.0, 2.0, 4.0)]
    means, stds = [s[0] for s in stats], [s[1] for s in stats]
    assert np.all(np.diff(stds) < 0), f"the spread must shrink as w grows, got stds {np.round(stds, 4)}"
    assert np.all(np.diff(means) > 0), f"the mean must move away from the other class as w grows, got means {np.round(means, 4)}"


def test_no_overflow_with_large_logs():
    lc, lm = two_class()
    p = guided_density(lc + 800.0, lm + 800.0, 2.0, DX)
    assert np.all(np.isfinite(p)), "subtract the maximum log density before exponentiating; the result has inf or nan"
    assert np.isclose(p.sum() * DX, 1.0), f"expected integral 1 with offset logs, got {p.sum() * DX}"
    want = guided_density(lc, lm, 2.0, DX)
    assert np.allclose(p, want, atol=1e-8), "a constant added to both log densities must not change the normalized result"


def test_stats_of_a_known_density():
    p = np.exp(log_gauss(1.5, 0.7))
    m, s = density_stats(X, p, DX)
    assert np.isclose(m, 1.5, atol=1e-6) and np.isclose(s, 0.7, atol=1e-6), f"expected mean 1.5, std 0.7; got {m:.6f}, {s:.6f}"
