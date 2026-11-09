import numpy as np
from submission import neg_elbo, grad_neg_elbo, fit


def _numeric(w, c, x, eps, h=1e-6):
    dw = (neg_elbo(w + h, c, x, eps) - neg_elbo(w - h, c, x, eps)) / (2 * h)
    dc = (neg_elbo(w, c + h, x, eps) - neg_elbo(w, c - h, x, eps)) / (2 * h)
    return dw, dc


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    x, eps = rng.normal(size=16), rng.normal(size=16)
    for w, c in [(0.0, 0.0), (0.5, -0.7), (-1.2, 0.9)]:
        want = _numeric(w, c, x, eps)
        got = grad_neg_elbo(w, c, x, eps)
        assert np.allclose(got, want, atol=1e-5), f"w={w}, c={c}: expected {want}, got {got}"


def test_gradient_reaches_log_variance_through_the_sample():
    # With x = 0 and eps = 0, z = 0 and the reconstruction term adds nothing to dc; only the KL part remains.
    got = grad_neg_elbo(0.3, 0.4, np.zeros(4), np.zeros(4))
    want = 0.5 * (np.exp(0.4) - 1)
    assert np.isclose(got[1], want), f"dc expected {want:.4f}, got {got[1]:.4f}"
    assert np.isclose(got[0], 0.0), f"dw expected 0.0 for x = 0, got {got[0]}"


def test_gradient_shapes_are_scalars():
    dw, dc = grad_neg_elbo(0.1, 0.1, np.ones(5), np.ones(5))
    assert np.ndim(dw) == 0 and np.ndim(dc) == 0, f"expected scalars, got shapes {np.shape(dw)} and {np.shape(dc)}"


def test_fit_recovers_posterior():
    w, c = fit()
    assert abs(w - 0.5) < 0.05, f"expected w near 0.5 (posterior mean x / 2), got {w:.3f}"
    assert abs(c - np.log(0.5)) < 0.08, f"expected c near {np.log(0.5):.3f} (posterior variance 0.5), got {c:.3f}"
