import numpy as np
from submission import gaussian_vf, vp_path

BMIN, BMAX = 0.1, 20.0


def vp_eq19(t, x, x1):
    """Equation 19, written out with T(1 - t)."""
    s = 1.0 - t
    T = BMIN * s + 0.5 * s**2 * (BMAX - BMIN)
    Tp = BMIN + s * (BMAX - BMIN)
    return -Tp / 2 * (np.exp(-T) * x - np.exp(-0.5 * T) * x1) / (1 - np.exp(-T))


def ot_pieces(sm):
    return (
        lambda t, x1: t * x1,
        lambda t, x1: x1,
        lambda t: 1 - (1 - sm) * t,
        lambda t: -(1 - sm),
    )


def test_gaussian_vf_reproduces_the_ot_field():
    sm = 0.1
    x = np.array([[0.45], [1.0]])
    x1 = np.array([[2.0], [2.0]])
    got = gaussian_vf(x, x1, 0.5, *ot_pieces(sm))
    want = (x1 - (1 - sm) * x) / (1 - (1 - sm) * 0.5)
    assert np.allclose(got, want), f"Equation 21 gives {want}, got {got}"
    assert np.allclose(got[0], [2.9]), f"on the page's example the field is 2.9, got {got[0]}"


def test_gaussian_vf_uses_both_terms():
    # sigma constant: only the mean term survives. mu constant: only the sigma term.
    x = np.array([[1.0]])
    x1 = np.array([[3.0]])
    const_sigma = gaussian_vf(x, x1, 0.4, lambda t, a: t * a, lambda t, a: a, lambda t: 2.0, lambda t: 0.0)
    assert np.allclose(const_sigma, [[3.0]]), f"with sigma' = 0 expected mu' = 3, got {const_sigma}"
    const_mu = gaussian_vf(x, x1, 0.4, lambda t, a: 0.0 * a, lambda t, a: 0.0 * a, lambda t: 2.0, lambda t: 1.0)
    assert np.allclose(const_mu, [[0.5]]), f"with mu = 0 expected sigma'/sigma * x = 0.5, got {const_mu}"


def test_vp_path_values():
    mu, dmu, sigma, dsigma = vp_path(BMIN, BMAX)
    x1 = np.array([[2.0]])
    assert np.allclose(mu(0.5, x1), [[0.5624]], atol=1e-4), f"expected mu_0.5 = 0.5624, got {mu(0.5, x1)}"
    assert np.isclose(sigma(0.5), 0.9597, atol=1e-4), f"expected sigma_0.5 = 0.9597, got {sigma(0.5)}"


def test_vp_derivatives_match_finite_differences():
    mu, dmu, sigma, dsigma = vp_path(BMIN, BMAX)
    x1 = np.array([[2.0]])
    h = 1e-6
    for t in (0.25, 0.5, 0.75):
        fd_mu = (mu(t + h, x1) - mu(t - h, x1)) / (2 * h)
        fd_sg = (sigma(t + h) - sigma(t - h)) / (2 * h)
        assert np.allclose(dmu(t, x1), fd_mu, rtol=1e-4), f"t = {t}: d mu/dt expected {fd_mu}, got {dmu(t, x1)}"
        assert np.isclose(dsigma(t), fd_sg, rtol=1e-4), f"t = {t}: d sigma/dt expected {fd_sg}, got {dsigma(t)}"


def test_vp_field_matches_equation_19():
    pieces = vp_path(BMIN, BMAX)
    mu, _, sigma, _ = pieces
    x1 = np.array([[2.0]])
    x0 = np.array([[-1.0]])
    for t, want in ((0.25, 0.9073), (0.5, 3.2399), (0.75, 5.5980)):
        x = mu(t, x1) + sigma(t) * x0
        got = gaussian_vf(x, x1, t, *pieces)
        eq19 = vp_eq19(t, x, x1)
        assert np.allclose(got, eq19), f"t = {t}: Equation 19 gives {eq19}, got {got}"
        assert np.allclose(got, [[want]], atol=1e-3), f"t = {t}: expected {want} from the page, got {got}"


def test_vp_field_points_toward_data_late_in_the_path():
    pieces = vp_path(BMIN, BMAX)
    mu, _, sigma, _ = pieces
    x1 = np.array([[2.0]])
    x = mu(0.9, x1) - 0.3  # a little below the mean, so the field must push up
    got = gaussian_vf(x, x1, 0.9, *pieces)
    assert got[0, 0] > 0, f"the field at t = 0.9 should be positive, got {got}"
