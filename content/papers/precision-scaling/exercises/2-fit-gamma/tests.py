import numpy as np
from submission import n_eff_fraction, fit_gamma

A, B, E, ALPHA = 4.299e3, 1.806e4, 2.7648, 0.4965
GAMMA_TRUE = 2.6745
N, D = 60e6, 6e9
PS = np.arange(3, 13)


def make_losses(ps, gamma=GAMMA_TRUE):
    """Losses from Equation 3 for a known gamma, and the full-precision loss."""
    ps = np.asarray(ps, dtype=float)
    loss = A * (N * (1 - np.exp(-ps / gamma))) ** -ALPHA + B * D**-ALPHA + E
    loss_inf = A * N**-ALPHA + B * D**-ALPHA + E
    return loss, loss_inf


def test_fraction_is_one_at_full_precision():
    _, loss_inf = make_losses([4])
    f = float(n_eff_fraction(loss_inf, loss_inf, N))
    assert abs(f - 1.0) < 1e-9, f"no loss gap means N_eff = N, expected 1.0, got {f}"


def test_fraction_inverts_the_loss():
    loss, loss_inf = make_losses(PS)
    f = n_eff_fraction(loss, loss_inf, N)
    want = 1 - np.exp(-PS / GAMMA_TRUE)
    assert np.allclose(f, want, atol=1e-8), f"expected {np.round(want, 4)}, got {np.round(f, 4)}"


def test_fraction_depends_on_n():
    # The same loss gap means a different fraction at a different N.
    f1 = float(n_eff_fraction(3.5, 3.3, 60e6))
    f2 = float(n_eff_fraction(3.5, 3.3, 220e6))
    assert abs(f1 - 0.5559) < 2e-3 and abs(f2 - 0.3669) < 2e-3, f"expected 0.5559 and 0.3669, got {f1} and {f2}"


def test_fit_recovers_gamma_exactly():
    loss, loss_inf = make_losses(PS)
    g = fit_gamma(PS, n_eff_fraction(loss, loss_inf, N))
    assert abs(g - GAMMA_TRUE) < 1e-6, f"expected gamma {GAMMA_TRUE}, got {g}"


def test_fit_recovers_another_gamma():
    loss, loss_inf = make_losses(PS, gamma=1.2)
    g = fit_gamma(PS, n_eff_fraction(loss, loss_inf, N))
    assert abs(g - 1.2) < 1e-6, f"expected gamma 1.2, got {g}"


def test_fit_with_noise():
    ps = PS[:5]
    loss, loss_inf = make_losses(ps)
    noisy = loss + np.random.default_rng(0).normal(0, 1e-3, len(ps))
    g = fit_gamma(ps, n_eff_fraction(noisy, loss_inf, N))
    assert abs(g - GAMMA_TRUE) < 0.1, f"with noise 1e-3 expected gamma near {GAMMA_TRUE}, got {g}"


def test_fit_is_a_scale_not_a_rate():
    # Larger gamma means a slower approach to 1.
    g = fit_gamma([2.0, 4.0], [1 - np.exp(-1.0), 1 - np.exp(-2.0)])
    assert abs(g - 2.0) < 1e-9, f"f = 1 - exp(-P/2) at P = 2, 4: expected gamma 2.0, got {g}"
