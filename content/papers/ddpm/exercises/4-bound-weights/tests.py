import numpy as np
from submission import bound_weight, kl_term_from_eps, make_schedule

betas, alphas, ab = make_schedule(1000, 1e-4, 0.02)


def beta_tilde(t):
    ab_prev = 1.0 if t == 1 else ab[t - 2]
    return (1 - ab_prev) / (1 - ab[t - 1]) * betas[t - 1]


def test_weight_at_t1_with_sigma_beta():
    w = bound_weight(1, betas, ab, betas[0])
    assert np.isclose(w, 0.5001, atol=1e-4), f"expected w_1 = 0.5001 for sigma^2 = beta_t, got {w}"


def test_weight_at_t1000_with_sigma_beta():
    w = bound_weight(1000, betas, ab, betas[999])
    assert np.isclose(w, 0.0102, atol=1e-4), f"expected w_1000 = 0.0102 for sigma^2 = beta_t, got {w}"


def test_weight_with_beta_tilde():
    w = bound_weight(2, betas, ab, beta_tilde(2))
    assert np.isclose(w, 0.5997, atol=1e-4), f"expected w_2 = 0.5997 for sigma^2 = beta_tilde_t, got {w}"


def test_small_t_is_weighted_more_in_the_bound():
    w1, w1000 = bound_weight(1, betas, ab, betas[0]), bound_weight(1000, betas, ab, betas[999])
    assert np.isclose(w1 / w1000, 49.0, rtol=0.02), f"expected w_1 / w_1000 near 49, got {w1 / w1000:.2f}"


def test_matches_direct_gaussian_kl():
    rng = np.random.default_rng(0)
    for t in (2, 10, 500, 1000):
        x0 = rng.normal(size=4)
        eps = rng.normal(size=4)
        eps_pred = eps + 0.3 * rng.normal(size=4)
        x_t = np.sqrt(ab[t - 1]) * x0 + np.sqrt(1 - ab[t - 1]) * eps
        s2 = beta_tilde(t)
        # Direct route: tilde-mu from Equation 7, mu_theta from Equation 11, then the Gaussian KL.
        ab_prev = ab[t - 2]
        mu_tilde = np.sqrt(ab_prev) * betas[t - 1] / (1 - ab[t - 1]) * x0 + np.sqrt(alphas[t - 1]) * (1 - ab_prev) / (1 - ab[t - 1]) * x_t
        mu_theta = (x_t - betas[t - 1] / np.sqrt(1 - ab[t - 1]) * eps_pred) / np.sqrt(alphas[t - 1])
        want = np.sum((mu_tilde - mu_theta) ** 2) / (2 * s2)
        got = kl_term_from_eps(eps, eps_pred, t, betas, ab, s2)
        assert np.isclose(got, want, rtol=1e-6), f"t={t}: expected KL {want:.6g}, got {got:.6g}"


def test_zero_when_prediction_is_exact():
    eps = np.array([0.3, -1.2])
    got = kl_term_from_eps(eps, eps.copy(), 5, betas, ab, beta_tilde(5))
    assert got == 0.0, f"expected 0 for an exact epsilon, got {got}"


def test_sums_over_dimensions():
    got = kl_term_from_eps(np.ones(3), np.zeros(3), 10, betas, ab, betas[9])
    want = 3 * bound_weight(10, betas, ab, betas[9])
    assert np.isclose(got, want), f"expected 3 * w_10 = {want:.6f}, got {got:.6f}"
