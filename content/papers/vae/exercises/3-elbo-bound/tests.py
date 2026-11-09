import numpy as np
from submission import elbo_estimate, log_px

EPS = np.random.default_rng(0).standard_normal(200_000)


def test_page_example():
    got = elbo_estimate(1.0, 0.2, np.log(0.64), EPS)
    assert abs(got - (-1.6221)) < 0.01, f"expected about -1.6221, got {got:.4f}"


def test_bound_is_below_log_evidence():
    for mu, var in [(0.0, 1.0), (0.2, 0.64), (2.0, 0.3), (-1.0, 2.0)]:
        got = elbo_estimate(1.0, mu, np.log(var), EPS)
        assert got < log_px(1.0), f"ELBO {got:.4f} must be below log p(x) = {log_px(1.0):.4f} for q = N({mu}, {var})"


def test_bound_touches_at_true_posterior():
    x = 1.0
    got = elbo_estimate(x, x / 2, np.log(0.5), EPS)
    assert abs(got - log_px(x)) < 0.01, f"at the true posterior expected {log_px(x):.4f}, got {got:.4f}"


def test_true_posterior_for_other_x():
    for x in (-2.0, 3.0):
        got = elbo_estimate(x, x / 2, np.log(0.5), EPS)
        assert abs(got - log_px(x)) < 0.02, f"x={x}: expected {log_px(x):.4f}, got {got:.4f}"


def test_gap_equals_kl_to_posterior():
    x, mu, var = 1.0, 0.2, 0.64
    kl = 0.5 * (np.log(0.5 / var) + (var + (mu - 0.5) ** 2) / 0.5 - 1)
    gap = log_px(x) - elbo_estimate(x, mu, np.log(var), EPS)
    assert abs(gap - kl) < 0.01, f"Equation 1: gap expected {kl:.4f}, got {gap:.4f}"
