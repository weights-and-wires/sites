import numpy as np
from submission import make_schedule, mean_from_eps, posterior_mean_var, q_sample

betas, alphas, ab = make_schedule(4, 0.1, 0.4)


def bayes(x_t, x0, t):
    """Posterior by Bayes' rule: prior q(x_{t-1}|x0) times likelihood q(x_t|x_{t-1})."""
    ab_prev = 1.0 if t == 1 else ab[t - 2]
    if t == 1:
        return x0, 0.0
    prec = 1 / (1 - ab_prev) + alphas[t - 1] / betas[t - 1]
    mean = (np.sqrt(ab_prev) * x0 / (1 - ab_prev) + np.sqrt(alphas[t - 1]) * x_t / betas[t - 1]) / prec
    return mean, 1 / prec


def test_hand_values_at_t3():
    mean, var = posterior_mean_var(0.5, 2.0, 3, betas, ab)
    assert np.isclose(mean, 1.2626, atol=1e-4), f"expected mean 1.2626, got {mean}"
    assert np.isclose(var, 0.1694, atol=1e-4), f"expected beta_tilde 0.1694, got {var}"


def test_matches_bayes_rule():
    for t in (2, 3, 4):
        for x_t, x0 in ((0.5, 2.0), (-1.3, 0.4), (3.0, -2.0)):
            want_m, want_v = bayes(x_t, x0, t)
            m, v = posterior_mean_var(x_t, x0, t, betas, ab)
            assert np.isclose(m, want_m), f"t={t}: expected mean {want_m:.4f}, got {m}"
            assert np.isclose(v, want_v), f"t={t}: expected variance {want_v:.4f}, got {v}"


def test_posterior_variance_is_below_beta():
    for t in (2, 3, 4):
        _, v = posterior_mean_var(0.0, 0.0, t, betas, ab)
        assert v < betas[t - 1], f"t={t}: beta_tilde must be smaller than beta_t = {betas[t - 1]}, got {v}"


def test_first_step_returns_x0():
    mean, var = posterior_mean_var(0.7, 2.0, 1, betas, ab)
    assert np.isclose(var, 0.0), f"at t = 1 expected variance 0 (alpha_bar_0 = 1), got {var}"
    assert np.isclose(mean, 2.0), f"at t = 1 expected the mean to be x0 = 2.0, got {mean}"


def test_vector_inputs():
    x_t = np.array([[0.5, -1.0], [2.0, 0.1]])
    x0 = np.array([[2.0, 0.3], [-1.0, 0.4]])
    mean, _ = posterior_mean_var(x_t, x0, 3, betas, ab)
    assert mean.shape == (2, 2), f"expected shape (2, 2), got {mean.shape}"
    m00, _ = bayes(0.5, 2.0, 3)
    assert np.isclose(mean[0, 0], m00), f"expected mean[0, 0] = {m00:.4f}, got {mean[0, 0]}"


def test_eps_form_equals_posterior_mean():
    rng = np.random.default_rng(1)
    for t in (1, 2, 3, 4):
        x0 = rng.normal(size=(5, 3))
        eps = rng.normal(size=(5, 3))
        x_t = q_sample(x0, t, eps, ab)
        want, _ = posterior_mean_var(x_t, x0, t, betas, ab)
        got = mean_from_eps(x_t, eps, t, betas, ab)
        assert np.allclose(got, want), f"t={t}: Equation 11 with the true epsilon must equal Equation 7; expected\n{want}\ngot\n{got}"


def test_eps_form_hand_value():
    got = mean_from_eps(0.5, -1.3061, 3, betas, ab)
    assert np.isclose(got, 1.2626, atol=1e-3), f"expected 1.2626, got {got}"
