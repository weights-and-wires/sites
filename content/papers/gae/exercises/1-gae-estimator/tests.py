import numpy as np
from submission import gae, td_residuals

GAMMA = 0.9
R = np.array([1.0, 0.0, 0.0, 2.0])
V = np.array([0.5, 0.4, 0.6, 0.3])


def random_trajectory(seed, T=12):
    rng = np.random.default_rng(seed)
    return rng.normal(size=T), rng.normal(size=T), rng.normal()


def test_td_residuals_page_example():
    got = td_residuals(R, V, 0.0, GAMMA)
    want = np.array([0.86, 0.14, -0.33, 1.7])
    assert np.allclose(got, want, atol=1e-6), f"expected {want}, got {got}"


def test_td_residuals_use_last_value():
    got = td_residuals(R, V, 5.0, GAMMA)
    want = 2.0 + GAMMA * 5.0 - 0.3
    assert np.isclose(got[-1], want), f"expected last residual {want}, got {got[-1]}"


def test_gae_page_example():
    got = gae(R, V, 0.0, GAMMA, 0.5)
    want = np.array([1.01109, 0.33575, 0.435, 1.7])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_lambda_zero_is_the_td_residual():
    # Equation 17: GAE(gamma, 0) = delta_t
    r, v, last = random_trajectory(1)
    got = gae(r, v, last, 0.97, 0.0)
    want = td_residuals(r, v, last, 0.97)
    assert np.allclose(got, want), f"lambda = 0 should give the TD residuals {want}, got {got}"


def test_lambda_one_is_return_minus_value():
    # Equation 18: GAE(gamma, 1) = sum_l gamma^l r_{t+l} - V(s_t), with the bootstrap value added at the end
    r, v, last = random_trajectory(2)
    gamma = 0.95
    T = len(r)
    want = np.array(
        [sum(gamma ** (i - t) * r[i] for i in range(t, T)) + gamma ** (T - t) * last - v[t] for t in range(T)]
    )
    got = gae(r, v, last, gamma, 1.0)
    assert np.allclose(got, want), f"lambda = 1 should give return minus V:\nexpected {want}\ngot {got}"


def test_matches_direct_sum_of_equation_16():
    r, v, last = random_trajectory(3)
    gamma, lam = 0.93, 0.8
    delta = td_residuals(r, v, last, gamma)
    T = len(r)
    want = np.array([sum((gamma * lam) ** l * delta[t + l] for l in range(T - t)) for t in range(T)])
    got = gae(r, v, last, gamma, lam)
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_decay_is_the_product_gamma_lambda():
    # A single step back: A_{T-2} = delta_{T-2} + gamma * lambda * delta_{T-1}. Both factors matter.
    r = np.array([0.0, 1.0])
    v = np.array([0.0, 0.0])
    for gamma, lam in [(0.5, 1.0), (1.0, 0.5), (0.5, 0.5)]:
        got = gae(r, v, 0.0, gamma, lam)[0]
        want = 0.0 + gamma * lam * 1.0
        assert np.isclose(got, want), f"gamma={gamma}, lambda={lam}: expected A_0 = {want}, got {got}"


def test_output_has_one_advantage_per_step():
    r, v, last = random_trajectory(4, T=7)
    got = gae(r, v, last, 0.99, 0.95)
    assert got.shape == (7,), f"expected shape (7,), got {got.shape}"
