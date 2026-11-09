import numpy as np
from submission import gae_mixture, k_step_advantage

GAMMA = 0.9
R = np.array([1.0, 0.0, 0.0, 2.0])
V = np.array([0.5, 0.4, 0.6, 0.3])


def deltas(r, v, gamma):
    nv = np.append(v[1:], 0.0)
    return r + gamma * nv - v


def eq16(r, v, gamma, lam):
    d = deltas(r, v, gamma)
    T = len(r)
    return np.array([sum((gamma * lam) ** l * d[t + l] for l in range(T - t)) for t in range(T)])


def test_k_step_page_example():
    got = [k_step_advantage(R, V, GAMMA, 0, k) for k in (1, 2, 3, 4)]
    want = [0.86, 0.986, 0.7187, 1.958]
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_one_step_is_the_td_residual():
    rng = np.random.default_rng(0)
    r, v = rng.normal(size=8), rng.normal(size=8)
    d = deltas(r, v, 0.95)
    got = [k_step_advantage(r, v, 0.95, t, 1) for t in range(8)]
    assert np.allclose(got, d), f"A^(1) should be delta_t = {d}, got {got}"


def test_k_step_is_a_discounted_sum_of_residuals():
    # Equation 14: A^(k) = sum_{l<k} gamma^l delta_{t+l}, so the two forms must agree.
    rng = np.random.default_rng(1)
    r, v = rng.normal(size=9), rng.normal(size=9)
    d = deltas(r, v, 0.9)
    for t, k in [(0, 3), (2, 4), (5, 2), (3, 5)]:
        want = sum(0.9**l * d[t + l] for l in range(k))
        got = k_step_advantage(r, v, 0.9, t, k)
        assert np.isclose(got, want), f"t={t}, k={k}: expected {want}, got {got}"


def test_k_beyond_the_end_is_capped():
    got = k_step_advantage(R, V, GAMMA, 2, 50)
    want = k_step_advantage(R, V, GAMMA, 2, 2)
    assert np.isclose(got, want), f"k beyond the end should use the whole remaining return: expected {want}, got {got}"


def test_mixture_page_example():
    got = gae_mixture(R, V, GAMMA, 0.5)
    want = np.array([1.01109, 0.33575, 0.435, 1.7])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_mixture_equals_equation_16():
    rng = np.random.default_rng(2)
    r, v = rng.normal(size=10), rng.normal(size=10)
    for lam in (0.0, 0.3, 0.9, 0.99):
        got = gae_mixture(r, v, 0.97, lam)
        want = eq16(r, v, 0.97, lam)
        assert np.allclose(got, want), f"lambda={lam}: expected {want}, got {got}"


def test_lambda_one_is_return_minus_value():
    # Equation 18. All of the weight must land on the longest estimator.
    got = gae_mixture(R, V, GAMMA, 1.0)
    want = np.array([1.958, 1.22, 1.2, 1.7])
    assert np.allclose(got, want, atol=1e-6), f"expected {want}, got {got}"
