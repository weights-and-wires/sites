import numpy as np
from submission import discounted_cumsum, gae_via_shaping, shape_rewards

GAMMA = 0.9
R = np.array([1.0, 0.0, 0.0, 2.0])
V = np.array([0.5, 0.4, 0.6, 0.3])


def test_shape_rewards_page_example():
    got = shape_rewards(R, V, 0.0, GAMMA)
    want = np.array([0.86, 0.14, -0.33, 1.7])
    assert np.allclose(got, want, atol=1e-6), f"expected {want}, got {got}"


def test_shape_rewards_uses_last_phi_and_gamma():
    got = shape_rewards([0.0, 0.0], [1.0, 2.0], 10.0, 0.5)
    want = np.array([0.5 * 2.0 - 1.0, 0.5 * 10.0 - 2.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_discounted_cumsum_small_case():
    got = discounted_cumsum([1.0, 2.0, 3.0], 0.5)
    want = np.array([1.0 + 0.5 * 2.0 + 0.25 * 3.0, 2.0 + 0.5 * 3.0, 3.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_discounted_cumsum_matches_direct_sum():
    x = np.random.default_rng(0).normal(size=11)
    want = np.array([sum(0.8**l * x[t + l] for l in range(11 - t)) for t in range(11)])
    got = discounted_cumsum(x, 0.8)
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_shaping_subtracts_phi_from_the_return():
    # Equation 21: sum_l gamma^l r_tilde_{t+l} = sum_l gamma^l r_{t+l} - Phi(s_t), when Phi of the terminal state is 0.
    rng = np.random.default_rng(1)
    r, phi = rng.normal(size=9), rng.normal(size=9)
    gamma = 0.93
    shaped_return = discounted_cumsum(shape_rewards(r, phi, 0.0, gamma), gamma)
    plain_return = np.array([sum(gamma**l * r[t + l] for l in range(9 - t)) for t in range(9)])
    want = plain_return - phi
    assert np.allclose(shaped_return, want), f"expected {want}, got {shaped_return}"


def test_gae_via_shaping_page_example():
    got = gae_via_shaping(R, V, 0.0, GAMMA, 0.5)
    want = np.array([1.01109, 0.33575, 0.435, 1.7])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_gae_via_shaping_matches_equation_16():
    rng = np.random.default_rng(2)
    r, v, last = rng.normal(size=10), rng.normal(size=10), rng.normal()
    gamma, lam = 0.97, 0.9
    nv = np.append(v[1:], last)
    d = r + gamma * nv - v
    want = np.array([sum((gamma * lam) ** l * d[t + l] for l in range(10 - t)) for t in range(10)])
    got = gae_via_shaping(r, v, last, gamma, lam)
    assert np.allclose(got, want), f"expected {want}, got {got}"
