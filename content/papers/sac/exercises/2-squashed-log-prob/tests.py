import math

import numpy as np
from submission import squashed_log_prob


def cdf(a, mu, std):
    return 0.5 * (1 + math.erf((math.atanh(a) - mu) / (std * math.sqrt(2))))


def test_page_example():
    got = squashed_log_prob(np.array([0.5]), np.array([0.0]), np.array([1.0]))
    assert np.isclose(got, -0.8037, atol=1e-4), f"expected -0.8037, got {got}"
    assert np.ndim(got) == 0, f"a (D,) input must give a scalar, got shape {np.shape(got)}"


def test_matches_the_density_of_tanh_u():
    # The slope of P(tanh(u) <= a) is the density of a.
    for u, mu, std in ((0.5, 0.0, 1.0), (-1.2, 0.3, 0.7), (0.1, -0.5, 1.5)):
        a = math.tanh(u)
        h = 1e-6
        want = (cdf(a + h, mu, std) - cdf(a - h, mu, std)) / (2 * h)
        got = math.exp(float(squashed_log_prob(np.array([u]), np.array([mu]), np.array([std]))))
        assert np.isclose(got, want, rtol=1e-4), f"u={u}, mu={mu}, std={std}: density from the CDF is {want}, got {got}"


def test_density_integrates_to_one():
    grid = np.linspace(-0.9999, 0.9999, 400001)
    u = np.arctanh(grid)
    logp = np.array([squashed_log_prob(np.array([x]), np.array([0.2]), np.array([0.8])) for x in u[::100]])
    dens = np.exp(logp)
    total = np.sum(dens) * (grid[100] - grid[0])
    assert np.isclose(total, 1.0, atol=2e-3), f"the density over a in (-1, 1) must integrate to 1, got {total}"


def test_sums_over_dimensions():
    u = np.array([0.5, -1.0])
    mu = np.array([0.0, 0.5])
    std = np.array([1.0, 0.5])
    want = sum(float(squashed_log_prob(u[i:i + 1], mu[i:i + 1], std[i:i + 1])) for i in range(2))
    got = squashed_log_prob(u, mu, std)
    assert np.isclose(got, want), f"expected the sum of the two 1-D values, {want}, got {got}"
    assert np.isclose(got, -4.6619, atol=1e-4), f"expected -4.6619, got {got}"


def test_batch_shape_and_axis():
    rng = np.random.default_rng(1)
    u = rng.normal(size=(5, 3))
    mu = rng.normal(size=(5, 3))
    std = np.exp(rng.normal(size=(5, 3)) * 0.3)
    got = np.asarray(squashed_log_prob(u, mu, std))
    assert got.shape == (5,), f"expected shape (5,), got {got.shape}"
    for b in range(5):
        want = squashed_log_prob(u[b], mu[b], std[b])
        assert np.isclose(got[b], want), f"row {b}: expected {want}, got {got[b]}"


def test_broadcasts_scalar_mu_and_std():
    u = np.array([[0.5, 0.5]])
    got = np.asarray(squashed_log_prob(u, 0.0, 1.0))
    assert got.shape == (1,), f"expected shape (1,), got {got.shape}"
    assert np.isclose(got[0], -1.6074, atol=1e-4), f"expected 2 * -0.8037 = -1.6074, got {got[0]}"


def test_large_u_stays_finite():
    got = squashed_log_prob(np.array([25.0]), np.array([0.0]), np.array([1.0]))
    want = -0.5 * 25.0**2 - 0.5 * math.log(2 * math.pi) + 48.6137
    assert np.isfinite(got), f"u = 25 must give a finite log-prob, got {got}"
    assert np.isclose(got, want, atol=1e-3), f"expected {want}, got {got}"
    neg = squashed_log_prob(np.array([-25.0]), np.array([0.0]), np.array([1.0]))
    assert np.isclose(neg, want, atol=1e-3), f"the density is symmetric in u: expected {want}, got {neg}"
