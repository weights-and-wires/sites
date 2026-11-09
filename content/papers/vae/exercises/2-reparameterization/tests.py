import numpy as np
from submission import reparameterize, pathwise_grads


def test_reparameterize_values():
    z = reparameterize(np.array([1.0, 0.0]), np.log(np.array([4.0, 0.25])), np.array([1.0, 2.0]))
    assert np.allclose(z, [3.0, 1.0]), f"expected [3.0, 1.0], got {z}"


def test_zero_noise_gives_mean():
    z = reparameterize(np.array([0.7, -2.0]), np.array([3.0, -1.0]), np.zeros(2))
    assert np.allclose(z, [0.7, -2.0]), f"eps = 0 must return mu, got {z}"


def test_sample_moments():
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(200_000)
    z = reparameterize(1.5, np.log(0.25), eps)
    assert abs(z.mean() - 1.5) < 0.01, f"expected mean 1.5, got {z.mean():.4f}"
    assert abs(z.std() - 0.5) < 0.01, f"expected std 0.5, got {z.std():.4f}"


def test_gradient_of_square_exact_given_eps():
    # f(z) = z^2. With eps fixed, d/dmu mean((mu + s e)^2) = mean(2 z), d/dsigma = mean(2 z e).
    eps = np.array([-1.0, 0.0, 1.0, 2.0])
    mu, s = 1.0, 2.0
    z = mu + s * eps
    want = (np.mean(2 * z), np.mean(2 * z * eps))
    got = pathwise_grads(mu, s, eps, lambda z: 2 * z)
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_gradient_of_square_converges():
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(200_000)
    d_mu, d_s = pathwise_grads(1.0, 1.0, eps, lambda z: 2 * z)
    assert abs(d_mu - 2.0) < 0.03, f"d/dmu expected 2.0 (= 2 mu), got {d_mu:.4f}"
    assert abs(d_s - 2.0) < 0.05, f"d/dsigma expected 2.0 (= 2 sigma), got {d_s:.4f}"


def test_sigma_gradient_uses_the_noise():
    # For f(z) = z, E[f] = mu: d/dmu = 1 and d/dsigma = E[eps] = 0 up to sampling error.
    eps = np.array([1.0, 3.0])
    d_mu, d_s = pathwise_grads(0.5, 2.0, eps, lambda z: np.ones_like(z))
    assert np.isclose(d_mu, 1.0), f"d/dmu expected 1.0, got {d_mu}"
    assert np.isclose(d_s, 2.0), f"d/dsigma expected mean(eps) = 2.0, got {d_s}"
