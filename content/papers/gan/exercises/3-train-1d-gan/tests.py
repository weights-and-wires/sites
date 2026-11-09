import numpy as np
from submission import d_grad, g_grad, train, sigmoid, features


def d_loss(pd, x_real, x_fake):
    return -np.mean(np.log(sigmoid(features(x_real) @ pd))) - np.mean(np.log(1 - sigmoid(features(x_fake) @ pd)))


def g_loss(pg, pd, z):
    x = pg[0] + np.exp(pg[1]) * z
    return -np.mean(np.log(sigmoid(features(x) @ pd)))


def numeric_grad(f, p, eps=1e-6):
    g = np.zeros_like(p)
    for i in range(p.size):
        up, down = p.copy(), p.copy()
        up[i] += eps
        down[i] -= eps
        g[i] = (f(up) - f(down)) / (2 * eps)
    return g


def test_d_grad_matches_finite_differences():
    rng = np.random.default_rng(0)
    xr, xf = 3 + rng.normal(size=16), rng.normal(size=16)
    pd = np.array([0.3, -0.1, 0.2])
    want = numeric_grad(lambda p: d_loss(p, xr, xf), pd)
    got = d_grad(pd, xr, xf)
    assert np.allclose(got, want, atol=1e-6), f"expected {want}, got {got}"


def test_d_grad_by_hand():
    # At pd = 0, D = 1/2 everywhere. Real x = 1 contributes -0.5 * [1, 1, 1]; fake x = -1 contributes +0.5 * [-1, 1, 1].
    got = d_grad(np.zeros(3), np.array([1.0]), np.array([-1.0]))
    want = np.array([-1.0, 0.0, 0.0])
    assert np.allclose(got, want), f"expected {want}, got {got}; real points raise D, fake points lower it"


def test_g_grad_matches_finite_differences():
    rng = np.random.default_rng(1)
    z = rng.normal(size=16)
    pd = np.array([0.8, -0.1, -1.5])
    pg = np.array([0.5, -0.3])
    want = numeric_grad(lambda p: g_loss(p, pd, z), pg)
    got = g_grad(pg, pd, z)
    assert np.allclose(got, want, atol=1e-6), f"expected {want}, got {got}; remember dx/ds = exp(s) z"


def test_g_grad_moves_toward_the_data():
    # D prefers larger x (w1 > 0, w2 = 0), so descending on the loss must raise mu.
    z = np.random.default_rng(2).normal(size=64)
    grad = g_grad(np.array([0.0, 0.0]), np.array([1.0, 0.0, 0.0]), z)
    assert grad[0] < 0, f"d loss / d mu should be negative so that mu rises; got {grad[0]}"


def test_training_recovers_the_gaussian():
    for seed in (0, 1):
        mu, s = train(seed=seed)
        sigma = np.exp(s)
        assert abs(mu - 3.0) < 0.3, f"seed {seed}: expected mu near 3.0, got {mu:.2f}"
        assert 0.7 < sigma < 1.4, f"seed {seed}: expected sigma near 1.0, got {sigma:.2f}"
