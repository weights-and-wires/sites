import numpy as np
from submission import dsm_loss, int_beta


def kernel(t):
    ib = int_beta(np.asarray(t, dtype=float))[:, None]
    return np.exp(-0.5 * ib), 1.0 - np.exp(-ib)


def test_exact_score_gives_zero_loss():
    rng = np.random.default_rng(0)
    c = np.array([1.5, -0.5])
    x0 = np.tile(c, (6, 1))  # a point mass, so the exact score of p_t is known
    eps = rng.standard_normal((6, 2))
    t = rng.uniform(0.05, 1.0, size=6)

    def exact(x_t, t):
        m, var = kernel(t)
        return -(x_t - m * c) / var

    loss = dsm_loss(exact, x0, t, eps)
    assert abs(loss) < 1e-12, f"the exact score of a point mass must give loss 0, got {loss}"


def test_zero_score_value():
    # With s = 0 the loss is sigma^2 * ||eps / sigma||^2 = ||eps||^2, averaged over the batch.
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((8, 3))
    eps = rng.standard_normal((8, 3))
    t = np.linspace(0.1, 0.9, 8)
    got = dsm_loss(lambda x, t: np.zeros_like(x), x0, t, eps)
    want = float(np.mean(np.sum(eps**2, axis=1)))
    assert np.isclose(got, want), f"expected lambda = sigma^2 to give {want:.4f}, got {got:.4f}"


def test_matches_epsilon_prediction_loss():
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((10, 4))
    eps = rng.standard_normal((10, 4))
    t = rng.uniform(0.05, 1.0, size=10)

    def eps_net(x_t, t):
        return 0.3 * x_t + 0.1

    def score_from_eps(x_t, t):
        _, var = kernel(t)
        return -eps_net(x_t, t) / np.sqrt(var)

    m, var = kernel(t)
    x_t = m * x0 + np.sqrt(var) * eps
    want = float(np.mean(np.sum((eps_net(x_t, t) - eps) ** 2, axis=1)))
    got = dsm_loss(score_from_eps, x0, t, eps)
    assert np.isclose(got, want), f"expected the DDPM-style loss {want:.4f}, got {got:.4f}"


def test_weight_is_applied_per_row():
    x0 = np.zeros((2, 1))
    eps = np.array([[1.0], [1.0]])
    t = np.array([0.1, 1.0])
    _, var = kernel(t)
    got = dsm_loss(lambda x, t: np.zeros_like(x), x0, t, eps)
    want = float(np.mean(var[:, 0] * (1.0 / var[:, 0])))  # = 1 for each row
    assert np.isclose(got, want), f"expected {want}, got {got}"
    got2 = dsm_loss(lambda x, t: np.ones_like(x), x0, t, eps)
    target = -eps[:, 0] / np.sqrt(var[:, 0])
    want2 = float(np.mean(var[:, 0] * (1.0 - target) ** 2))
    assert np.isclose(got2, want2), f"expected {want2:.4f} for a constant score of 1, got {got2:.4f}"


def test_model_receives_noisy_input_and_time():
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal((5, 2)) + 10.0
    eps = rng.standard_normal((5, 2))
    t = np.full(5, 0.5)
    seen = {}

    def spy(x_t, tt):
        seen["x"], seen["t"] = x_t.copy(), np.asarray(tt).copy()
        return np.zeros_like(x_t)

    dsm_loss(spy, x0, t, eps)
    m, var = kernel(t)
    want = m * x0 + np.sqrt(var) * eps
    assert np.allclose(seen["x"], want), f"the model must get x(t) = m x0 + sigma eps, got {seen['x'][0]} for the first row, expected {want[0]}"
    assert np.allclose(seen["t"], t), f"the model must get the times {t}, got {seen['t']}"


def test_sum_over_dimensions_not_mean():
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal((4, 6))
    eps = rng.standard_normal((4, 6))
    t = np.full(4, 0.7)
    got = dsm_loss(lambda x, t: np.zeros_like(x), x0, t, eps)
    want = float(np.mean(np.sum(eps**2, axis=1)))
    assert np.isclose(got, want), f"expected the norm to sum over 6 dimensions: {want:.4f}, got {got:.4f}"
