import numpy as np
from submission import make_schedule, q_sample


def test_hand_schedule():
    betas, alphas, ab = make_schedule(4, 0.1, 0.4)
    assert np.allclose(betas, [0.1, 0.2, 0.3, 0.4]), f"expected betas [0.1, 0.2, 0.3, 0.4], got {betas}"
    assert np.allclose(alphas, [0.9, 0.8, 0.7, 0.6]), f"expected alphas [0.9, 0.8, 0.7, 0.6], got {alphas}"
    assert np.allclose(ab, [0.9, 0.72, 0.504, 0.3024]), f"expected alpha_bars [0.9, 0.72, 0.504, 0.3024], got {ab}"


def test_paper_schedule():
    betas, alphas, ab = make_schedule(1000)
    assert betas.shape == (1000,), f"expected shape (1000,), got {betas.shape}"
    assert np.isclose(betas[0], 1e-4) and np.isclose(betas[-1], 0.02), f"expected betas from 1e-4 to 0.02, got {betas[0]} to {betas[-1]}"
    assert np.allclose(np.diff(betas), (0.02 - 1e-4) / 999), "betas must increase linearly"
    assert np.isclose(ab[-1], 4.0e-5, rtol=0.05), f"expected alpha_bar_T near 4.0e-5, got {ab[-1]}"
    assert np.all(np.diff(ab) < 0), "alpha_bars must decrease at every step"


def test_noise_free_limit():
    _, _, ab = make_schedule(4, 0.1, 0.4)
    x0 = np.array([[2.0, -1.0]])
    got = q_sample(x0, 3, np.zeros_like(x0), ab)
    want = np.sqrt(0.504) * x0
    assert np.allclose(got, want), f"with eps = 0 expected sqrt(alpha_bar_3) * x0 = {want}, got {got}"


def test_pure_noise_limit():
    _, _, ab = make_schedule(4, 0.1, 0.4)
    x0 = np.zeros((1, 2))
    eps = np.array([[1.0, -2.0]])
    got = q_sample(x0, 2, eps, ab)
    want = np.sqrt(1 - 0.72) * eps
    assert np.allclose(got, want), f"with x0 = 0 expected sqrt(1 - alpha_bar_2) * eps = {want}, got {got}"


def test_timestep_is_one_based():
    _, _, ab = make_schedule(4, 0.1, 0.4)
    x0 = np.ones((1, 1))
    got = q_sample(x0, 1, np.zeros((1, 1)), ab)
    assert np.allclose(got, np.sqrt(0.9)), f"t = 1 must use alpha_bars[0] = 0.9, expected {np.sqrt(0.9):.4f}, got {got}"
    got = q_sample(x0, 4, np.zeros((1, 1)), ab)
    assert np.allclose(got, np.sqrt(0.3024)), f"t = 4 must use alpha_bars[3] = 0.3024, expected {np.sqrt(0.3024):.4f}, got {got}"


def test_per_row_timesteps():
    _, _, ab = make_schedule(4, 0.1, 0.4)
    x0 = np.ones((3, 2))
    got = q_sample(x0, np.array([1, 2, 4]), np.zeros((3, 2)), ab)
    want = np.sqrt(np.array([0.9, 0.72, 0.3024]))[:, None] * x0
    assert np.allclose(got, want), f"each row needs its own alpha_bar; expected\n{want}\ngot\n{got}"


def test_matches_stepwise_noising():
    betas, alphas, ab = make_schedule(4, 0.1, 0.4)
    rng = np.random.default_rng(0)
    n, x0 = 200_000, 2.0
    x = np.full(n, x0)
    for i in range(3):
        x = np.sqrt(alphas[i]) * x + np.sqrt(betas[i]) * rng.standard_normal(n)
    y = q_sample(np.full((n, 1), x0), 3, rng.standard_normal((n, 1)), ab)[:, 0]
    assert abs(x.mean() - y.mean()) < 0.02, f"means differ: stepwise {x.mean():.4f}, closed form {y.mean():.4f}"
    assert abs(x.var() - y.var()) < 0.02, f"variances differ: stepwise {x.var():.4f}, closed form {y.var():.4f}"
    assert abs(y.var() - 0.496) < 0.02, f"expected variance 1 - alpha_bar_3 = 0.496, got {y.var():.4f}"
