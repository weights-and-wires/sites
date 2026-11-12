import numpy as np
from submission import OUNoise


def test_shape_and_start():
    n = OUNoise(3, mu=0.5, rng=np.random.default_rng(0))
    assert np.allclose(n.x, 0.5), f"the state must start at mu = 0.5, got {n.x}"
    s = n.sample()
    assert s.shape == (3,), f"expected shape (3,), got {s.shape}"


def test_noise_free_decay_toward_mu():
    n = OUNoise(2, theta=0.15, sigma=0.0, mu=0.0, rng=np.random.default_rng(0))
    n.x = np.array([1.0, -2.0])
    got = [float(n.sample()[0]) for _ in range(3)]
    want = [0.85, 0.7225, 0.614125]
    assert np.allclose(got, want), f"with sigma = 0 the state must shrink by 1 - theta each step: expected {want}, got {got}"
    n.x = np.array([5.0])
    n.mu = 5.0
    assert np.allclose(n.sample(), 5.0), "a state at mu with sigma = 0 must stay at mu"


def test_pull_goes_toward_mu_not_away():
    n = OUNoise(1, theta=0.5, sigma=0.0, mu=2.0, rng=np.random.default_rng(0))
    n.x = np.array([0.0])
    s = n.sample()
    assert np.allclose(s, 1.0), f"expected 0 + 0.5 * (2 - 0) = 1.0, got {s}"


def test_noise_term_scales_with_sqrt_dt():
    rng = np.random.default_rng(1)
    n = OUNoise(20000, theta=0.0, sigma=0.2, dt=0.01, rng=rng)
    s = n.sample()
    want = 0.2 * np.sqrt(0.01)
    assert abs(s.std() - want) < 0.05 * want, f"one step from rest must have std sigma * sqrt(dt) = {want}, got {s.std():.5f}"


def test_reset_returns_to_mu():
    n = OUNoise(4, mu=1.0, rng=np.random.default_rng(0))
    for _ in range(10):
        n.sample()
    n.reset()
    assert np.allclose(n.x, 1.0), f"reset must put the state back at mu = 1.0, got {n.x}"


def test_sample_returns_a_copy():
    n = OUNoise(2, theta=0.5, sigma=0.0, mu=0.0, rng=np.random.default_rng(0))
    n.x = np.array([1.0, 1.0])
    first = n.sample()
    first[:] = 100.0
    second = n.sample()
    assert np.allclose(second, 0.25), f"changing a returned sample must not change the process state: expected 0.25, got {second}"


def test_pull_scales_with_dt():
    n = OUNoise(1, theta=0.2, sigma=0.0, mu=0.0, dt=0.5, rng=np.random.default_rng(0))
    n.x = np.array([1.0])
    s = n.sample()
    assert np.allclose(s, 0.9), f"expected 1 + 0.2 * (0 - 1) * 0.5 = 0.9, got {s}"


def test_stationary_std_and_lag_one_correlation():
    n = OUNoise(2000, theta=0.15, sigma=0.2, rng=np.random.default_rng(3))
    for _ in range(120):
        prev = n.sample()
    cur = n.sample()
    want_std = 0.2 / np.sqrt(1 - 0.85**2)
    assert abs(cur.std() - want_std) < 0.1 * want_std, f"expected stationary std about {want_std:.3f}, got {cur.std():.3f}"
    corr = np.corrcoef(prev, cur)[0, 1]
    assert abs(corr - 0.85) < 0.05, f"expected lag-1 correlation about 0.85, got {corr:.3f}"


def test_reproducible_with_a_seeded_rng():
    a = OUNoise(3, rng=np.random.default_rng(5))
    b = OUNoise(3, rng=np.random.default_rng(5))
    assert np.allclose(a.sample(), b.sample()), "two processes with equal seeds must produce equal samples"
