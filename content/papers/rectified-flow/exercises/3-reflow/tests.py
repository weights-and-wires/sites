import numpy as np
from submission import euler_path, kernel_velocity, reflow


def crossing_pairs():
    """Noise points on a line, data points that are a random permutation of them."""
    rng = np.random.default_rng(0)
    x0 = np.linspace(-2, 2, 41)[:, None]
    return x0, x0[rng.permutation(41)]


def cost(a, b):
    return float(((b - a) ** 2).sum(-1).mean())


def test_shape_and_start():
    x0 = np.array([[0.0, 1.0], [2.0, -1.0]])
    x1 = np.array([[1.0, 1.0], [0.0, 3.0]])
    path = euler_path(x0, x1, 0.5, 10)
    assert path.shape == (11, 2, 2), f"expected shape (11, 2, 2), got {path.shape}"
    assert np.allclose(path[0], x0), f"path[0] should be x0 {x0}, got {path[0]}"


def test_one_pair_follows_its_straight_line():
    x0 = np.array([[1.0, -1.0]])
    x1 = np.array([[3.0, 2.0]])
    path = euler_path(x0, x1, 0.3, 8)
    t = np.linspace(0.0, 1.0, 9)[:, None, None]
    want = (1 - t) * x0 + t * x1
    assert np.allclose(path, want), f"expected the line {want[:, 0].tolist()}, got {path[:, 0].tolist()}"


def test_final_step_is_finite():
    x0, x1 = crossing_pairs()
    path = euler_path(x0, x1, 0.1, 20)
    assert np.all(np.isfinite(path)), "a field evaluated at t = 1 divides by zero; use the time at the start of the step"


def test_inputs_are_not_modified():
    x0, x1 = crossing_pairs()
    a, b = x0.copy(), x1.copy()
    reflow(x0, x1, 0.1, 20)
    assert np.array_equal(x0, a) and np.array_equal(x1, b), "reflow must not change its inputs"


def test_reflow_keeps_the_noise_points_and_the_pairing():
    x0, x1 = crossing_pairs()
    z0, z1 = reflow(x0, x1, 0.1, 50)
    assert np.allclose(z0, x0), "the first element of the new pair must be x0"
    assert z1.shape == x1.shape, f"expected shape {x1.shape}, got {z1.shape}"
    ordering = np.corrcoef(z0[:, 0], z1[:, 0])[0, 1]
    assert ordering > 0.99, f"in 1-D the rectified coupling is monotone; correlation of z0 and z1 was {ordering:.3f}"


def test_reflow_lowers_the_transport_cost_of_a_crossing_coupling():
    x0, x1 = crossing_pairs()
    before = cost(x0, x1)
    z0, z1 = reflow(x0, x1, 0.1, 50)
    after = cost(z0, z1)
    assert after < 0.1 * before, f"Theorem 3.5: cost should fall from {before:.3f} to well under {0.1 * before:.3f}, got {after:.3f}"


def test_reflow_endpoints_stay_in_the_data_range():
    x0, x1 = crossing_pairs()
    _, z1 = reflow(x0, x1, 0.1, 50)
    assert z1.min() > x1.min() - 0.3 and z1.max() < x1.max() + 0.3, f"endpoints in [{z1.min():.2f}, {z1.max():.2f}] left the data range [{x1.min():.2f}, {x1.max():.2f}]"


def test_straight_coupling_is_a_fixed_point():
    # x1 = 2 x0 + 1 is monotone, so the lines never cross (Theorem 3.6).
    x0 = np.linspace(-2, 2, 41)[:, None]
    x1 = 2 * x0 + 1
    _, z1 = reflow(x0, x1, 0.03, 50)
    err = float(np.abs(z1 - x1).max())
    assert err < 1e-3, f"a straight coupling should be unchanged; largest change {err:.2e}"


def test_two_dimensional_crossing_pairs_get_cheaper():
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(60, 2))
    x1 = rng.normal(size=(60, 2)) + np.array([4.0, 0.0])
    before = cost(x0, x1)
    z0, z1 = reflow(x0, x1, 0.4, 50)
    after = cost(z0, z1)
    assert after < before, f"Theorem 3.5: cost should fall below {before:.3f}, got {after:.3f}"
