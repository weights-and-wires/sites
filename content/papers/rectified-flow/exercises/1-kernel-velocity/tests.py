import numpy as np
from submission import kernel_velocity


def reference(z, t, x0, x1, h):
    """Slow loop version of Equation 5."""
    out = np.zeros_like(z, dtype=float)
    for j in range(len(z)):
        logw = np.array(
            [-np.sum((t * b + (1 - t) * a - z[j]) ** 2) / (2 * h * h) for a, b in zip(x0, x1)]
        )
        w = np.exp(logw - logw.max())
        w /= w.sum()
        out[j] = sum(wi * (b - z[j]) for wi, b in zip(w, x1)) / (1 - t)
    return out


def test_shape():
    z = np.zeros((5, 3))
    x0 = np.zeros((7, 3))
    x1 = np.ones((7, 3))
    got = kernel_velocity(z, 0.3, x0, x1, 0.5)
    assert got.shape == (5, 3), f"expected shape (5, 3), got {got.shape}"


def test_one_pair_on_its_line_gives_the_line_direction():
    x0 = np.array([[1.0, -1.0]])
    x1 = np.array([[3.0, 2.0]])
    t = 0.4
    z = t * x1 + (1 - t) * x0
    got = kernel_velocity(z, t, x0, x1, 0.1)
    assert np.allclose(got, x1 - x0), f"expected {x1 - x0}, got {got}"


def test_one_pair_off_its_line_points_at_the_endpoint():
    # (x1 - z) / (1 - t) = (4 - 1) / 0.5 = 6, not x1 - x0 = 4.
    got = kernel_velocity(np.array([[1.0]]), 0.5, np.array([[0.0]]), np.array([[4.0]]), 0.3)
    assert np.allclose(got, 6.0), f"expected [[6.0]], got {got}"


def test_two_pairs_with_equal_weights():
    # Kernel centers at t = 0.5 are 1 and 3, so z = 2 weighs them equally:
    # v = ((2 + 6) / 2 - 2) / 0.5 = 4.
    x0 = np.array([[0.0], [0.0]])
    x1 = np.array([[2.0], [6.0]])
    got = kernel_velocity(np.array([[2.0]]), 0.5, x0, x1, 0.5)
    assert np.allclose(got, 4.0), f"expected [[4.0]], got {got}"


def test_kernel_is_centered_at_the_interpolation_point():
    # Pairs (-1 -> 1) and (3 -> 3) have centers 0 and 3 at t = 0.5. At z = 0 the
    # first pair dominates, so v is close to 1 - (-1) = 2.
    x0 = np.array([[-1.0], [3.0]])
    x1 = np.array([[1.0], [3.0]])
    got = kernel_velocity(np.array([[0.0]]), 0.5, x0, x1, 0.3)
    assert np.allclose(got, 2.0, atol=1e-3), f"expected about [[2.0]], got {got}"


def test_matches_the_loop_reference():
    rng = np.random.default_rng(3)
    x0 = rng.normal(size=(12, 2))
    x1 = rng.normal(size=(12, 2)) + 2.0
    z = rng.normal(size=(6, 2))
    for t, h in [(0.0, 0.7), (0.35, 0.4), (0.8, 1.3)]:
        want = reference(z, t, x0, x1, h)
        got = kernel_velocity(z, t, x0, x1, h)
        assert np.allclose(got, want), f"t={t}, h={h}: expected {want}, got {got}"


def test_bandwidth_is_a_standard_deviation():
    # Centers 0 and 1, z = 0.25, h = 1: the weights are proportional to exp(-d^2 / 2)
    # with d = 0.25 and 0.75, so the second pair gets 1 / (1 + exp((0.75^2 - 0.25^2) / 2)).
    x0 = np.array([[0.0], [1.0]])
    x1 = np.array([[0.0], [1.0]])
    got = kernel_velocity(np.array([[0.25]]), 0.5, x0, x1, 1.0)
    w2 = 1.0 / (1.0 + np.exp((0.75**2 - 0.25**2) / 2))
    want = (w2 * 1.0 - 0.25) / 0.5
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_weights_sum_to_one():
    # If every pair ends at c, the estimate is (c - z) / (1 - t) whatever the weights are.
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(9, 2))
    c = np.array([2.0, -1.0])
    x1 = np.tile(c, (9, 1))
    z = rng.normal(size=(4, 2))
    got = kernel_velocity(z, 0.6, x0, x1, 0.8)
    want = (c - z) / 0.4
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_far_points_and_small_bandwidth_stay_finite():
    x0 = np.array([[0.0], [1.0]])
    x1 = np.array([[1.0], [2.0]])
    got = kernel_velocity(np.array([[500.0]]), 0.5, x0, x1, 0.01)
    assert np.all(np.isfinite(got)), f"expected a finite value, got {got}"
    # Kernel centers are 0.5 and 1.5, so the pair ending at 2 is the nearest: (2 - 500) / 0.5.
    assert np.allclose(got, (2.0 - 500.0) / 0.5), f"expected {(2.0 - 500.0) / 0.5}, got {got}"
