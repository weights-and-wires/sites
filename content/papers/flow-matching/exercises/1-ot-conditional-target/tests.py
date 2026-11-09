import numpy as np
from submission import cfm_loss, ot_pair


def test_page_example():
    xt, ut = ot_pair(np.array([[-1.0]]), np.array([[2.0]]), np.array([0.5]), 0.1)
    assert np.allclose(xt, [[0.45]]), f"expected x_t = 0.45, got {xt}"
    assert np.allclose(ut, [[2.9]]), f"expected target 2.9, got {ut}"


def test_endpoints():
    x0 = np.array([[1.0, -2.0]])
    x1 = np.array([[3.0, 0.5]])
    sm = 0.2
    xa, _ = ot_pair(x0, x1, np.array([0.0]), sm)
    xb, _ = ot_pair(x0, x1, np.array([1.0]), sm)
    assert np.allclose(xa, x0), f"at t = 0 expected the noise sample {x0}, got {xa}"
    want = x1 + sm * x0
    assert np.allclose(xb, want), f"at t = 1 expected x1 + sigma_min x0 = {want}, got {xb}"


def test_sigma_min_enters_the_target():
    x0 = np.array([[1.0, 2.0]])
    x1 = np.array([[4.0, 1.0]])
    _, u0 = ot_pair(x0, x1, np.array([0.3]), 0.0)
    _, u1 = ot_pair(x0, x1, np.array([0.3]), 0.5)
    assert np.allclose(u0, x1 - x0), f"with sigma_min = 0 expected {x1 - x0}, got {u0}"
    assert np.allclose(u1, x1 - 0.5 * x0), f"with sigma_min = 0.5 expected {x1 - 0.5 * x0}, got {u1}"


def test_straight_line_with_constant_velocity():
    rng = np.random.default_rng(0)
    x0, x1 = rng.normal(size=(1, 3)), rng.normal(size=(1, 3))
    sm = 0.1
    ts = np.linspace(0, 1, 6)
    pts = np.concatenate([ot_pair(x0, x1, np.array([t]), sm)[0] for t in ts])
    steps = np.diff(pts, axis=0)
    _, u = ot_pair(x0, x1, np.array([0.5]), sm)
    assert np.allclose(steps, u * 0.2), f"each step of 0.2 in time should move by 0.2 * {u}, got {steps}"


def test_batch_uses_each_row_time():
    x0 = np.array([[1.0], [1.0]])
    x1 = np.array([[2.0], [2.0]])
    xt, _ = ot_pair(x0, x1, np.array([0.0, 1.0]), 0.0)
    assert np.allclose(xt, [[1.0], [2.0]]), f"expected [[1], [2]] for t = [0, 1], got {xt}"


def test_loss_values():
    x0 = np.array([[-1.0, 0.0], [1.0, 1.0]])
    x1 = np.array([[2.0, 0.0], [0.0, 3.0]])
    t = np.array([0.5, 0.25])
    sm = 0.1
    _, ut = ot_pair(x0, x1, t, sm)
    zero = lambda x, s: np.zeros_like(x)
    want = np.mean(np.sum(ut**2, axis=1))
    got = cfm_loss(zero, x0, x1, t, sm)
    assert np.isclose(got, want), f"v = 0 should give mean ||u||^2 = {want}, got {got}"
    oracle = lambda x, s: ut
    got = cfm_loss(oracle, x0, x1, t, sm)
    assert np.isclose(got, 0.0, atol=1e-12), f"a field equal to the target should give 0, got {got}"


def test_loss_sums_over_dimensions_and_averages_over_batch():
    x0 = np.zeros((4, 2))
    x1 = np.ones((4, 2))
    t = np.full(4, 0.5)
    v = lambda x, s: np.zeros_like(x)
    got = cfm_loss(v, x0, x1, t, 0.0)
    assert np.isclose(got, 2.0), f"target is [1, 1] so ||u||^2 = 2 for every row; expected 2.0, got {got}"


def test_v_receives_the_path_point_and_time():
    seen = {}

    def v(x, s):
        seen["x"], seen["t"] = x, s
        return np.zeros_like(x)

    x0 = np.array([[-1.0]])
    x1 = np.array([[2.0]])
    cfm_loss(v, x0, x1, np.array([0.5]), 0.1)
    assert np.allclose(seen["x"], [[0.45]]), f"v should be evaluated at x_t = 0.45, got {seen['x']}"
    assert np.allclose(seen["t"], [0.5]), f"v should receive t = [0.5], got {seen['t']}"
