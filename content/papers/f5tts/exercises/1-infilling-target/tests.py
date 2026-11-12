import numpy as np
from submission import make_infill_inputs, masked_cfm_loss

X1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])[:, None]
X0 = np.array([0.5, -1.0, 0.0, 1.0, -0.5, 2.0])[:, None]
M = np.array([0, 0, 1, 1, 1, 0])


def test_page_example_inputs():
    x_t, cond, target = make_infill_inputs(X1, X0, 0.5, M)
    want_xt = np.array([0.75, 0.5, 1.5, 2.5, 2.25, 4.0])[:, None]
    want_cond = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 6.0])[:, None]
    want_target = np.array([0.5, 3.0, 3.0, 3.0, 5.5, 4.0])[:, None]
    assert np.allclose(x_t, want_xt), f"expected x_t {want_xt.ravel()}, got {np.asarray(x_t).ravel()}"
    assert np.allclose(cond, want_cond), f"expected cond {want_cond.ravel()}, got {np.asarray(cond).ravel()}"
    assert np.allclose(target, want_target), f"expected target {want_target.ravel()}, got {np.asarray(target).ravel()}"


def test_path_endpoints():
    x_t0, _, _ = make_infill_inputs(X1, X0, 0.0, M)
    x_t1, _, _ = make_infill_inputs(X1, X0, 1.0, M)
    assert np.allclose(x_t0, X0), f"at t = 0 expected the noise {X0.ravel()}, got {np.asarray(x_t0).ravel()}"
    assert np.allclose(x_t1, X1), f"at t = 1 expected the data {X1.ravel()}, got {np.asarray(x_t1).ravel()}"


def test_target_is_the_path_velocity():
    t, eps = 0.3, 1e-4
    a, _, target = make_infill_inputs(X1, X0, t, M)
    b, _, _ = make_infill_inputs(X1, X0, t + eps, M)
    fd = (b - a) / eps
    assert np.allclose(fd, target, atol=1e-6), f"expected d x_t / dt = {np.asarray(fd).ravel()}, got target {np.asarray(target).ravel()}"


def test_multichannel_shapes_and_mask_broadcast():
    rng = np.random.default_rng(1)
    x1, x0 = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    m = np.array([1, 0, 0, 1, 1])
    x_t, cond, target = make_infill_inputs(x1, x0, 0.7, m)
    assert cond.shape == (5, 3), f"expected cond shape (5, 3), got {cond.shape}"
    assert np.allclose(cond[[1, 2]], x1[[1, 2]]), "context frames 1 and 2 must keep their data"
    assert np.allclose(cond[[0, 3, 4]], 0.0), f"masked frames must be zero, got {cond[[0, 3, 4]]}"
    assert np.allclose(target, x1 - x0), "target must be x1 - x0 on every frame"


def test_inputs_not_modified():
    x1, x0, m = X1.copy(), X0.copy(), M.copy()
    make_infill_inputs(x1, x0, 0.5, m)
    assert np.array_equal(x1, X1) and np.array_equal(x0, X0) and np.array_equal(m, M), "inputs were modified"


def test_loss_ignores_context_frames():
    _, _, target = make_infill_inputs(X1, X0, 0.5, M)
    pred = target.copy()
    pred[M == 0] += 3.0
    got = masked_cfm_loss(pred, target, M)
    assert abs(got) < 1e-12, f"expected 0 (errors only on context frames), got {got}"


def test_loss_value_and_normalization():
    _, _, target = make_infill_inputs(X1, X0, 0.5, M)
    got = masked_cfm_loss(np.zeros_like(target), target, M)
    want = (3.0**2 + 3.0**2 + 5.5**2) / 3
    assert abs(got - want) < 1e-9, f"expected {want:.6f} (mean over the 3 masked frames), got {got}"
    pred2 = np.zeros((6, 2))
    tgt2 = np.ones((6, 2))
    got2 = masked_cfm_loss(pred2, tgt2, M)
    assert abs(got2 - 1.0) < 1e-9, f"expected 1.0 (divide by masked frames times channels), got {got2}"


def test_loss_is_zero_for_the_exact_target():
    _, _, target = make_infill_inputs(X1, X0, 0.5, M)
    got = masked_cfm_loss(target, target, M)
    assert abs(got) < 1e-12, f"expected 0, got {got}"
