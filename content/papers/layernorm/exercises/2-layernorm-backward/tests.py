import numpy as np
from submission import layer_norm_forward, layer_norm_backward

N, H = 4, 6


def setup(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, H)) * 2 + 1
    g, b = rng.normal(size=H), rng.normal(size=H)
    w = rng.normal(size=(N, H))  # loss = sum(w * y), so dy = w
    return x, g, b, w


def numeric_grad(f, p, h=1e-6):
    grad = np.zeros_like(p)
    for i in np.ndindex(*p.shape):
        old = p[i]
        p[i] = old + h
        hi = f()
        p[i] = old - h
        lo = f()
        p[i] = old
        grad[i] = (hi - lo) / (2 * h)
    return grad


def test_shapes():
    x, g, b, w = setup()
    _, cache = layer_norm_forward(x, g, b)
    dx, dg, db = layer_norm_backward(w, cache)
    assert dx.shape == (N, H), f"dx expected shape {(N, H)}, got {dx.shape}"
    assert dg.shape == (H,), f"dgain expected shape {(H,)}, got {dg.shape}"
    assert db.shape == (H,), f"dbias expected shape {(H,)}, got {db.shape}"


def test_dx_matches_finite_differences():
    x, g, b, w = setup()
    _, cache = layer_norm_forward(x, g, b)
    dx, _, _ = layer_norm_backward(w, cache)
    want = numeric_grad(lambda: float((w * layer_norm_forward(x, g, b)[0]).sum()), x)
    assert np.allclose(dx, want, atol=1e-6), f"max error {np.abs(dx - want).max():.2e}; expected\n{want}\ngot\n{dx}"


def test_dgain_matches_finite_differences():
    x, g, b, w = setup(1)
    _, cache = layer_norm_forward(x, g, b)
    _, dg, _ = layer_norm_backward(w, cache)
    want = numeric_grad(lambda: float((w * layer_norm_forward(x, g, b)[0]).sum()), g)
    assert np.allclose(dg, want, atol=1e-6), f"expected {want}, got {dg}"


def test_dbias_matches_finite_differences():
    x, g, b, w = setup(2)
    _, cache = layer_norm_forward(x, g, b)
    _, _, db = layer_norm_backward(w, cache)
    want = numeric_grad(lambda: float((w * layer_norm_forward(x, g, b)[0]).sum()), b)
    assert np.allclose(db, want, atol=1e-6), f"expected {want}, got {db}"


def test_dx_rows_sum_to_zero():
    x, g, b, w = setup(3)
    _, cache = layer_norm_forward(x, g, b)
    dx, _, _ = layer_norm_backward(w, cache)
    assert np.allclose(dx.sum(axis=-1), 0, atol=1e-10), f"each row of dx must sum to 0, got {dx.sum(axis=-1)}"


def test_dx_rows_are_orthogonal_to_xhat():
    x, g, b, w = setup(4)
    _, cache = layer_norm_forward(x, g, b, eps=0.0)  # exact only when eps = 0
    dx, _, _ = layer_norm_backward(w, cache)
    dots = (dx * cache[0]).sum(axis=-1)
    assert np.allclose(dots, 0, atol=1e-10), f"each row of dx must be orthogonal to xhat, got {dots}"


def test_dx_is_unchanged_by_rescaling_the_input():
    # The output is invariant to x -> 10 x, so the gradient with respect to x
    # must shrink by exactly 10.
    x, g, b, w = setup(5)
    _, c1 = layer_norm_forward(x, g, b, eps=0.0)
    _, c2 = layer_norm_forward(10 * x, g, b, eps=0.0)
    dx1, _, _ = layer_norm_backward(w, c1)
    dx2, _, _ = layer_norm_backward(w, c2)
    assert np.allclose(dx2, dx1 / 10, atol=1e-10), f"expected dx / 10 = {dx1 / 10}, got {dx2}"


def test_gradient_of_a_nonlinear_loss():
    x, g, b, w = setup(6)

    def loss():
        y, _ = layer_norm_forward(x, g, b)
        return float((np.tanh(y) * w).sum())

    y, cache = layer_norm_forward(x, g, b)
    dy = (1 - np.tanh(y) ** 2) * w
    dx, dg, db = layer_norm_backward(dy, cache)
    for name, got, p in (("dx", dx, x), ("dgain", dg, g), ("dbias", db, b)):
        want = numeric_grad(loss, p)
        assert np.allclose(got, want, atol=1e-6), f"{name} expected\n{want}\ngot\n{got}"
