import numpy as np

from submission import rmsnorm, grouped_rmsnorm

X = np.array([8.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
ONES = np.ones(8)


def test_rmsnorm_worked_example():
    got = rmsnorm(X, ONES, eps=0.0)
    want = X / np.sqrt(np.mean(X**2))
    assert np.allclose(got, want), f"expected {np.round(want, 4)}, got {np.round(got, 4)}"
    assert np.isclose(got[0], 2.744, atol=1e-3), f"expected x[0] -> 2.744, got {got[0]:.4f}"


def test_grouped_worked_example():
    got = grouped_rmsnorm(X, ONES, 2, eps=0.0)
    want = np.array([2.0, 0, 0, 0, 1, 1, 1, 1])
    assert np.allclose(got, want), f"expected {want}, got {np.round(got, 4)}"


def test_groups_are_contiguous():
    x = np.arange(1.0, 9.0)
    got = grouped_rmsnorm(x, ONES, 2)
    a, b = x[:4], x[4:]
    want = np.concatenate([a / np.sqrt(np.mean(a**2) + 1e-6), b / np.sqrt(np.mean(b**2) + 1e-6)])
    assert np.allclose(got, want), f"groups must be x[:4] and x[4:]; expected {np.round(want, 4)}, got {np.round(got, 4)}"


def test_k1_equals_rmsnorm():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 12))
    w = rng.normal(size=12)
    got, want = grouped_rmsnorm(x, w, 1), rmsnorm(x, w)
    assert np.allclose(got, want), f"k=1 must match rmsnorm; max diff {np.abs(got - want).max():.2e}"


def test_batched_shape_and_weight():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 5, 16))
    w = rng.normal(size=16)
    got = grouped_rmsnorm(x, w, 4)
    assert got.shape == x.shape, f"expected shape {x.shape}, got {got.shape}"
    g = x.reshape(2, 5, 4, 4)
    want = (g / np.sqrt(np.mean(g**2, axis=-1, keepdims=True) + 1e-6)).reshape(x.shape) * w
    assert np.allclose(got, want), f"max diff {np.abs(got - want).max():.2e}"


def test_bound_is_sqrt_group_size():
    e = np.zeros(16)
    e[3] = 5.0
    got = grouped_rmsnorm(e, np.ones(16), 4)
    assert np.isclose(got.max(), 2.0, atol=1e-4), f"one-hot with d_g=4 should reach sqrt(4)=2, got {got.max():.4f}"


def test_rejects_uneven_groups():
    try:
        grouped_rmsnorm(np.ones(10), np.ones(10), 3)
    except ValueError:
        return
    assert False, "expected ValueError for d=10, k=3"
