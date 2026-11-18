import numpy as np

from submission import silu, route, mova_values

X = np.array([[1.0, -1.0]])
W_V = np.array([
    [[1.0, 0.0], [0.0, 1.0]],
    [[0.0, 1.0], [1.0, 0.0]],
    [[2.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, -2.0]],
])
W_R = np.array([[1.0, 0.0, 2.0, 0.5], [0.0, 1.0, -1.0, 0.0]])


def test_silu():
    got = silu(np.array([-1.0, 0.0, 2.0]))
    want = np.array([-0.2689, 0.0, 1.7616])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {np.round(got, 4)}"


def test_route_worked_example():
    idx, p = route(X, W_R, 2)
    assert idx.tolist() == [[2, 0]], f"logits [1, -1, 3, 0.5]: expected [[2, 0]] best first, got {idx.tolist()}"
    assert np.allclose(p, [[0.8808, 0.1192]], atol=1e-4), f"expected [[0.8808, 0.1192]], got {np.round(p, 4).tolist()}"


def test_route_no_overflow():
    idx, p = route(np.array([[1000.0, 0.0]]), W_R, 2)
    assert np.all(np.isfinite(p)), f"huge logits overflowed: {p}"
    assert np.allclose(p.sum(axis=1), 1.0), f"weights must sum to 1, got {p.sum(axis=1)}"


def test_mova_worked_example():
    got = mova_values(X, W_R, W_V, 2)
    want = np.array([[1.6388, -0.0321]])
    assert np.allclose(got, want, atol=1e-4), f"expected {want.tolist()}, got {np.round(got, 4).tolist()}"


def test_silu_before_weighting():
    got = mova_values(X, W_R, W_V, 2)
    linear = 0.8808 * np.array([2.0, 0.0]) + 0.1192 * np.array([1.0, -1.0])
    assert not np.allclose(got, [linear], atol=1e-3), "output equals the linear mix; apply SiLU to each expert first"
    after = silu(linear)
    assert not np.allclose(got, [after], atol=1e-3), "SiLU goes on each expert output, not on the mixed sum"


def test_k_equals_one_and_batch():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 5))
    W_r = rng.normal(size=(5, 8))
    W_v = rng.normal(size=(8, 5, 3))
    got = mova_values(x, W_r, W_v, 1)
    best = np.argmax(x @ W_r, axis=1)
    want = np.stack([silu(x[t] @ W_v[best[t]]) for t in range(6)])
    assert got.shape == (6, 3), f"expected shape (6, 3), got {got.shape}"
    assert np.allclose(got, want), f"k=1 must return SiLU of the best expert's output; max diff {np.abs(got - want).max():.2e}"
