import numpy as np
from submission import annotations


def weights(seed, n=3, m=2):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n, m)), rng.normal(size=(n, n)) * 0.5, rng.normal(size=(n, m)), rng.normal(size=(n, n)) * 0.5]


def test_tiny_example_by_hand():
    # scripts/bidirectional_example.py: n = 1, w = 1, u = 0.5, x = [1, 0, -1].
    X = np.array([[1.0], [0.0], [-1.0]])
    W, U = np.array([[1.0]]), np.array([[0.5]])
    h = annotations(X, W, U, W, U)
    want = np.array([[0.7616, 0.6741], [0.3634, -0.3634], [-0.6741, -0.7616]])
    assert h.shape == (3, 2), f"expected shape (3, 2), got {h.shape}"
    assert np.allclose(h, want, atol=1e-4), f"expected\n{want}\ngot\n{h}"


def test_shape():
    X = np.random.default_rng(0).normal(size=(5, 2))
    h = annotations(X, *weights(1))
    assert h.shape == (5, 6), f"expected (5, 6), got {h.shape}"


def test_forward_half_ignores_the_future():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(6, 2))
    W = weights(3)
    j = 2
    X2 = X.copy()
    X2[j + 1:] += 1.0  # change only words after j
    a, b = annotations(X, *W), annotations(X2, *W)
    assert np.allclose(a[j, :3], b[j, :3]), f"the forward half at j={j} must not depend on later words\nbefore {a[j, :3]}\nafter  {b[j, :3]}"
    assert not np.allclose(a[j, 3:], b[j, 3:]), "the backward half at j should change when later words change"


def test_backward_half_ignores_the_past():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(6, 2))
    W = weights(5)
    j = 3
    X2 = X.copy()
    X2[:j] += 1.0  # change only words before j
    a, b = annotations(X, *W), annotations(X2, *W)
    assert np.allclose(a[j, 3:], b[j, 3:]), f"the backward half at j={j} must not depend on earlier words\nbefore {a[j, 3:]}\nafter  {b[j, 3:]}"
    assert not np.allclose(a[j, :3], b[j, :3]), "the forward half at j should change when earlier words change"


def run(X, W, U, order):
    """Reference RNN that visits positions in the given order and returns states by position."""
    out = np.zeros((X.shape[0], W.shape[0]))
    prev = np.zeros(W.shape[0])
    for t in order:
        prev = np.tanh(W @ X[t] + U @ prev)
        out[t] = prev
    return out


def test_each_direction_has_its_own_weights():
    X = np.random.default_rng(6).normal(size=(4, 2))
    Wf, Uf, Wb, Ub = weights(7)
    h = annotations(X, Wf, Uf, Wb, Ub)
    want_f = run(X, Wf, Uf, range(4))
    want_b = run(X, Wb, Ub, reversed(range(4)))
    assert np.allclose(h[:, :3], want_f), f"forward half must use Wf and Uf\nexpected {want_f}\ngot {h[:, :3]}"
    assert np.allclose(h[:, 3:], want_b), f"backward half must use Wb and Ub\nexpected {want_b}\ngot {h[:, 3:]}"


def test_first_backward_state_sees_only_the_last_word():
    X = np.random.default_rng(8).normal(size=(5, 2))
    Wf, Uf, Wb, Ub = weights(9)
    h = annotations(X, Wf, Uf, Wb, Ub)
    want = np.tanh(Wb @ X[-1])
    assert np.allclose(h[-1, 3:], want), f"backward state at the last position is tanh(Wb x_T)\nexpected {want}\ngot {h[-1, 3:]}"
    want_f = np.tanh(Wf @ X[0])
    assert np.allclose(h[0, :3], want_f), f"forward state at position 0 is tanh(Wf x_1)\nexpected {want_f}\ngot {h[0, :3]}"
