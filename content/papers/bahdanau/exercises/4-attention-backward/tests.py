import numpy as np
from submission import attention_backward


def softmax(E):
    e = np.exp(E - E.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def loss(E, H, G):
    return float(np.sum((softmax(E) @ H) * G))


def numeric_grad(f, X, eps=1e-6):
    g = np.zeros_like(X)
    for idx in np.ndindex(*X.shape):
        Xp, Xm = X.copy(), X.copy()
        Xp[idx] += eps
        Xm[idx] -= eps
        g[idx] = (f(Xp) - f(Xm)) / (2 * eps)
    return g


def test_tiny_example_by_hand():
    # scripts/gradient_example.py
    E = np.array([[1.0, 0.0, -1.0]])
    H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    G = np.array([[1.0, 0.0]])
    dE, _ = attention_backward(softmax(E), H, G)
    want = [[0.1628, -0.1848, 0.0220]]
    assert np.allclose(dE, want, atol=1e-4), f"expected {want}, got {dE}"


def test_scores_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    E, H, G = rng.normal(size=(3, 5)), rng.normal(size=(5, 4)), rng.normal(size=(3, 4))
    dE, _ = attention_backward(softmax(E), H, G)
    want = numeric_grad(lambda X: loss(X, H, G), E)
    assert np.allclose(dE, want, atol=1e-5), f"dE differs from finite differences\nexpected {want[0]}\ngot {dE[0]}"


def test_annotation_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    E, H, G = rng.normal(size=(3, 5)), rng.normal(size=(5, 4)), rng.normal(size=(3, 4))
    _, dH = attention_backward(softmax(E), H, G)
    want = numeric_grad(lambda X: loss(E, X, G), H)
    assert dH.shape == H.shape, f"dH shape: expected {H.shape}, got {dH.shape}"
    assert np.allclose(dH, want, atol=1e-5), f"dH differs from finite differences\nexpected {want[0]}\ngot {dH[0]}"


def test_score_gradient_rows_sum_to_zero():
    rng = np.random.default_rng(2)
    E, H, G = rng.normal(size=(4, 6)), rng.normal(size=(6, 3)), rng.normal(size=(4, 3))
    dE, _ = attention_backward(softmax(E), H, G)
    assert np.allclose(dE.sum(axis=1), 0, atol=1e-10), f"shifting a row of scores changes nothing, so each row of dE must sum to 0, got {dE.sum(axis=1)}"


def test_non_square_shapes():
    rng = np.random.default_rng(3)
    E, H, G = rng.normal(size=(2, 7)), rng.normal(size=(7, 6)), rng.normal(size=(2, 6))
    dE, dH = attention_backward(softmax(E), H, G)
    assert dE.shape == (2, 7), f"dE shape: expected (2, 7), got {dE.shape}"
    assert dH.shape == (7, 6), f"dH shape: expected (7, 6), got {dH.shape}"
