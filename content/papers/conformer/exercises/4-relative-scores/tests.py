import numpy as np
from submission import rel_scores


def naive(q, k, p, u, v):
    T, dk = q.shape
    A = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            r = i - j
            row = (T - 1) - r
            A[i, j] = (np.dot(q[i] + u, k[j]) + np.dot(q[i] + v, p[row])) / np.sqrt(dk)
    return A


def test_hand_worked_example():
    q = np.array([[1.0], [2.0]])
    k = np.array([[3.0], [4.0]])
    p = np.array([[10.0], [20.0], [30.0]])
    got = rel_scores(q, k, p, np.array([1.0]), np.array([2.0]))
    want = [[66.0, 98.0], [49.0, 92.0]]
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_matches_naive_loop():
    rng = np.random.default_rng(0)
    T, dk = 6, 4
    q, k = rng.normal(size=(T, dk)), rng.normal(size=(T, dk))
    p = rng.normal(size=(2 * T - 1, dk))
    u, v = rng.normal(size=dk), rng.normal(size=dk)
    got = rel_scores(q, k, p, u, v)
    want = naive(q, k, p, u, v)
    assert got.shape == (T, T), f"expected shape {(T, T)}, got {got.shape}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_position_term_depends_only_on_offset():
    rng = np.random.default_rng(1)
    T, dk = 5, 3
    q = np.tile(rng.normal(size=(1, dk)), (T, 1))
    k = np.zeros((T, dk))
    p = rng.normal(size=(2 * T - 1, dk))
    zero = np.zeros(dk)
    got = rel_scores(q, k, p, zero, zero)
    for i in range(T - 1):
        for j in range(T - 1):
            assert abs(got[i, j] - got[i + 1, j + 1]) < 1e-12, f"expected A[{i},{j}] = A[{i + 1},{j + 1}], got {got[i, j]} and {got[i + 1, j + 1]}"


def test_u_and_v_terms_are_present():
    rng = np.random.default_rng(2)
    T, dk = 4, 3
    q, k = rng.normal(size=(T, dk)), rng.normal(size=(T, dk))
    p = rng.normal(size=(2 * T - 1, dk))
    u, v = rng.normal(size=dk), rng.normal(size=dk)
    full = rel_scores(q, k, p, u, v)
    no_u = rel_scores(q, k, p, np.zeros(dk), v)
    no_v = rel_scores(q, k, p, u, np.zeros(dk))
    assert not np.allclose(full, no_u), "expected the u term to change the scores"
    assert not np.allclose(full, no_v), "expected the v term to change the scores"
    assert np.allclose(full - no_u, naive(q, k, p, u, v) - naive(q, k, p, np.zeros(dk), v)), "expected u to add (u . k_j) / sqrt(dk)"


def test_scaling_uses_sqrt_of_dk():
    rng = np.random.default_rng(3)
    T, dk = 3, 16
    q, k = rng.normal(size=(T, dk)), rng.normal(size=(T, dk))
    p = rng.normal(size=(2 * T - 1, dk))
    u, v = rng.normal(size=dk), rng.normal(size=dk)
    got = rel_scores(q, k, p, u, v)
    want = naive(q, k, p, u, v)
    assert np.allclose(got, want), f"expected {want}, got {got}"
