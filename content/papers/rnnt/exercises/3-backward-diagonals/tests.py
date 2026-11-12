import numpy as np
from submission import backward, diagonal_sums

EMIT = np.array([[0.6, 1 / 3], [0.2, 0.75], [0.25, 0.4]])
NULL = np.array([[0.2, 1 / 6, 0.5], [0.2, 0.125, 0.5], [0.5, 0.4, 0.8]])
PROB = 0.1135 * 0.8


def ref_forward(emit, null):
    T, U1 = null.shape
    a = np.zeros((T, U1))
    a[0, 0] = 1.0
    for t in range(T):
        for u in range(U1):
            if t == 0 and u == 0:
                continue
            a[t, u] = (a[t - 1, u] * null[t - 1, u] if t else 0.0) + (a[t, u - 1] * emit[t, u - 1] if u else 0.0)
    return a, a[-1, -1] * null[-1, -1]


def test_beta_table_from_the_page():
    beta = backward(EMIT, NULL)
    want = np.array([[0.0908, 0.1233333333, 0.2], [0.084, 0.34, 0.4], [0.08, 0.32, 0.8]])
    assert np.allclose(beta, want, atol=1e-9), f"expected beta\n{want}\ngot\n{beta}"


def test_corner_is_the_final_null():
    beta = backward(EMIT, NULL)
    assert abs(beta[2, 2] - 0.8) < 1e-12, f"expected beta(T, U) = null(T, U) = 0.8, got {beta[2, 2]}"
    assert abs(beta[2, 1] - 0.32) < 1e-12, f"expected beta(3, 1) = y(3, 1) * beta(3, 2) = 0.4 * 0.8 = 0.32, got {beta[2, 1]}"
    assert abs(beta[2, 0] - 0.08) < 1e-12, f"expected beta(3, 0) = 0.25 * 0.32 = 0.08, got {beta[2, 0]}"


def test_beta_at_the_origin_is_the_probability():
    beta = backward(EMIT, NULL)
    assert abs(beta[0, 0] - PROB) < 1e-9, f"expected beta(1, 0) = Pr(y|x) = {PROB}, got {beta[0, 0]}"


def test_beta_matches_a_random_lattice():
    rng = np.random.default_rng(5)
    emit = rng.uniform(0.05, 0.6, size=(4, 3))
    null = rng.uniform(0.05, 0.6, size=(4, 4))
    _, prob = ref_forward(emit, null)
    beta = backward(emit, null)
    assert abs(beta[0, 0] - prob) < 1e-12, f"expected beta(1, 0) = {prob}, got {beta[0, 0]}"
    # beta(t, u) must also satisfy Equation 18 at an interior node.
    want = beta[2, 1] * null[1, 1] + beta[1, 2] * emit[1, 1]
    assert abs(beta[1, 1] - want) < 1e-12, f"expected beta(2, 1) = {want} from Equation 18, got {beta[1, 1]}"


def test_diagonal_sums_on_known_arrays():
    alpha = np.arange(1, 7, dtype=float).reshape(2, 3)  # [[1, 2, 3], [4, 5, 6]]
    beta = np.ones((2, 3))
    got = diagonal_sums(alpha, beta)
    want = np.array([1.0, 2 + 4, 3 + 5, 6])
    assert got.shape == (4,), f"expected T + U = 4 diagonals, got shape {got.shape}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_every_diagonal_equals_the_probability():
    alpha, prob = ref_forward(EMIT, NULL)
    beta = backward(EMIT, NULL)
    sums = diagonal_sums(alpha, beta)
    assert sums.shape == (5,), f"expected 5 diagonals (T + U), got shape {sums.shape}"
    assert np.allclose(sums, prob, atol=1e-9), f"expected every diagonal to sum to {prob}, got {sums}"


def test_identity_on_random_lattices():
    for T, U, seed in [(2, 3, 1), (5, 2, 2), (4, 4, 3)]:
        rng = np.random.default_rng(seed)
        emit = rng.uniform(0.05, 0.6, size=(T, U))
        null = rng.uniform(0.05, 0.6, size=(T, U + 1))
        alpha, prob = ref_forward(emit, null)
        sums = diagonal_sums(alpha, backward(emit, null))
        assert np.allclose(sums, prob, atol=1e-12), f"T={T}, U={U}: expected all {prob}, got {sums}"
