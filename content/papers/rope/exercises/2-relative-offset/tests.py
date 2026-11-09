import numpy as np
from submission import rotary_matrix, rope_score


def rotate_directly(v, m, base=10000.0):
    # Rotate each adjacent pair by m * theta_j with plain trigonometry (Equation 13).
    d = len(v)
    out = np.empty(d)
    for j in range(d // 2):
        th = m * base ** (-2.0 * j / d)
        a, b = v[2 * j], v[2 * j + 1]
        out[2 * j] = a * np.cos(th) - b * np.sin(th)
        out[2 * j + 1] = a * np.sin(th) + b * np.cos(th)
    return out


def test_shape_and_zero_position():
    R = rotary_matrix(0, 6)
    assert R.shape == (6, 6), f"expected (6, 6), got {R.shape}"
    assert np.allclose(R, np.eye(6)), f"R at position 0 should be the identity, got\n{R}"


def test_block_structure_by_hand():
    # d = 4, m = 2: theta = [1, 0.01]. Blocks sit on the diagonal; everything else is 0.
    R = rotary_matrix(2, 4)
    c1, s1, c2, s2 = np.cos(2.0), np.sin(2.0), np.cos(0.02), np.sin(0.02)
    want = np.array([[c1, -s1, 0, 0], [s1, c1, 0, 0], [0, 0, c2, -s2], [0, 0, s2, c2]])
    assert np.allclose(R, want), f"expected\n{want}\ngot\n{R}"


def test_orthogonal():
    R = rotary_matrix(37, 8)
    assert np.allclose(R.T @ R, np.eye(8)), "R^T R should be the identity (the paper notes R is orthogonal)"


def test_transpose_composition_equation_16():
    d = 8
    lhs = rotary_matrix(5, d).T @ rotary_matrix(12, d)
    rhs = rotary_matrix(12 - 5, d)
    assert np.allclose(lhs, rhs), f"R_m^T R_n should equal R_(n-m); max difference {np.abs(lhs - rhs).max():.3g}"


def test_score_matches_rotating_both_vectors():
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=8), rng.normal(size=8)
    for m, n in [(0, 0), (3, 1), (1, 3), (40, 25), (7, 90)]:
        want = rotate_directly(q, m) @ rotate_directly(k, n)
        got = rope_score(q, k, m, n)
        assert np.isclose(got, want), f"m={m}, n={n}: expected {want:.6f}, got {got:.6f}"


def test_score_depends_only_on_offset():
    rng = np.random.default_rng(1)
    q, k = rng.normal(size=16), rng.normal(size=16)
    base = rope_score(q, k, 2, 9)
    for shift in [1, 10, 100, 1000]:
        got = rope_score(q, k, 2 + shift, 9 + shift)
        assert np.isclose(got, base), f"shift {shift}: offset is still 7, expected {base:.6f}, got {got:.6f}"


def test_sign_of_offset_matters():
    # q^T R_(n-m) k with q = [1, 0], k = [0, 1], m = 0, n = 1 is the top-right entry of R_1: -sin(1).
    # The transposed choice R_(m-n) would give +sin(1).
    q, k = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    got = rope_score(q, k, 0, 1)
    assert np.isclose(got, -np.sin(1.0)), f"expected -sin(1) = {-np.sin(1.0):.6f}, got {got:.6f}"


def test_same_content_gives_cosine_decay_in_one_pair():
    # With d = 2 and q = k, the score is |q|^2 cos((n - m) * theta) for theta = 1.
    q = np.array([2.0, 0.0])
    for off in [0, 1, 2, 5]:
        got = rope_score(q, q, 3, 3 + off)
        want = 4.0 * np.cos(off)
        assert np.isclose(got, want), f"offset {off}: expected {want:.6f}, got {got:.6f}"
