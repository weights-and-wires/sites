import numpy as np
from submission import rope


def rotary_matrix(m, d, base=10000.0):
    # Equation 15, built block by block.
    R = np.zeros((d, d))
    for j in range(d // 2):
        th = base ** (-2.0 * j / d)
        c, s = np.cos(m * th), np.sin(m * th)
        R[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = [[c, -s], [s, c]]
    return R


def test_shape_and_leading_axes():
    x = np.random.default_rng(0).normal(size=(2, 3, 5, 8))
    out = rope(x, np.arange(5))
    assert out.shape == x.shape, f"expected shape {x.shape}, got {out.shape}"


def test_position_zero_is_identity():
    x = np.random.default_rng(1).normal(size=(1, 6))
    out = rope(x, np.array([0]))
    assert np.allclose(out, x), f"position 0 should not rotate: expected {x}, got {out}"


def test_two_dimensions_by_hand():
    # d = 2 has one pair with theta = 1. Rotating [1, 0] by m = 3 gives [cos 3, sin 3].
    out = rope(np.array([[1.0, 0.0]]), np.array([3]))
    want = [np.cos(3.0), np.sin(3.0)]
    assert np.allclose(out[0], want), f"expected [cos 3, sin 3] = {want}, got {out[0]}"


def test_counterclockwise_sign():
    # [1, 0] rotated by +pi/2 is [0, 1] (Equation 13), not [0, -1].
    out = rope(np.array([[1.0, 0.0]]), np.array([np.pi / 2]))
    assert np.allclose(out[0], [0.0, 1.0], atol=1e-12), f"expected [0, 1], got {out[0]}"


def test_matches_rotary_matrix_equation_15():
    rng = np.random.default_rng(2)
    d = 8
    x = rng.normal(size=(5, d))
    pos = np.array([0, 1, 4, 9, 30])
    got = rope(x, pos)
    for t, m in enumerate(pos):
        want = rotary_matrix(m, d) @ x[t]
        assert np.allclose(got[t], want), f"row {t} (m={m}): expected {want}, got {got[t]}"


def test_pairs_are_adjacent_not_split_in_half():
    # Only coordinates 0 and 1 are nonzero. With the paper's pairing, the rotation stays in
    # coordinates 0 and 1. A half-split pairing would move energy into coordinate d/2.
    d = 8
    x = np.zeros((1, d))
    x[0, 0] = 1.0
    out = rope(x, np.array([2]))
    assert np.allclose(out[0, 2:], 0.0), f"coordinates 2.. must stay 0, got {out[0, 2:]}"
    assert np.allclose(out[0, :2], [np.cos(2.0), np.sin(2.0)]), f"expected [cos 2, sin 2], got {out[0, :2]}"


def test_frequencies_fall_with_pair_index():
    # Pair j has theta_j = 10000 ** (-2j / d). For d = 4: theta = [1, 0.01].
    x = np.array([[1.0, 0.0, 1.0, 0.0]])
    out = rope(x, np.array([10]))
    want = [np.cos(10.0), np.sin(10.0), np.cos(0.1), np.sin(0.1)]
    assert np.allclose(out[0], want), f"expected {want}, got {out[0]}"


def test_custom_base():
    x = np.array([[1.0, 0.0, 1.0, 0.0]])
    out = rope(x, np.array([10]), base=100.0)
    want = [np.cos(10.0), np.sin(10.0), np.cos(1.0), np.sin(1.0)]
    assert np.allclose(out[0], want), f"base=100 gives theta = [1, 0.1]; expected {want}, got {out[0]}"


def test_norm_preserved_per_pair():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 16))
    out = rope(x, np.arange(6) * 7)
    before = np.hypot(x[:, 0::2], x[:, 1::2])
    after = np.hypot(out[:, 0::2], out[:, 1::2])
    assert np.allclose(before, after), f"rotation must keep each pair's length; max difference {np.abs(before - after).max():.3g}"


def test_each_row_uses_its_own_position():
    x = np.tile(np.array([[1.0, 0.0]]), (3, 1))
    out = rope(x, np.array([0, 1, 2]))
    want = np.array([[1.0, 0.0], [np.cos(1.0), np.sin(1.0)], [np.cos(2.0), np.sin(2.0)]])
    assert np.allclose(out, want), f"expected {want}, got {out}"
