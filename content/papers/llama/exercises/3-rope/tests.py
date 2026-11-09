import numpy as np
from submission import apply_rope


def test_hand_example_adjacent_counterclockwise():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    got = apply_rope(x, np.array([1]))
    want = np.array([[-1.1426, 1.9221, 2.9599, 4.0298]])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_position_zero_is_identity():
    x = np.arange(8.0).reshape(1, 8)
    got = apply_rope(x, np.array([0]))
    assert np.allclose(got, x), f"expected {x}, got {got}"


def test_rotation_keeps_pair_lengths():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 8))
    y = apply_rope(x, np.arange(6) * 37)
    before = np.hypot(x[:, 0::2], x[:, 1::2])
    after = np.hypot(y[:, 0::2], y[:, 1::2])
    assert np.allclose(before, after), f"a rotation must keep each pair's length; max change {np.abs(before - after).max():.3g}"


def test_score_depends_only_on_the_offset():
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=(1, 8)), rng.normal(size=(1, 8))

    def score(m, n):
        return float((apply_rope(q, np.array([m])) * apply_rope(k, np.array([n]))).sum())

    assert np.isclose(score(3, 1), score(50, 48)), f"expected equal scores at offset 2, got {score(3, 1)} and {score(50, 48)}"
    assert np.isclose(score(0, 0), score(40, 40)), f"expected equal scores at offset 0, got {score(0, 0)} and {score(40, 40)}"
    assert np.isclose(score(3, 1), -1.763129, atol=1e-5), f"expected -1.7631 for (m, n) = (3, 1), got {score(3, 1):.6f}"
    assert not np.isclose(score(1, 3), score(3, 1)), "the score at offset -2 must differ from the score at offset +2"


def test_base_argument_is_used():
    x = np.array([[1.0, 0.0, 1.0, 0.0]])
    got = apply_rope(x, np.array([2]), base=100.0)
    want = np.array([[np.cos(2.0), np.sin(2.0), np.cos(0.2), np.sin(0.2)]])
    assert np.allclose(got, want, atol=1e-6), f"expected {want}, got {got}"


def test_rows_use_their_own_positions():
    x = np.tile(np.array([[1.0, 0.0, 1.0, 0.0]]), (3, 1))
    y = apply_rope(x, np.array([0, 1, 2]))
    want = np.array([[np.cos(m), np.sin(m), np.cos(0.01 * m), np.sin(0.01 * m)] for m in (0, 1, 2)])
    assert np.allclose(y, want, atol=1e-6), f"expected\n{want}\ngot\n{y}"
