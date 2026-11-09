import numpy as np

from submission import joint_sequence, sampled_rows

W = np.array([101, 102, 103, 104])
A = np.arange(1, 13).reshape(3, 4)
B = 200 + np.arange(1, 13).reshape(3, 4)


def test_page_example():
    want = np.array(
        [
            [101, 102, 103, 104],
            [1, 2, 3, 4],
            [0, 5, 6, 7],
            [0, 9, 10, 11],
            [201, 202, 203, 204],
            [0, 205, 206, 207],
            [0, 209, 210, 211],
        ]
    )
    got = joint_sequence(W, A, B, 1)
    assert np.array_equal(got, want), f"expected\n{want}\ngot\n{got}"


def test_shape_is_2q_plus_1():
    got = joint_sequence(W, A, B, 1)
    assert got.shape == (7, 4), f"expected shape (7, 4), got {got.shape}"
    rng = np.random.default_rng(0)
    A8 = rng.integers(1, 2048, size=(8, 6))
    B8 = rng.integers(1, 2048, size=(8, 6))
    got8 = joint_sequence(np.arange(1, 7), A8, B8, 2)
    assert got8.shape == (17, 6), f"expected shape (17, 6) for Q = 8, got {got8.shape}"


def test_text_and_semantic_rows_are_not_delayed():
    got = joint_sequence(W, A, B, 2)
    assert np.array_equal(got[0], W), f"the text row must be W, got {got[0]}"
    assert np.array_equal(got[1], A[0]), f"Moshi's semantic row must be undelayed, got {got[1]}"
    assert np.array_equal(got[4], B[0]), f"the user's semantic row must be undelayed, got {got[4]}"


def test_acoustic_rows_are_delayed_by_tau():
    got = joint_sequence(W, A, B, 2)
    assert np.array_equal(got[2], [0, 0, 5, 6]), f"expected [0, 0, 5, 6], got {got[2]}"
    assert np.array_equal(got[6], [0, 0, 209, 210]), f"expected [0, 0, 209, 210], got {got[6]}"


def test_tau_zero_stacks_the_inputs():
    got = joint_sequence(W, A, B, 0)
    want = np.vstack([W[None], A, B])
    assert np.array_equal(got, want), f"expected\n{want}\ngot\n{got}"


def test_sampled_rows():
    assert list(sampled_rows(8)) == list(range(9)), f"expected rows 0 to 8, got {list(sampled_rows(8))}"
    assert list(sampled_rows(3)) == [0, 1, 2, 3], f"expected [0, 1, 2, 3], got {list(sampled_rows(3))}"


def test_sampled_rows_leave_the_user_rows_out():
    Q = 8
    user = set(range(Q + 1, 2 * Q + 1))
    got = set(sampled_rows(Q))
    assert not (got & user), f"Moshi must not sample the user's rows {sorted(user)}; got overlap {sorted(got & user)}"
