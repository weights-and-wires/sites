import numpy as np
from submission import drafter_input, drafter_step, smooth_l1

F = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
E = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0], [50.0, 51.0]])
TOK = np.array([4, 2, 0, 3])


def test_input_shape():
    got = drafter_input(F, TOK, E)
    assert got.shape == (3, 4), f"expected shape (3, 4), got {got.shape}"


def test_input_is_feature_then_advanced_token_embedding():
    want = np.array([
        [1.0, 2.0, 30.0, 31.0],   # F[0] with E[tokens[1]] = E[2]
        [3.0, 4.0, 10.0, 11.0],   # F[1] with E[tokens[2]] = E[0]
        [5.0, 6.0, 40.0, 41.0],   # F[2] with E[tokens[3]] = E[3]
    ])
    got = drafter_input(F, TOK, E)
    assert np.array_equal(got, want), f"expected\n{want}\ngot\n{got}"


def test_the_token_is_advanced_not_aligned():
    got = drafter_input(F, TOK, E)
    aligned = np.concatenate([F[:-1], E[TOK[:-1]]], axis=1)
    assert not np.array_equal(got, aligned), "rows must use tokens[i + 1], not tokens[i]"


def test_step_uses_both_halves_with_their_own_weights():
    W = np.vstack([np.eye(2), 0.1 * np.eye(2)])  # features pass through, token embeddings scaled by 0.1
    b = np.array([1.0, -1.0])
    got = drafter_step(F, TOK, E, W, b)
    want = np.array([[1 + 3.0 + 1, 2 + 3.1 - 1], [3 + 1.0 + 1, 4 + 1.1 - 1], [5 + 4.0 + 1, 6 + 4.1 - 1]])
    assert np.allclose(got, want), f"expected\n{want}\ngot\n{got}"


def test_step_does_not_change_the_inputs():
    F0, T0 = F.copy(), TOK.copy()
    drafter_step(F, TOK, E, np.ones((4, 2)), np.zeros(2))
    assert np.array_equal(F, F0) and np.array_equal(TOK, T0), "inputs must not be modified"


def test_smooth_l1_quadratic_branch():
    got = smooth_l1(np.array([0.5]), np.array([0.0]))
    assert abs(got - 0.125) < 1e-12, f"expected 0.125, got {got}"


def test_smooth_l1_linear_branch():
    got = smooth_l1(np.array([3.0]), np.array([0.0]))
    assert abs(got - 2.5) < 1e-12, f"expected 2.5, got {got}"


def test_smooth_l1_is_symmetric_and_a_mean():
    pred = np.array([[0.5, -3.0], [0.0, 1.0]])
    got = smooth_l1(pred, np.zeros((2, 2)))
    want = (0.125 + 2.5 + 0.0 + 0.5) / 4
    assert abs(got - want) < 1e-12, f"expected {want}, got {got}"


def test_smooth_l1_beta():
    got = smooth_l1(np.array([0.5]), np.array([0.0]), beta=0.25)
    want = 0.5 - 0.125
    assert abs(got - want) < 1e-12, f"beta = 0.25 puts x = 0.5 on the linear branch, expected {want}, got {got}"
