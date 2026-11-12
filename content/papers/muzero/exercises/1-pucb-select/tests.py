import math

import numpy as np
from submission import minmax_normalize, puct_scores, select_action

Q = np.array([2.0, 5.0, 0.0])
N = np.array([10.0, 2.0, 0.0])
P = np.array([0.6, 0.3, 0.1])


def test_minmax_maps_tree_range_to_unit_interval():
    got = np.asarray(minmax_normalize(np.array([10.0, 20.0, 30.0]), 10.0, 30.0))
    want = np.array([0.0, 0.5, 1.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_minmax_uses_the_tree_range_not_the_node_range():
    got = np.asarray(minmax_normalize(np.array([2.0, 5.0]), 0.0, 10.0))
    want = np.array([0.2, 0.5])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_minmax_flat_range_gives_zeros_not_nan():
    got = np.asarray(minmax_normalize(np.array([4.0, 4.0]), 4.0, 4.0))
    assert np.all(np.isfinite(got)), f"expected finite values, got {got}"
    assert np.allclose(got, [0.0, 0.0]), f"expected [0, 0], got {got}"


def test_scores_on_page_example():
    got = np.asarray(puct_scores(Q, N, P, 0.0, 10.0))
    want = np.array([0.4363137, 0.9332418, 0.4332418])
    assert got.shape == (3,), f"expected shape (3,), got {got.shape}"
    assert np.allclose(got, want, atol=1e-5), f"expected {want}, got {got}"


def test_selects_move_1_on_page_example():
    got = select_action(Q, N, P, 0.0, 10.0)
    assert got == 1, f"expected move 1, got {got}"


def test_log_term_with_small_c2():
    # c1 = 1, c2 = 2, sum N = 6: factor = 1 + log((6 + 2 + 1) / 2).
    n = np.array([2.0, 4.0])
    p = np.array([0.5, 0.5])
    factor = 1.0 + math.log(9.0 / 2.0)
    want = np.array([0.5 * math.sqrt(6.0) / 3.0 * factor, 0.5 * math.sqrt(6.0) / 5.0 * factor])
    got = np.asarray(puct_scores(np.zeros(2), n, p, 0.0, 1.0, c1=1.0, c2=2.0))
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_default_constants_are_c1_1_25_and_c2_19652():
    n = np.array([800.0])
    got = float(np.asarray(puct_scores(np.zeros(1), n, np.array([1.0]), 0.0, 1.0))[0])
    want = math.sqrt(800.0) / 801.0 * (1.25 + math.log((800.0 + 19652.0 + 1.0) / 19652.0))
    assert abs(got - want) < 1e-9, f"expected {want}, got {got}"


def test_fresh_node_has_no_bonus_and_pure_normalized_q():
    got = np.asarray(puct_scores(np.array([1.0, 3.0]), np.zeros(2), np.array([0.9, 0.1]), 0.0, 4.0))
    assert np.allclose(got, [0.25, 0.75]), f"with no visits the score is the normalized Q, expected [0.25, 0.75], got {got}"


def test_normalization_makes_scores_independent_of_reward_scale():
    base = np.asarray(puct_scores(Q, N, P, 0.0, 10.0))
    scaled = np.asarray(puct_scores(Q * 1000.0, N, P, 0.0, 10000.0))
    assert np.allclose(base, scaled), f"expected equal scores {base}, got {scaled}"


def test_normalization_does_not_scale_the_bonus():
    n = np.array([3.0, 1.0])
    p = np.array([0.5, 0.5])
    a = np.asarray(puct_scores(np.zeros(2), n, p, 0.0, 1.0))
    b = np.asarray(puct_scores(np.zeros(2), n, p, 0.0, 100.0))
    assert np.allclose(a, b), f"Q = 0 so only the bonus remains and lo/hi must not change it, got {a} and {b}"


def test_value_wins_late_and_prior_wins_early():
    q = np.array([0.0, 0.3])
    p = np.array([0.9, 0.1])
    early = select_action(q, np.array([2.0, 0.0]), p, 0.0, 1.0)
    late = select_action(q, np.array([5000.0, 5000.0]), p, 0.0, 1.0)
    assert early == 0, f"with few visits the high prior should win, expected 0, got {early}"
    assert late == 1, f"with many visits the high value should win, expected 1, got {late}"


def test_illegal_edges_are_never_selected():
    q = np.array([9.0, 0.0, 1.0])
    n = np.array([1.0, 1.0, 1.0])
    p = np.array([0.8, 0.1, 0.1])
    got = select_action(q, n, p, 0.0, 10.0, legal=np.array([False, True, True]))
    assert got == 2, f"edge 0 is illegal and edge 2 scores higher than edge 1, expected 2, got {got}"


def test_ties_go_to_the_lowest_index():
    got = select_action(np.zeros(3), np.zeros(3), np.array([1 / 3] * 3), 0.0, 1.0)
    assert got == 0, f"a fresh node ties, expected index 0, got {got}"
