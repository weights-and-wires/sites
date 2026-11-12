import numpy as np
from submission import puct_scores, select_action

Q = np.array([0.2, 0.5, 0.0])
N = np.array([10.0, 2.0, 0.0])
P = np.array([0.6, 0.3, 0.1])
ALL = np.array([True, True, True])


def test_scores_on_page_example():
    got = np.asarray(puct_scores(Q, N, P, 5.0))
    want = np.array([1.1448, 2.2321, 1.7321])
    assert got.shape == (3,), f"expected shape (3,), got {got.shape}"
    assert np.allclose(got, want, atol=1e-3), f"expected {want}, got {got}"


def test_selects_move_1_on_page_example():
    got = select_action(Q, N, P, 5.0, ALL)
    assert got == 1, f"expected move 1, got {got}"


def test_square_root_uses_the_total_visits():
    # sqrt(sum N) = 2 here. Using N[a] instead of the sum would give other numbers.
    got = np.asarray(puct_scores(np.zeros(2), np.array([1.0, 3.0]), np.array([0.5, 0.5]), 1.0))
    want = np.array([0.5 * 2.0 / 2.0, 0.5 * 2.0 / 4.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_denominator_is_one_plus_n():
    got = np.asarray(puct_scores(np.zeros(2), np.array([0.0, 8.0]), np.array([1.0, 1.0]), 1.0))
    want = np.array([np.sqrt(8.0), np.sqrt(8.0) / 9.0])
    assert np.all(np.isfinite(got)), f"scores must be finite for an unvisited move, got {got}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_c_puct_scales_only_the_bonus():
    q = np.array([0.5, -0.5])
    n = np.array([3.0, 1.0])
    p = np.array([0.5, 0.5])
    s1 = np.asarray(puct_scores(q, n, p, 1.0))
    s3 = np.asarray(puct_scores(q, n, p, 3.0))
    assert np.allclose(s3 - q, 3.0 * (s1 - q)), f"expected the bonus to triple, got {s1} and {s3}"


def test_fresh_node_has_no_bonus():
    got = np.asarray(puct_scores(np.array([0.1, 0.4]), np.zeros(2), np.array([0.9, 0.1]), 5.0))
    assert np.allclose(got, [0.1, 0.4]), f"with no visits the score is Q, expected [0.1, 0.4], got {got}"


def test_prior_drives_early_choice_and_value_drives_late_choice():
    q = np.array([0.0, 0.6])
    p = np.array([0.9, 0.1])
    early = select_action(q, np.array([2.0, 0.0]), p, 5.0, ALL[:2])
    late = select_action(q, np.array([500.0, 500.0]), p, 5.0, ALL[:2])
    assert early == 0, f"with few visits the high prior should win, expected 0, got {early}"
    assert late == 1, f"with many visits the high value should win, expected 1, got {late}"


def test_illegal_moves_are_never_selected():
    q = np.array([5.0, 0.0, 0.1])
    got = select_action(q, np.array([1.0, 1.0, 1.0]), np.array([0.8, 0.1, 0.1]), 1.0, np.array([False, True, True]))
    assert got in (1, 2) and got != 0, f"move 0 is illegal, got {got}"
    assert got == 2, f"among legal moves 2 scores highest, expected 2, got {got}"


def test_ties_go_to_the_lowest_index():
    got = select_action(np.zeros(3), np.zeros(3), np.array([1 / 3] * 3), 5.0, ALL)
    assert got == 0, f"a fresh node ties, expected index 0, got {got}"
