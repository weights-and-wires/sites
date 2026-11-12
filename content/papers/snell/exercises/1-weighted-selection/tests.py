import numpy as np
from submission import step_score, select_answer

# The two problems of scripts/weighted_selection.py.
A_ANSWERS, A_SCORES = [7, 12, 12, 12, 9], [0.90, 0.55, 0.50, 0.45, 0.60]   # correct answer 12
B_ANSWERS, B_SCORES = [5, 5, 5, 8, 8], [0.20, 0.30, 0.20, 0.80, 0.70]      # correct answer 8


def test_step_score_last():
    got = step_score([0.9, 0.2, 0.8], "last")
    assert np.isclose(got, 0.8), f"expected 0.8 (the final step), got {got}"
    assert np.isclose(step_score([0.9, 0.2, 0.8]), 0.8), "the default should be \"last\""


def test_step_score_min_and_prod():
    got = step_score([0.9, 0.2, 0.8], "min")
    assert np.isclose(got, 0.2), f"expected min 0.2, got {got}"
    got = step_score([0.9, 0.2, 0.8], "prod")
    assert np.isclose(got, 0.144), f"expected product 0.144, got {got}"


def test_step_score_single_step_and_order():
    for how in ("last", "min", "prod"):
        got = step_score([0.37], how)
        assert np.isclose(got, 0.37), f"{how}: expected 0.37 for one step, got {got}"
    got = step_score([0.1, 0.9], "last")
    assert np.isclose(got, 0.9), f"expected 0.9 (last, not first), got {got}"


def test_step_score_rejects_unknown_mode():
    try:
        step_score([0.5, 0.5], "mean")
    except ValueError:
        return
    raise AssertionError("expected ValueError for how=\"mean\"")


def test_best_of_n_follows_the_top_score():
    got = select_answer(A_ANSWERS, A_SCORES, "best_of_n")
    assert got == 7, f"problem A: expected 7 (score 0.90), got {got}"
    got = select_answer(B_ANSWERS, B_SCORES, "best_of_n")
    assert got == 8, f"problem B: expected 8 (score 0.80), got {got}"


def test_majority_ignores_scores():
    got = select_answer(A_ANSWERS, A_SCORES, "majority")
    assert got == 12, f"problem A: expected 12 (three votes), got {got}"
    got = select_answer(B_ANSWERS, B_SCORES, "majority")
    assert got == 5, f"problem B: expected 5 (three votes against two), got {got}"
    got = select_answer(B_ANSWERS, [0.0, 0.0, 0.0, 1.0, 1.0], "majority")
    assert got == 5, f"scores must not change a vote count, expected 5, got {got}"


def test_weighted_adds_scores_per_answer():
    got = select_answer(A_ANSWERS, A_SCORES, "weighted")
    assert got == 12, f"problem A: expected 12 (total 1.50 against 0.90), got {got}"
    got = select_answer(B_ANSWERS, B_SCORES, "weighted")
    assert got == 8, f"problem B: expected 8 (total 1.50 against 0.70), got {got}"


def test_weighted_is_a_sum_not_a_mean_or_a_max():
    # Answer "x" has one score of 0.9. Answer "y" has three scores of 0.4: sum 1.2, mean 0.4.
    got = select_answer(["x", "y", "y", "y"], [0.9, 0.4, 0.4, 0.4], "weighted")
    assert got == "y", f"expected y (sum 1.2 against 0.9), got {got}"
    got = select_answer(["x", "x", "y"], [0.5, 0.5, 0.9], "weighted")
    assert got == "x", f"expected x (sum 1.0 against 0.9), got {got}"


def test_ties_go_to_the_first_answer():
    got = select_answer([3, 4, 4, 3], [0.5, 0.5, 0.5, 0.5], "majority")
    assert got == 3, f"majority tie: expected 3 (appears first), got {got}"
    got = select_answer([3, 4, 4, 3], [0.5, 0.5, 0.5, 0.5], "weighted")
    assert got == 3, f"weighted tie: expected 3 (appears first), got {got}"
    got = select_answer([3, 4, 5], [0.5, 0.7, 0.7], "best_of_n")
    assert got == 4, f"best_of_n tie: expected 4 (first of the tied scores), got {got}"


def test_returns_the_answer_not_an_index():
    got = select_answer(["a", "b", "b"], [0.1, 0.2, 0.3], "weighted")
    assert got == "b", f"expected the answer b, got {got!r}"


def test_rejects_unknown_method():
    try:
        select_answer([1, 2], [0.5, 0.5], "median")
    except ValueError:
        return
    raise AssertionError("expected ValueError for method=\"median\"")
