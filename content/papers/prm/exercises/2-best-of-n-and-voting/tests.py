import numpy as np
from submission import best_of_n, majority_vote, weighted_vote

# The five candidates of scripts/best_of_n_toy.py. The right answer is 12.
ANSWERS = [12, 15, 15, 12, 15]
ORM = [0.80, 0.85, 0.60, 0.70, 0.82]
STEPS = [[0.95, 0.90, 0.92], [0.97, 0.95, 0.20, 0.90], [0.90, 0.93, 0.30], [0.90, 0.30, 0.90], [0.96, 0.90, 0.40, 0.90]]
PRM = [float(np.prod(s)) for s in STEPS]


def test_best_of_n_on_the_page_example():
    assert best_of_n(ORM) == 1, f"ORM expected index 1, got {best_of_n(ORM)}"
    assert best_of_n(PRM) == 0, f"PRM expected index 0, got {best_of_n(PRM)}"


def test_best_of_n_returns_the_index_not_the_value():
    got = best_of_n([0.1, 0.9, 0.5])
    assert got == 1 and isinstance(got, (int, np.integer)), f"expected index 1, got {got!r}"


def test_best_of_n_tie_goes_to_the_first():
    got = best_of_n([0.2, 0.7, 0.7, 0.1])
    assert got == 1, f"expected the first of the tied scores (index 1), got {got}"


def test_best_of_n_last_and_negative():
    assert best_of_n([-3.0, -2.0, -1.0]) == 2, f"expected 2, got {best_of_n([-3.0, -2.0, -1.0])}"
    assert best_of_n([5.0]) == 0, f"expected 0, got {best_of_n([5.0])}"


def test_majority_vote_on_the_page_example():
    got = majority_vote(ANSWERS)
    assert got == 15, f"expected 15 (three votes against two), got {got}"


def test_majority_vote_is_by_count_not_by_value():
    got = majority_vote([100, 3, 3, 7])
    assert got == 3, f"expected 3 (two votes), got {got}"


def test_majority_vote_tie_goes_to_the_first_seen():
    got = majority_vote([9, 4, 9, 4])
    assert got == 9, f"expected 9 (first seen among the tied answers), got {got}"
    got = majority_vote(["b", "a", "b", "a"])
    assert got == "b", f"expected 'b', got {got!r}"


def test_weighted_vote_with_prm_scores_flips_the_vote():
    got = weighted_vote(ANSWERS, PRM)
    assert got == 12, f"expected 12 (total 1.0296 against 0.7280), got {got}"


def test_weighted_vote_with_orm_scores_keeps_the_wrong_answer():
    got = weighted_vote(ANSWERS, ORM)
    assert got == 15, f"expected 15 (total 2.27 against 1.50), got {got}"


def test_weighted_vote_sums_scores_instead_of_taking_the_max():
    got = weighted_vote(["a", "b", "b"], [0.9, 0.5, 0.5])
    assert got == "b", f"expected 'b' (total 1.0 against 0.9), got {got!r}"


def test_weighted_vote_tie_goes_to_the_first_seen():
    got = weighted_vote(["x", "y", "y", "x"], [0.5, 0.25, 0.25, 0.25])
    assert got == "x", f"expected 'x' (totals 0.75 and 0.5), got {got!r}"
    got = weighted_vote(["x", "y"], [0.5, 0.5])
    assert got == "x", f"expected 'x' on an exact tie, got {got!r}"
