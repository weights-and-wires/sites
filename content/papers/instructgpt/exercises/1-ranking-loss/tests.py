import numpy as np
from submission import reward_model_loss

R = np.array([0.2, 1.5, -0.7, 0.9])
RANK = [1, 3, 0, 2]


def test_two_responses():
    got = reward_model_loss(np.array([2.0, 1.0]), [0, 1])
    assert abs(got - 0.3133) < 1e-4, f"expected -log sigmoid(1) = 0.3133, got {got}"


def test_direction():
    right = reward_model_loss(np.array([2.0, 1.0]), [0, 1])
    wrong = reward_model_loss(np.array([2.0, 1.0]), [1, 0])
    assert abs(wrong - 1.3133) < 1e-4, f"expected -log sigmoid(-1) = 1.3133 for the reversed label, got {wrong}"
    assert wrong > right, f"a ranking that disagrees with the scores must cost more: {wrong} vs {right}"


def test_four_responses():
    got = reward_model_loss(R, RANK)
    assert abs(got - 0.2853) < 1e-4, f"expected 0.2853 (mean over 6 pairs), got {got}"


def test_reversed_ranking():
    got = reward_model_loss(R, RANK[::-1])
    assert abs(got - 1.5020) < 1e-4, f"expected 1.5020 for the reversed ranking, got {got}"


def test_uses_all_pairs_not_only_neighbors():
    # Neighbors: (B,D) and (D,A) and (A,C) only gives a different mean.
    got = reward_model_loss(R, RANK)
    neighbor = np.mean([np.logaddexp(0, -(R[RANK[i]] - R[RANK[i + 1]])) for i in range(3)])
    assert abs(got - neighbor) > 1e-3, "test setup is wrong"
    assert abs(got - 0.2853) < 1e-4, f"expected the mean over all 6 pairs, 0.2853, got {got}"


def test_mean_not_sum():
    got = reward_model_loss(R, RANK)
    assert got < 0.5, f"expected a mean near 0.2853, got {got}, which looks like a sum (1.7118)"


def test_shift_invariance():
    a = reward_model_loss(R, RANK)
    b = reward_model_loss(R + 100.0, RANK)
    assert abs(a - b) < 1e-9, f"adding a constant to every score must not change the loss: {a} vs {b}"


def test_large_gap_is_finite():
    got = reward_model_loss(np.array([1000.0, -1000.0, 0.0]), [0, 2, 1])
    assert np.isfinite(got) and abs(got) < 1e-6, f"expected a finite loss near 0, got {got}"
    bad = reward_model_loss(np.array([-1000.0, 1000.0]), [0, 1])
    assert np.isfinite(bad) and abs(bad - 2000.0) < 1e-6, f"expected 2000 for a gap of -2000, got {bad}"


def test_returns_python_float():
    got = reward_model_loss(R, RANK)
    assert isinstance(got, float), f"expected a Python float, got {type(got)}"
