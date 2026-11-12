import numpy as np
from submission import best_of_n_accuracy, majority_vote_accuracy

TRIALS = 20000
TOL = 0.015  # about 5 standard errors at 20000 trials


def rng(seed=0):
    return np.random.default_rng(seed)


def close(got, want, tol=TOL):
    return abs(got - want) <= tol


def test_returns_a_python_float():
    a = best_of_n_accuracy(3, 0.5, 1.0, 1.0, 100, rng())
    b = majority_vote_accuracy(3, 0.5, 5, 100, rng())
    assert isinstance(a, float), f"best_of_n_accuracy should return a float, got {type(a).__name__}"
    assert isinstance(b, float), f"majority_vote_accuracy should return a float, got {type(b).__name__}"


def test_n_equal_one_gives_p_for_any_scorer():
    for sep, sigma in [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]:
        got = best_of_n_accuracy(1, 0.3, sep, sigma, TRIALS, rng(1))
        assert close(got, 0.3), f"expected about 0.3 with n=1, sep={sep}, sigma={sigma}, got {got:.4f}"


def test_oracle_scorer_matches_one_minus_q_to_the_n():
    for n, p in [(2, 0.3), (5, 0.3), (8, 0.2)]:
        want = 1 - (1 - p) ** n
        got = best_of_n_accuracy(n, p, 1.0, 0.0, TRIALS, rng(2))
        assert close(got, want), f"expected about {want:.4f} for n={n}, p={p}, got {got:.4f}"


def test_random_scorer_stays_at_p():
    got = best_of_n_accuracy(16, 0.3, 0.0, 1.0, TRIALS, rng(3))
    assert close(got, 0.3), f"expected about 0.3 for a scorer that ignores correctness, got {got:.4f}"


def test_noisier_scorer_is_worse_and_more_samples_help():
    clean = best_of_n_accuracy(8, 0.3, 1.0, 0.5, TRIALS, rng(4))
    noisy = best_of_n_accuracy(8, 0.3, 1.0, 2.0, TRIALS, rng(4))
    assert clean > noisy + 0.2, f"expected sigma=0.5 far above sigma=2, got {clean:.3f} and {noisy:.3f}"
    small = best_of_n_accuracy(2, 0.3, 1.0, 1.0, TRIALS, rng(5))
    large = best_of_n_accuracy(32, 0.3, 1.0, 1.0, TRIALS, rng(5))
    assert large > small + 0.2, f"expected n=32 far above n=2, got {large:.3f} and {small:.3f}"


def test_scorer_with_a_known_closed_form():
    # n=2, sigma=1, sep=1: with one correct and one wrong, the correct one wins when
    # 1 + e1 > e2, which has probability Phi(1 / sqrt(2)) = 0.7602.
    # P(exactly one correct) = 2 * 0.5 * 0.5 = 0.5, and both-correct counts as a hit.
    want = 0.25 + 0.5 * 0.7602
    got = best_of_n_accuracy(2, 0.5, 1.0, 1.0, TRIALS, rng(6))
    assert close(got, want), f"expected about {want:.4f}, got {got:.4f}"


def test_majority_vote_matches_the_binomial_with_one_wrong_answer():
    # n_wrong = 1: the vote is a majority of Bernoulli(p) draws.
    got = majority_vote_accuracy(3, 0.7, 1, TRIALS, rng(7))
    assert close(got, 0.784), f"expected about 0.784 for n=3, p=0.7, got {got:.4f}"
    got = majority_vote_accuracy(5, 0.6, 1, TRIALS, rng(8))
    assert close(got, 0.68256), f"expected about 0.6826 for n=5, p=0.6, got {got:.4f}"


def test_majority_vote_beats_one_sample_when_wrong_answers_scatter():
    one = majority_vote_accuracy(1, 0.3, 20, TRIALS, rng(9))
    many = majority_vote_accuracy(16, 0.3, 20, TRIALS, rng(9))
    assert close(one, 0.3), f"expected about 0.3 for one sample, got {one:.4f}"
    assert many > 0.8, f"expected above 0.8 for 16 samples with 20 scattered wrong answers, got {many:.4f}"


def test_majority_vote_loses_when_one_wrong_answer_dominates():
    got = majority_vote_accuracy(15, 0.3, 1, TRIALS, rng(10))
    assert got < 0.15, f"expected a low accuracy when wrong answers agree with each other, got {got:.4f}"


def test_majority_vote_tie_is_not_a_win_for_either_side():
    # n=2, one wrong answer, p=0.5: both right (0.25) wins, both wrong (0.25) loses,
    # a split (0.5) is a tie that goes to the answer seen first, so it wins half the time.
    got = majority_vote_accuracy(2, 0.5, 1, TRIALS, rng(11))
    assert close(got, 0.5), f"expected about 0.5 (ties split by who appears first), got {got:.4f}"
