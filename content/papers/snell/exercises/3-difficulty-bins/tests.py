import numpy as np
from submission import difficulty_bins, compute_optimal


def test_bins_of_ten_questions():
    pass1 = np.array([0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10])
    got = difficulty_bins(pass1, 5)
    want = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert list(got) == want, f"expected {want}, got {list(got)}"


def test_bins_do_not_depend_on_the_input_order():
    pass1 = np.array([0.10, 0.95, 0.40, 0.80, 0.20, 0.60, 0.90, 0.30, 0.70, 0.50])
    got = difficulty_bins(pass1, 5)
    want = [4, 0, 3, 1, 4, 2, 0, 3, 1, 2]
    assert list(got) == want, f"expected {want}, got {list(got)}"


def test_bins_are_quantiles_not_equal_width_ranges():
    # Nine questions sit between 0.90 and 0.99 and one at 0.05. Equal-width ranges
    # would put nine questions in one bin. Quantile bins hold two questions each.
    pass1 = np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.05])
    got = difficulty_bins(pass1, 5)
    counts = np.bincount(got, minlength=5)
    assert list(counts) == [2, 2, 2, 2, 2], f"expected two questions per bin, got counts {list(counts)}"


def test_uneven_split_and_tie_order():
    # Seven questions in five bins: bin = (rank * 5) // 7, so sizes are 2, 1, 2, 1, 1.
    pass1 = np.array([0.5, 0.9, 0.9, 0.1, 0.7, 0.3, 0.5])
    got = difficulty_bins(pass1, 5)
    # Ranks: q1 0, q2 1, q4 2, q0 3, q6 4, q5 5, q3 6. The tie at 0.9 and at 0.5 goes to the lower index.
    want = [2, 0, 0, 4, 1, 3, 2]
    assert list(got) == want, f"expected {want}, got {list(got)}"


def test_bin_zero_is_the_easiest():
    got = difficulty_bins(np.array([0.1, 0.9]), 2)
    assert list(got) == [1, 0], f"expected [1, 0] (high pass@1 is bin 0), got {list(got)}"


def test_default_is_five_bins():
    got = difficulty_bins(np.linspace(1, 0, 10))
    assert got.max() == 4, f"expected the default of 5 bins (max index 4), got max {got.max()}"


OUTCOMES = {
    "bon": np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0]),
    "beam": np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 0]),
}
BINS = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])


def test_compute_optimal_picks_the_best_per_bin():
    choices, acc = compute_optimal(BINS, OUTCOMES, 5)
    # Bin means: bin 0 bon 1.0 beam 0.5; bin 1 bon 0.5 beam 1.0; bin 2 bon 0 beam 0.5;
    # bin 3 bon 0.5 beam 0.5 (tie, first in the dict); bin 4 both 0 (tie).
    want = ["bon", "beam", "beam", "bon", "bon"]
    assert choices == want, f"expected {want}, got {choices}"


def test_compute_optimal_accuracy():
    choices, acc = compute_optimal(BINS, OUTCOMES, 5)
    # Chosen outcomes: bin 0 bon (1, 1), bin 1 beam (1, 1), bin 2 beam (1, 0),
    # bin 3 bon (1, 0), bin 4 bon (0, 0): 6 correct of 10.
    assert np.isclose(acc, 0.6), f"expected 0.6, got {acc}"


def test_accuracy_is_weighted_by_bin_size():
    bins = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    outcomes = {"a": np.array([1, 1, 1, 1, 1, 1, 1, 0])}
    choices, acc = compute_optimal(bins, outcomes, 2)
    assert np.isclose(acc, 7 / 8), f"expected 7/8 = 0.875 (a size-7 bin and a size-1 bin), got {acc}"


def test_compute_optimal_beats_every_fixed_strategy_here():
    choices, acc = compute_optimal(BINS, OUTCOMES, 5)
    best_fixed = max(float(o.mean()) for o in OUTCOMES.values())
    assert acc > best_fixed, f"expected more than the best fixed strategy {best_fixed}, got {acc}"
