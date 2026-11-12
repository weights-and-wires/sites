import numpy as np
from submission import bootstrap_ci


def test_hand_example_from_the_page():
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    idx = np.array([[0, 0, 1, 1, 2], [3, 4, 4, 2, 1], [0, 1, 2, 3, 4], [4, 4, 4, 3, 3]])
    # Resample means are 18, 38, 30, 46. Their 25th and 75th percentiles are 27 and 40.
    lo, hi = bootstrap_ci(x, idx, alpha=0.5)
    assert abs(lo - 27.0) < 1e-9 and abs(hi - 40.0) < 1e-9, f"expected (27.0, 40.0), got ({lo}, {hi})"


def test_percentiles_of_means_not_of_the_data():
    x = np.arange(11.0)
    idx = np.array([[k, k, k] for k in range(11)])  # the resample means are 0, 1, ..., 10
    lo, hi = bootstrap_ci(x, idx, alpha=0.2)
    assert abs(lo - 1.0) < 1e-9 and abs(hi - 9.0) < 1e-9, f"alpha=0.2 gives the 10th and 90th percentile, expected (1.0, 9.0), got ({lo}, {hi})"


def test_alpha_sets_the_width():
    rng = np.random.default_rng(3)
    x = rng.normal(size=8)
    idx = rng.integers(0, 8, size=(500, 8))
    lo90, hi90 = bootstrap_ci(x, idx, alpha=0.10)
    lo99, hi99 = bootstrap_ci(x, idx, alpha=0.01)
    assert lo99 < lo90 < hi90 < hi99, f"a 99% interval must contain the 90% one, got 90%=({lo90}, {hi90}) and 99%=({lo99}, {hi99})"


def test_order_and_means_axis():
    x = np.array([0.0, 100.0])
    idx = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]])  # means 0, 100, 50
    lo, hi = bootstrap_ci(x, idx, alpha=0.5)
    assert lo <= hi, f"lo must not exceed hi, got ({lo}, {hi})"
    assert abs(lo - 25.0) < 1e-9 and abs(hi - 75.0) < 1e-9, f"expected (25.0, 75.0), got ({lo}, {hi})"


def test_degenerate_sample_gives_a_point():
    x = np.full(6, 7.5)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 6, size=(100, 6))
    lo, hi = bootstrap_ci(x, idx)
    assert lo == 7.5 and hi == 7.5, f"all seeds equal, so the interval is the point 7.5, got ({lo}, {hi})"
