import numpy as np
from submission import welch_t


def test_hand_example_from_the_page():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 4.0, 6.0, 8.0, 10.0]
    # var(a) = 2.5, var(b) = 10, so va + vb = 0.5 + 2 = 2.5 and t = -3 / sqrt(2.5).
    t, df = welch_t(a, b)
    assert abs(t - (-1.8974)) < 1e-3, f"expected t = -1.8974, got {t}"
    assert abs(df - 5.8824) < 1e-3, f"expected df = 6.25 / 1.0625 = 5.8824, got {df}"


def test_unequal_sizes_and_variances():
    a = np.arange(1.0, 9.0)
    b = np.array([10.0, 12.0, 14.0])
    t, df = welch_t(a, b)
    assert abs(t - (-5.1962)) < 1e-3, f"expected t = -5.1962, got {t}"
    assert abs(df - 4.4780) < 1e-3, f"expected df = 4.4780, got {df}"


def test_equal_variances_and_sizes_give_two_n_minus_two():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a + 10.0
    t, df = welch_t(a, b)
    assert abs(df - 8.0) < 1e-9, f"same variance and size, so df = 2n - 2 = 8, got {df}"
    assert abs(t - (-10.0 / np.sqrt(2.0 * 2.5 / 5.0))) < 1e-9, f"expected t = -10, got {t}"


def test_sign_follows_the_first_sample():
    a = [5.0, 6.0, 7.0, 9.0]
    b = [1.0, 2.0, 2.5, 3.0, 4.0]
    t, _ = welch_t(a, b)
    t_swapped, _ = welch_t(b, a)
    assert t > 0, f"a has the larger mean, so t must be positive, got {t}"
    assert abs(t + t_swapped) < 1e-9, f"swapping the samples must flip the sign of t, got {t} and {t_swapped}"


def test_df_stays_between_the_bounds():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.normal(0, rng.uniform(0.5, 5), size=rng.integers(3, 9))
        b = rng.normal(1, rng.uniform(0.5, 5), size=rng.integers(3, 9))
        _, df = welch_t(a, b)
        low, high = min(len(a), len(b)) - 1, len(a) + len(b) - 2
        assert low - 1e-9 <= df <= high + 1e-9, f"df must lie in [{low}, {high}], got {df}"
