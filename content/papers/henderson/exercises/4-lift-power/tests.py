import numpy as np
from submission import lift_power


def test_identity_resamples_by_hand():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ident = np.tile(np.arange(5), (3, 1))
    # Group B is [2, 4, 6, 8, 10], so |t| = 1.8974 in every row.
    got = lift_power(x, 2.0, ident, ident, 1.5)
    assert got == 1.0, f"|t| = 1.8974 > 1.5 in all rows, expected 1.0, got {got}"
    got = lift_power(x, 2.0, ident, ident, 2.0)
    assert got == 0.0, f"|t| = 1.8974 < 2.0 in all rows, expected 0.0, got {got}"


def test_lift_applies_to_one_group_only():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ident = np.tile(np.arange(5), (2, 1))
    got = lift_power(x, 1.0, ident, ident, 0.1)
    assert got == 0.0, f"no lift and identical resamples give t = 0, expected 0.0, got {got}"
    got = lift_power(x, 3.0, ident, ident, 1.0)
    assert got == 1.0, f"a lift of 3 on group B only must separate the groups, expected 1.0, got {got}"


def test_two_sided():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ident = np.tile(np.arange(5), (2, 1))
    got = lift_power(x, 0.5, ident, ident, 1.5)
    assert got == 1.0, f"a lift of 0.5 makes group B smaller; t is positive and |t| counts, expected 1.0, got {got}"


def test_groups_use_their_own_resamples():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    idx_a = np.array([[0, 1, 2, 3, 4]])
    idx_b = np.array([[0, 0, 0, 0, 1]])  # B = [1, 1, 1, 1, 2], mean 1.2, var 0.2
    # var(A)/5 = 0.5, var(B)/5 = 0.04, t = 1.8 / sqrt(0.54) = 2.4495
    assert lift_power(x, 1.0, idx_a, idx_b, 2.0) == 1.0, "t = 2.4495 > 2.0, so this row counts; B must use idx_b"
    assert lift_power(x, 1.0, idx_a, idx_b, 3.0) == 0.0, "t = 2.4495 < 3.0, so no row counts"


def test_fraction_over_rows():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    idx_a = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    idx_b = np.array([[0, 1, 2, 3, 4], [0, 0, 0, 0, 1], [4, 4, 4, 4, 3], [0, 1, 2, 3, 4]])
    # t for the four rows with lift 1: 0, 2.4495, -(2.4495), 0.
    got = lift_power(x, 1.0, idx_a, idx_b, 2.0)
    assert abs(got - 0.5) < 1e-9, f"rows 2 and 3 pass |t| > 2.0, expected 0.5, got {got}"


def test_noisy_sample_needs_more_seeds():
    x = np.array([800.0, 4200.0, 2500.0, 5200.0, 1300.0])
    rng = np.random.default_rng(0)
    p5 = lift_power(x, 1.25, rng.integers(0, 5, (1000, 5)), rng.integers(0, 5, (1000, 5)), 2.306)
    p40 = lift_power(x, 1.25, rng.integers(0, 5, (1000, 40)), rng.integers(0, 5, (1000, 40)), 1.990)
    assert p40 > p5 + 0.15, f"power must grow with the seeds per group, got {p5} at 5 and {p40} at 40"
