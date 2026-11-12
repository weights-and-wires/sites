import numpy as np
from submission import band_gap_rate, welch_gap_rate


def test_band_rule_by_hand():
    runs = np.array(
        [
            [0.0, 2.0, 10.0, 12.0],  # means 1 and 11, se 1 and 1: the gap 10 is more than 2
            [0.0, 10.0, 4.0, 6.0],  # means 5 and 5: no gap
        ]
    )
    got = band_gap_rate(runs)
    assert got == 0.5, f"one of two rows has non-overlapping bands, expected 0.5, got {got}"


def test_band_rule_uses_standard_error_not_standard_deviation():
    # Group A: [0, 6, 0, 6], mean 3, sd 3.4641, se 1.7321. Group B: [8, 8, 8, 8], mean 8, se 0.
    # The gap 5 is more than 1.7321 + 0, but less than sd_a + sd_b = 3.4641.
    runs = np.array([[0.0, 6.0, 0.0, 6.0, 8.0, 8.0, 8.0, 8.0]])
    got = band_gap_rate(runs)
    assert got == 1.0, f"the bands use SE = s / sqrt(n), so the gap 5 beats 1.73 and expected 1.0, got {got}"


def test_band_rule_uses_sample_variance():
    # n = 2: the sample sd of [0, 2] is sqrt(2) with ddof=1 (the population sd is 1).
    # Means are 1 and 2.8, so the gap is 1.8. With ddof=1 each se is 1 and the bound is 2: no gap.
    # With ddof=0 each se is 0.7071 and the bound is 1.414: a false gap.
    runs = np.array([[0.0, 2.0, 1.8, 3.8]])
    got = band_gap_rate(runs)
    assert got == 0.0, f"gap 1.8 is inside the bound 2 (sample variance, ddof=1), expected 0.0, got {got}"


def test_the_split_is_first_half_versus_second_half():
    row = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    runs = row[None, :]
    assert band_gap_rate(runs) == 1.0, "groups [1..5] and [11..15] are far apart; check the split"
    assert welch_gap_rate(runs, 2.306) == 1.0, "Welch |t| is 10 here; check the split"
    assert band_gap_rate(row[::-1][None, :]) == 1.0, "reversing the row swaps the groups; the answer is the same"


def test_welch_rule_by_hand():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b_near = [2.0, 4.0, 6.0, 8.0, 10.0]  # |t| = 1.8974
    b_far = [11.0, 12.0, 13.0, 14.0, 15.0]  # t = -10
    runs = np.array([a + b_near, a + b_far, b_far + a])
    got = welch_gap_rate(runs, 2.306)
    assert abs(got - 2 / 3) < 1e-9, f"rows 2 and 3 pass |t| > 2.306 (row 3 has t = +10), expected 2/3, got {got}"
    got = welch_gap_rate(runs, 1.5)
    assert got == 1.0, f"with crit 1.5 all three rows pass, expected 1.0, got {got}"
    got = welch_gap_rate(runs, 20.0)
    assert got == 0.0, f"with crit 20 no row passes, expected 0.0, got {got}"


def test_simulated_false_alarm_rates():
    rng = np.random.default_rng(0)
    runs = rng.normal(3000.0, 800.0, size=(4000, 10))
    band = band_gap_rate(runs)
    welch = welch_gap_rate(runs, 2.306)
    assert 0.15 < band < 0.26, f"band rule on same-algorithm seeds fires about 20% of the time, got {band}"
    assert 0.03 < welch < 0.07, f"Welch at the 5% level fires about 5% of the time, got {welch}"
