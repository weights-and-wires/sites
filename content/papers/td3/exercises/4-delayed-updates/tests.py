import math

from submission import actor_update_steps, target_gap


def test_d2_ten_steps():
    got = list(actor_update_steps(10, 2))
    assert got == [2, 4, 6, 8, 10], f"expected [2, 4, 6, 8, 10], got {got}"


def test_d1_updates_every_step():
    got = list(actor_update_steps(5, 1))
    assert got == [1, 2, 3, 4, 5], f"expected [1, 2, 3, 4, 5], got {got}"


def test_d3_does_not_include_a_partial_cycle():
    got = list(actor_update_steps(10, 3))
    assert got == [3, 6, 9], f"expected [3, 6, 9], got {got}"


def test_first_step_is_not_an_actor_step_when_d_is_2():
    assert 1 not in actor_update_steps(4, 2), "step 1 is not a multiple of 2"
    assert 0 not in actor_update_steps(4, 2), "steps are 1-based, so 0 never appears"


def test_count_over_a_long_run():
    got = len(actor_update_steps(1000, 2))
    assert got == 500, f"half of 1000 steps, expected 500, got {got}"


def test_d_larger_than_run():
    got = list(actor_update_steps(3, 5))
    assert got == [], f"expected no actor steps, got {got}"


def test_gap_with_delay():
    got = target_gap(1.0, 0.005, 1000, 2)
    assert math.isclose(got, 0.995 ** 500, rel_tol=1e-9), f"expected {0.995 ** 500:.4f}, got {got}"


def test_gap_without_delay():
    got = target_gap(1.0, 0.005, 1000, 1)
    assert math.isclose(got, 0.995 ** 1000, rel_tol=1e-9), f"expected {0.995 ** 1000:.4f}, got {got}"


def test_gap_scales_with_the_start():
    got = target_gap(4.0, 0.1, 6, 3)
    assert math.isclose(got, 4.0 * 0.9 ** 2, rel_tol=1e-9), f"two target updates: expected {4.0 * 0.81}, got {got}"
