import numpy as np
from submission import n_step_target


def test_page_example():
    got = n_step_target([1.0, 0.0, 2.0], False, 5.0, 0.9)
    assert np.isclose(got, 6.265), f"expected 2.62 + 0.729 * 5 = 6.265, got {got}"


def test_first_reward_is_undiscounted():
    got = n_step_target([3.0], False, 0.0, 0.5)
    assert np.isclose(got, 3.0), f"expected 3.0, got {got}"


def test_reward_order():
    got = n_step_target([0.0, 0.0, 8.0], True, 99.0, 0.5)
    assert np.isclose(got, 2.0), f"the reward 8 sits at offset 2 and gives 0.25 * 8 = 2.0, got {got}"
    got = n_step_target([8.0, 0.0, 0.0], True, 99.0, 0.5)
    assert np.isclose(got, 8.0), f"the reward 8 sits at offset 0 and gives 8.0, got {got}"


def test_terminal_ignores_bootstrap():
    got = n_step_target([1.0, 2.0], True, 1e6, 0.9)
    assert np.isclose(got, 2.8), f"expected 1 + 0.9 * 2 = 2.8 whatever q_boot is, got {got}"


def test_short_window_discount_uses_its_own_length():
    got = n_step_target([1.0, 2.0], False, 10.0, 0.9)
    assert np.isclose(got, 2.8 + 0.81 * 10.0), f"expected 2.8 + 0.9**2 * 10 = 10.9, got {got}"


def test_gamma_one_is_a_plain_sum():
    got = n_step_target([1.0, -1.0, 4.0], False, 2.0, 1.0)
    assert np.isclose(got, 6.0), f"expected 1 - 1 + 4 + 2 = 6.0, got {got}"


def test_negative_bootstrap():
    got = n_step_target([0.0, 0.0], False, -4.0, 0.5)
    assert np.isclose(got, -1.0), f"expected 0.25 * -4 = -1.0, got {got}"
