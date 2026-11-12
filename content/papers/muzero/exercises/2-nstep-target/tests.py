import numpy as np
from submission import nstep_target, nstep_targets

U = [1.0, 0.0, 2.0, 0.0, 1.0]
V = [5.0, 4.0, 3.0, 2.0, 1.0]


def test_page_example_with_bootstrap():
    # n = 2, gamma = 0.5, t = 1: u[1] + 0.5 u[2] + 0.25 v[3] = 0 + 1 + 0.5
    got = nstep_target(U, V, 1, 2, 0.5)
    assert abs(got - 1.5) < 1e-12, f"expected 1.5, got {got}"


def test_bootstrap_is_discounted_by_gamma_to_the_n():
    # n = 3, gamma = 0.9, t = 0: 1 + 0 + 0.81 * 2 + 0.729 * v[3]
    got = nstep_target(U, V, 0, 3, 0.9)
    want = 1.0 + 0.0 + 0.81 * 2.0 + 0.729 * 2.0
    assert abs(got - want) < 1e-12, f"expected {want}, got {got}"


def test_bootstrap_uses_v_at_t_plus_n_not_t_plus_n_minus_1():
    v = [100.0, 0.0, 0.0, 7.0, 100.0]
    got = nstep_target([0.0] * 5, v, 0, 3, 1.0)
    assert abs(got - 7.0) < 1e-12, f"expected v[3] = 7.0, got {got}"


def test_n_equals_one_is_a_td_target():
    got = nstep_target(U, V, 2, 1, 0.9)
    want = 2.0 + 0.9 * 2.0
    assert abs(got - want) < 1e-12, f"expected {want}, got {got}"


def test_no_bootstrap_when_the_window_reaches_the_end():
    # t = 3, n = 2: window covers u[3], u[4], and t + n = 5 = T, so no value term.
    got = nstep_target(U, V, 3, 2, 0.5)
    want = 0.0 + 0.5 * 1.0
    assert abs(got - want) < 1e-12, f"expected {want}, got {got}"


def test_no_bootstrap_when_the_window_passes_the_end():
    # t = 4, n = 10: only u[4] remains, and the episode has ended.
    got = nstep_target(U, V, 4, 10, 0.9)
    assert abs(got - 1.0) < 1e-12, f"expected 1.0, got {got}"


def test_all_targets_match_single_targets():
    got = np.asarray(nstep_targets(U, V, 2, 0.5))
    want = np.array([nstep_target(U, V, t, 2, 0.5) for t in range(5)])
    assert got.shape == (5,), f"expected shape (5,), got {got.shape}"
    assert np.allclose(got, want), f"expected {want}, got {got}"
    assert np.allclose(got, [1.0 + 0.0 + 0.25 * 3.0, 0.0 + 1.0 + 0.25 * 2.0, 2.0 + 0.0 + 0.25 * 1.0, 0.0 + 0.5, 1.0]), f"page values differ, got {got}"


def test_gamma_one_and_zero_rewards_give_the_value():
    got = nstep_target([0.0] * 4, [9.0, 8.0, 7.0, 6.0], 0, 2, 1.0)
    assert abs(got - 7.0) < 1e-12, f"expected 7.0, got {got}"
