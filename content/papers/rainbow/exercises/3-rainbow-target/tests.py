import numpy as np
from submission import kl_priority, rainbow_target

z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
p_online = np.array([[0.05, 0.15, 0.30, 0.30, 0.20], [0.10, 0.30, 0.40, 0.15, 0.05]])
p_target = np.array([[0.30, 0.30, 0.20, 0.10, 0.10], [0.05, 0.10, 0.20, 0.35, 0.30]])


def test_page_example():
    m, a = rainbow_target(p_online, p_target, z, [1.0, 0.0, 0.0], False, 0.9)
    want = np.array([0.0, 0.1374, 0.3813, 0.3084, 0.1729])
    assert a == 0, f"the online means are [0.45, -0.25], so a_star = 0, got {a}"
    assert np.allclose(m, want, atol=1e-4), f"expected {want}, got {np.asarray(m)}"


def test_online_picks_and_target_scores():
    m, a = rainbow_target(p_online, p_target, z, [1.0, 0.0, 0.0], False, 0.9)
    wrong = np.array([0.0, 0.0229, 0.1, 0.3219, 0.5552])
    assert not np.allclose(m, wrong, atol=1e-3), "the target network chose the action (that is C51, not the Rainbow target)"
    swapped_m, swapped_a = rainbow_target(p_target, p_online, z, [1.0, 0.0, 0.0], False, 0.9)
    assert swapped_a == 1, f"with the roles swapped the online means are [-0.6, 0.75], so a_star = 1, got {swapped_a}"


def test_probabilities_stay_a_distribution():
    m, _ = rainbow_target(p_online, p_target, z, [1.0, 0.0, 0.0], False, 0.9)
    assert np.isclose(np.sum(m), 1.0), f"target should sum to 1, got {np.sum(m)}"


def test_terminal_window_collapses_onto_the_return():
    m, _ = rainbow_target(p_online, p_target, z, [1.0, 1.0], True, 0.9)
    want = np.array([0.0, 0.0, 0.0, 0.1, 0.9])
    assert np.allclose(m, want), f"the return is 1.9, so mass splits 0.1 / 0.9 between atoms 3 and 4: expected {want}, got {np.asarray(m)}"


def test_discount_uses_window_length():
    one = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
    m, _ = rainbow_target(one, one, z, [0.0, 0.0], False, 0.5)
    assert np.allclose(m, [0.0, 0.0, 1.0, 0.0, 0.0]), f"atom 0 maps to 0 whatever the discount, got {np.asarray(m)}"
    two = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
    m, _ = rainbow_target(two, two, z, [0.0, 0.0], False, 0.5)
    want = np.array([0.0, 0.0, 0.5, 0.5, 0.0])
    assert np.allclose(m, want), f"the atom at 2 moves to 0.25 * 2 = 0.5 after two steps: expected {want}, got {np.asarray(m)}"


def test_kl_priority_value():
    m = np.array([0.0, 0.1374, 0.3813, 0.3084, 0.1729])
    p = np.array([0.10, 0.20, 0.30, 0.25, 0.15])
    got = kl_priority(m, p, 0.5)
    assert np.isclose(got, 0.3594, atol=2e-3), f"expected 0.3594, got {got}"
    got1 = kl_priority(m, p, 1.0)
    assert np.isclose(got1, 0.1292, atol=2e-3), f"omega = 1 gives the KL itself, 0.1292, got {got1}"


def test_kl_is_zero_for_a_perfect_prediction():
    m = np.array([0.0, 0.5, 0.5, 0.0, 0.0])
    got = kl_priority(m, m + 1e-300, 0.5)
    assert abs(got) < 1e-6, f"a prediction equal to the target has priority 0, got {got}"


def test_kl_ignores_zero_target_mass():
    m = np.array([0.0, 1.0, 0.0])
    p = np.array([0.0, 0.5, 0.5])
    got = kl_priority(m, p, 1.0)
    assert np.isclose(got, np.log(2.0)), f"expected log 2 = 0.6931 (0 log 0 is 0), got {got}"


def test_kl_direction():
    m = np.array([0.9, 0.1])
    p = np.array([0.5, 0.5])
    got = kl_priority(m, p, 1.0)
    want = 0.9 * np.log(1.8) + 0.1 * np.log(0.2)
    assert np.isclose(got, want), f"KL(m || p) weights by m: expected {want}, got {got}"
