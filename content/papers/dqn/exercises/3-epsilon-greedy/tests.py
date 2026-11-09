import numpy as np
from submission import epsilon_at, select_action


def test_schedule_start_and_end():
    assert np.isclose(epsilon_at(0), 1.0), f"epsilon at frame 0 should be 1.0, got {epsilon_at(0)}"
    assert np.isclose(epsilon_at(1_000_000), 0.1), f"epsilon at 1M frames should be 0.1, got {epsilon_at(1_000_000)}"


def test_schedule_is_linear():
    got = [epsilon_at(f) for f in (250_000, 500_000, 750_000)]
    assert np.allclose(got, [0.775, 0.55, 0.325]), f"expected [0.775, 0.55, 0.325], got {got}"


def test_schedule_stays_at_the_floor():
    for f in (1_000_001, 5_000_000, 10_000_000):
        assert np.isclose(epsilon_at(f), 0.1), f"epsilon at frame {f} should stay 0.1, got {epsilon_at(f)}"


def test_schedule_custom_arguments():
    got = epsilon_at(50, eps_start=0.5, eps_end=0.0, anneal_frames=100)
    assert np.isclose(got, 0.25), f"expected 0.25, got {got}"


def test_epsilon_zero_is_greedy():
    rng = np.random.default_rng(0)
    q = np.array([0.1, 0.9, 0.3, 0.2])
    acts = [select_action(q, 0.0, rng) for _ in range(200)]
    assert set(acts) == {1}, f"with epsilon 0 every action must be 1 (the argmax), got {sorted(set(acts))}"


def test_greedy_picks_the_maximum_not_the_minimum():
    rng = np.random.default_rng(0)
    assert select_action(np.array([-5.0, -1.0, -3.0]), 0.0, rng) == 1, "the argmax of [-5, -1, -3] is index 1"


def test_returns_a_python_int_in_range():
    rng = np.random.default_rng(0)
    a = select_action(np.zeros(6), 1.0, rng)
    assert isinstance(a, (int, np.integer)), f"expected an integer, got {type(a)}"
    assert 0 <= a < 6, f"expected an action in [0, 6), got {a}"


def test_epsilon_one_is_uniform_including_the_greedy_action():
    rng = np.random.default_rng(1)
    q = np.array([0.1, 0.9, 0.3, 0.2])
    acts = np.array([select_action(q, 1.0, rng) for _ in range(4000)])
    freq = np.bincount(acts, minlength=4) / 4000
    assert np.all(np.abs(freq - 0.25) < 0.04), f"expected about 0.25 each, got {freq}"


def test_greedy_probability_at_half_epsilon():
    rng = np.random.default_rng(2)
    q = np.array([0.1, 0.9, 0.3, 0.2])
    acts = np.array([select_action(q, 0.5, rng) for _ in range(6000)])
    p_greedy = np.mean(acts == 1)
    assert abs(p_greedy - 0.625) < 0.03, f"expected 0.5 + 0.5/4 = 0.625 for the greedy action, got {p_greedy:.3f}"
