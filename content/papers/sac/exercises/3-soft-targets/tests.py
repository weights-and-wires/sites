import numpy as np
from submission import half_mse, q_target, v_target

q1 = np.array([1.0, 0.5, 2.0])
q2 = np.array([0.8, 0.9, 1.5])
logp = np.array([-1.2, -0.3, -0.5])
r = np.array([1.0, 0.0, 2.0])
v_next = np.array([3.0, 2.0, 1.0])
done = np.array([0.0, 0.0, 1.0])


def test_value_target_on_page_example():
    got = np.asarray(v_target(q1, q2, logp))
    want = np.array([2.0, 0.8, 2.0])
    assert got.shape == (3,), f"expected shape (3,), got {got.shape}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_value_target_takes_the_smaller_critic_per_row():
    got = np.asarray(v_target(np.array([5.0, 0.0]), np.array([1.0, 3.0]), np.array([0.0, 0.0])))
    assert np.allclose(got, [1.0, 0.0]), f"expected the row-wise minimum [1, 0], got {got}"
    swapped = np.asarray(v_target(np.array([1.0, 3.0]), np.array([5.0, 0.0]), np.array([0.0, 0.0])))
    assert np.allclose(swapped, [1.0, 0.0]), f"the order of the critics must not matter, got {swapped}"


def test_entropy_bonus_sign():
    low = np.asarray(v_target(np.array([1.0]), np.array([1.0]), np.array([-2.0])))
    high = np.asarray(v_target(np.array([1.0]), np.array([1.0]), np.array([2.0])))
    assert np.allclose(low, [3.0]), f"logp = -2 must add 2: expected 3.0, got {low}"
    assert np.allclose(high, [-1.0]), f"logp = 2 must subtract 2: expected -1.0, got {high}"


def test_q_target_on_page_example():
    got = np.asarray(q_target(r, v_next, done, 0.99))
    want = np.array([3.97, 1.98, 2.0])
    assert got.shape == (3,), f"expected shape (3,), got {got.shape}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_terminal_rows_ignore_the_value_network():
    got = np.asarray(q_target(np.array([2.0]), np.array([1e6]), np.array([1.0]), 0.99))
    assert np.allclose(got, [2.0]), f"a terminal row must give r = 2.0, got {got}"
    got = np.asarray(q_target(np.array([2.0]), np.array([10.0]), np.array([0.0]), 0.5))
    assert np.allclose(got, [7.0]), f"a live row must bootstrap: 2 + 0.5 * 10 = 7, got {got}"


def test_gamma_scales_only_the_bootstrap():
    got = np.asarray(q_target(np.array([1.0, 1.0]), np.array([4.0, 4.0]), np.array([0.0, 0.0]), 0.25))
    assert np.allclose(got, [2.0, 2.0]), f"expected 1 + 0.25 * 4 = 2, got {got}"


def test_loss_values():
    v = np.array([1.5, 0.4, 1.9])
    got = half_mse(v, np.array([2.0, 0.8, 2.0]))
    assert np.isclose(got, 0.07), f"page example: expected 0.07, got {got}"
    got = half_mse(np.array([0.0, 0.0]), np.array([2.0, -1.0]))
    assert np.isclose(got, 1.25), f"errors 2 and -1 give (2 + 0.5) / 2 = 1.25, got {got}"


def test_loss_is_zero_at_the_target_and_symmetric():
    t = np.array([1.0, -2.0, 3.0])
    assert np.isclose(half_mse(t, t), 0.0), "the loss must be 0 when pred equals target"
    a = half_mse(t + 1.0, t)
    b = half_mse(t - 1.0, t)
    assert np.isclose(a, b) and np.isclose(a, 0.5), f"expected 0.5 both ways, got {a} and {b}"
