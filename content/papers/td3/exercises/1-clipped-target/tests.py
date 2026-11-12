import numpy as np
from submission import clipped_target, twin_critic_loss

gamma = 0.9
r = np.array([1.0, 0.0, 2.0])
done = np.array([0.0, 0.0, 1.0])
q1 = np.array([2.0, 5.0, -1.0])
q2 = np.array([3.0, 4.0, -2.0])


def test_page_example():
    y = np.asarray(clipped_target(r, done, gamma, q1, q2))
    want = np.array([2.8, 3.6, 2.0])
    assert y.shape == (3,), f"expected shape (3,), got {y.shape}"
    assert np.allclose(y, want), f"expected {want}, got {y}"


def test_uses_the_smaller_critic_on_each_row():
    y = np.asarray(clipped_target(np.zeros(2), np.zeros(2), 1.0, np.array([1.0, 9.0]), np.array([4.0, 2.0])))
    assert np.allclose(y, [1.0, 2.0]), f"expected the row-wise minimum [1, 2], got {y}"


def test_is_symmetric_in_the_two_critics():
    a = np.asarray(clipped_target(r, done, gamma, q1, q2))
    b = np.asarray(clipped_target(r, done, gamma, q2, q1))
    assert np.allclose(a, b), f"swapping the critics must not change y: {a} vs {b}"


def test_is_not_the_mean_or_the_max():
    y = np.asarray(clipped_target(np.zeros(1), np.zeros(1), 1.0, np.array([0.0]), np.array([10.0])))
    assert np.allclose(y, [0.0]), f"min(0, 10) = 0, got {y}"


def test_terminal_rows_ignore_the_critics():
    big = np.full(3, 1e6)
    y = np.asarray(clipped_target(r, done, gamma, big, big))
    assert np.isclose(y[2], 2.0), f"a terminal row must give y = r = 2.0, got {y[2]}"
    assert np.isclose(y[0], 1.0 + gamma * 1e6), f"a non-terminal row must bootstrap, got {y[0]}"


def test_gamma_scales_only_the_bootstrap():
    y = np.asarray(clipped_target(np.array([1.0]), np.array([0.0]), 0.5, np.array([4.0]), np.array([6.0])))
    assert np.allclose(y, [3.0]), f"expected 1 + 0.5 * 4 = 3.0, got {y}"


def test_negative_values_are_kept():
    y = np.asarray(clipped_target(np.array([0.0]), np.array([0.0]), 1.0, np.array([-3.0]), np.array([-1.0])))
    assert np.allclose(y, [-3.0]), f"expected -3.0, got {y}"


def test_twin_loss_on_page_example():
    y = clipped_target(r, done, gamma, q1, q2)
    got = twin_critic_loss(np.array([2.5, 3.0, 1.5]), np.array([2.0, 3.5, 2.5]), y)
    assert np.isclose(got, 0.533333, atol=1e-4), f"expected 0.53333, got {got}"


def test_twin_loss_sums_two_means():
    got = twin_critic_loss(np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert np.isclose(got, 5.0), f"errors of 2 and 1 give 4 + 1 = 5, got {got}"
    got2 = twin_critic_loss(np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0, 1.0]))
    assert np.isclose(got2, 2.0), f"two critics with mean error 1 each give 2, got {got2}"
