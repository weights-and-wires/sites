import numpy as np
from submission import euler_sample, exact_endpoint, gaussian_field, midpoint_sample

X0 = np.random.default_rng(0).normal(size=(200, 1))
M, S, SM = 2.0, 0.5, 0.0


def err(solver, n):
    out = solver(gaussian_field(M, S, SM), X0, n)
    return float(np.sqrt(np.mean((out - exact_endpoint(X0, M, S, SM)) ** 2)))


def test_constant_field_is_exact():
    x0 = np.array([[1.0, -1.0]])
    c = np.array([[3.0, 2.0]])
    for solver in (euler_sample, midpoint_sample):
        got = solver(lambda x, t: c + 0 * x, x0, 4)
        assert np.allclose(got, x0 + c), f"{solver.__name__}: expected {x0 + c}, got {got}"


def test_euler_uses_the_time_at_the_start_of_the_step():
    x0 = np.zeros((1, 1))
    one = euler_sample(lambda x, t: np.full_like(x, t), x0, 1)
    two = euler_sample(lambda x, t: np.full_like(x, t), x0, 2)
    assert np.allclose(one, 0.0), f"v = t with one step: expected 0, got {one}"
    assert np.allclose(two, 0.25), f"v = t with two steps: expected 0.25, got {two}"


def test_midpoint_is_exact_for_a_field_linear_in_time():
    x0 = np.zeros((1, 1))
    got = midpoint_sample(lambda x, t: np.full_like(x, t), x0, 1)
    assert np.allclose(got, 0.5), f"v = t with one midpoint step: expected 0.5, got {got}"


def test_midpoint_evaluates_at_the_half_step_point():
    # dx/dt = x from x0 = 1: one midpoint step gives 1 + (1 + 0.5) = 2.5.
    got = midpoint_sample(lambda x, t: x, np.ones((1, 1)), 1)
    assert np.allclose(got, 2.5), f"expected 1 + 1.5 = 2.5, got {got}"
    got_e = euler_sample(lambda x, t: x, np.ones((1, 1)), 1)
    assert np.allclose(got_e, 2.0), f"one Euler step of dx/dt = x from 1 should give 2.0, got {got_e}"


def test_does_not_modify_the_input():
    x0 = np.ones((3, 2))
    before = x0.copy()
    euler_sample(lambda x, t: x, x0, 5)
    midpoint_sample(lambda x, t: x, x0, 5)
    assert np.array_equal(x0, before), f"x0 must not change; before {before}, after {x0}"


def test_more_steps_gives_a_smaller_error():
    e5, e50 = err(euler_sample, 5), err(euler_sample, 50)
    assert e50 < e5 / 5, f"50-step Euler error {e50:.5f} should be far below the 5-step error {e5:.5f}"


def test_euler_is_first_order():
    ratio = err(euler_sample, 20) / err(euler_sample, 40)
    assert 1.7 < ratio < 2.3, f"halving h should halve the Euler error; ratio of errors was {ratio:.2f}"


def test_midpoint_is_at_least_second_order():
    ratio = err(midpoint_sample, 10) / err(midpoint_sample, 20)
    assert ratio > 3.3, f"halving h should cut the midpoint error by 4 or more; ratio of errors was {ratio:.2f}"


def test_midpoint_beats_euler_at_equal_field_calls():
    e = err(euler_sample, 20)
    m = err(midpoint_sample, 10)
    assert m < e, f"10 midpoint steps (20 calls) error {m:.5f} should beat 20 Euler steps error {e:.5f}"
