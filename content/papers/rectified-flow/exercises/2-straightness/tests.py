import math

import numpy as np
from submission import straightness


def grid(n_steps):
    return np.linspace(0.0, 1.0, n_steps + 1)[:, None, None]


def test_constant_speed_line_is_exactly_straight():
    t = grid(20)
    start = np.array([[[1.0, -2.0], [0.0, 3.0]]])
    end = np.array([[[4.0, 2.0], [-5.0, 3.0]]])
    path = start + t * (end - start)
    got = straightness(path)
    assert abs(got) < 1e-12, f"expected 0 for constant-speed lines, got {got}"


def test_returns_a_float():
    got = straightness(grid(5) * np.ones((1, 1, 2)))
    assert isinstance(got, float), f"expected a Python float, got {type(got)}"


def test_straight_line_at_changing_speed_is_not_zero():
    # Z_t = t^2 (3, 4): S = |(3, 4)|^2 / 3 = 25 / 3.
    path = grid(200) ** 2 * np.array([[[3.0, 4.0]]])
    got = straightness(path)
    assert abs(got - 25 / 3) < 1e-3, f"expected {25 / 3:.4f}, got {got:.4f}"


def test_half_circle():
    # From (0, 0) to (2, 0) along (1 - cos(pi t), sin(pi t)): S = pi^2 - 4.
    t = grid(400)
    path = np.concatenate([1 - np.cos(math.pi * t), np.sin(math.pi * t)], axis=-1)
    got = straightness(path)
    want = math.pi**2 - 4
    assert abs(got - want) < 5e-3, f"expected {want:.4f}, got {got:.4f}"


def test_value_does_not_depend_on_the_number_of_steps():
    delta = np.array([[[3.0, 4.0]]])
    s50 = straightness(grid(50) ** 2 * delta)
    s400 = straightness(grid(400) ** 2 * delta)
    assert abs(s50 - s400) < 0.02, f"S should be a time integral: 50 steps gave {s50:.4f}, 400 steps gave {s400:.4f}"


def test_averages_over_trajectories():
    delta = np.array([[[3.0, 4.0], [3.0, 4.0]]])
    t = grid(200)
    path = np.concatenate([t * delta[:, :1], t**2 * delta[:, 1:]], axis=1)
    got = straightness(path)
    assert abs(got - 25 / 6) < 1e-3, f"one straight and one curved trajectory: expected {25 / 6:.4f}, got {got:.4f}"


def test_sums_over_coordinates():
    # The squared norm adds the coordinates: the same curve in two dimensions has twice the S.
    one = straightness(grid(200) ** 2 * np.array([[[5.0]]]))
    two = straightness(grid(200) ** 2 * np.array([[[5.0, 5.0]]]))
    assert abs(two - 2 * one) < 1e-6, f"expected twice {one:.4f}, got {two:.4f}"
    assert abs(one - 25 / 3) < 1e-3, f"1-D value: expected {25 / 3:.4f}, got {one:.4f}"
