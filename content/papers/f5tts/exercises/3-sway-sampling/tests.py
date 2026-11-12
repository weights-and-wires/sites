import numpy as np
from submission import sway, sway_steps


def test_s_zero_is_uniform():
    u = np.linspace(0, 1, 9)
    got = sway(u, 0.0)
    assert np.allclose(got, u), f"expected {u}, got {got}"


def test_s_minus_one_closed_form():
    u = np.linspace(0, 1, 11)
    want = 1 - np.cos(np.pi / 2 * u)
    got = sway(u, -1.0)
    assert np.allclose(got, want), f"expected 1 - cos(pi/2 u) = {want}, got {got}"


def test_page_values_for_eight_steps():
    got = sway_steps(8, -1.0)
    want = np.array([0.0, 0.0192, 0.0761, 0.1685, 0.2929, 0.4444, 0.6173, 0.8049, 1.0])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {np.round(got, 4)}"


def test_positive_coefficient_at_the_upper_bound():
    s = 2 / (np.pi - 2)
    got = sway_steps(8, s)
    want = np.array([0.0, 0.3103, 0.5546, 0.7367, 0.8628, 0.9413, 0.9825, 0.9978, 1.0])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {np.round(got, 4)}"


def test_endpoints_and_length():
    for s in (-1.0, -0.4, 0.5, 1.7):
        t = sway_steps(16, s)
        assert len(t) == 17, f"expected 17 boundaries for NFE = 16, got {len(t)}"
        assert abs(t[0]) < 1e-12 and abs(t[-1] - 1) < 1e-12, f"s = {s}: expected endpoints 0 and 1, got {t[0]} and {t[-1]}"


def test_monotonic_in_the_allowed_range():
    for s in (-1.0, -0.8, 0.5, 2 / (np.pi - 2)):
        t = sway_steps(64, s)
        assert np.all(np.diff(t) >= -1e-12), f"s = {s}: steps must not go backwards"


def test_negative_s_puts_more_steps_near_zero():
    uni = sway_steps(16, 0.0)
    neg = sway_steps(16, -1.0)
    pos = sway_steps(16, 0.5)
    n_uni, n_neg, n_pos = (int((t[:-1] < 0.5).sum()) for t in (uni, neg, pos))
    assert (n_uni, n_neg, n_pos) == (8, 11, 7), f"expected steps below 0.5 of (8, 11, 7) for s = 0, -1, 0.5; got {(n_uni, n_neg, n_pos)}"
    assert abs(np.diff(neg)[0] - 0.0048) < 1e-4, f"expected first step about 0.0048, got {np.diff(neg)[0]:.4f}"


def test_scalar_input():
    got = float(sway(0.5, -1.0))
    assert abs(got - (1 - np.cos(np.pi / 4))) < 1e-12, f"expected 0.2929, got {got}"


def test_out_of_range_raises():
    for bad in (-1.2, 2.0):
        try:
            sway_steps(8, bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for s = {bad}")
    sway_steps(8, -1.0)
    sway_steps(8, 2 / (np.pi - 2))
