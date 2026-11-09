import numpy as np
from submission import adam, adam_step


def test_first_step_is_alpha_times_sign():
    g = np.array([2.0, -3.0, 0.5])
    theta, m, v = adam_step(np.zeros(3), g, np.zeros(3), np.zeros(3), 1, alpha=0.01)
    want = -0.01 * np.sign(g)
    assert np.allclose(theta, want, atol=1e-8), f"expected {want}, got {theta}"
    assert np.allclose(m, 0.1 * g), f"expected m = (1 - beta1) * g = {0.1 * g}, got {m}"
    assert np.allclose(v, 0.001 * g * g), f"expected v = (1 - beta2) * g^2 = {0.001 * g * g}, got {v}"


def test_three_step_trace_from_the_page():
    theta, m, v = np.array([0.0]), np.zeros(1), np.zeros(1)
    got = []
    for t, g in enumerate([2.0, -1.0, 1.0], start=1):
        new, m, v = adam_step(theta, np.array([g]), m, v, t)
        got.append(float((theta - new)[0] / 1e-3))
        theta = new
    want = [1.0, 0.2663, 0.4489]
    assert np.allclose(got, want, atol=1e-4), f"expected step / alpha {want}, got {np.round(got, 4)}"
    assert np.allclose(m, 0.1720, atol=1e-4), f"expected m_3 = 0.1720, got {m}"
    assert np.allclose(v, 0.005991, atol=1e-6), f"expected v_3 = 0.005991, got {v}"


def test_epsilon_is_outside_the_square_root():
    g = np.array([1e-6])
    theta, _, _ = adam_step(np.zeros(1), g, np.zeros(1), np.zeros(1), 1)
    step = float(-theta[0] / 1e-3)
    want = 1e-6 / (1e-6 + 1e-8)
    assert abs(step - want) < 1e-6, f"expected step / alpha = {want:.6f} (eps added after the sqrt), got {step:.6f}"


def test_inputs_are_not_modified():
    theta, g, m, v = np.ones(3), np.array([1.0, 2.0, 3.0]), np.full(3, 0.5), np.full(3, 0.25)
    copies = [a.copy() for a in (theta, g, m, v)]
    adam_step(theta, g, m, v, 5)
    for name, before, after in zip("theta g m v".split(), copies, (theta, g, m, v)):
        assert np.array_equal(before, after), f"{name} was modified: before {before}, after {after}"


def test_matches_a_late_step():
    theta, g, m, v = np.array([1.0]), np.array([0.2]), np.array([0.1]), np.array([0.05])
    new, m_new, v_new = adam_step(theta, g, m, v, 100)
    m_ref = 0.9 * 0.1 + 0.1 * 0.2
    v_ref = 0.999 * 0.05 + 0.001 * 0.04
    want = 1.0 - 1e-3 * (m_ref / (1 - 0.9**100)) / (np.sqrt(v_ref / (1 - 0.999**100)) + 1e-8)
    assert np.allclose(new, want, atol=1e-12), f"expected {want}, got {new}"


def test_minimizes_a_quadratic():
    out = adam(lambda x: 2 * x, np.array([3.0, -2.0]), 300, alpha=0.1)
    assert np.all(np.abs(out) < 0.05), f"expected both coordinates within 0.05 of 0, got {out}"


def test_per_parameter_step_ignores_gradient_scale():
    grad = lambda x: np.array([1.0, 1000.0]) * (x - 1.0)
    out = adam(grad, np.zeros(2), 5, alpha=0.1)
    assert abs(out[0] - out[1]) < 1e-6, f"both coordinates take the same steps, got {out}"
    assert out[0] > 0.4, f"expected about 5 steps of 0.1 toward 1, got {out}"
