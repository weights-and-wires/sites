import numpy as np
from submission import adamax, adamax_step


def run(grads, alpha=0.002):
    theta, m, u = np.zeros(1), np.zeros(1), np.zeros(1)
    steps, us = [], []
    for t, g in enumerate(grads, start=1):
        new, m, u = adamax_step(theta, np.array([g]), m, u, t, alpha=alpha)
        steps.append(float((theta - new)[0] / alpha))
        us.append(float(u[0]))
        theta = new
    return steps, us


def test_page_trace():
    steps, us = run([2.0, -1.0, 0.5, 4.0])
    assert np.allclose(steps, [1.0, 0.2107, 0.2255, 0.3706], atol=1e-4), f"expected step / alpha [1.0, 0.2107, 0.2255, 0.3706], got {np.round(steps, 4)}"
    assert np.allclose(us, [2.0, 1.998, 1.996, 4.0], atol=1e-6), f"expected u [2.0, 1.998, 1.996, 4.0], got {us}"


def test_first_step_is_alpha_times_sign():
    theta, m, u = adamax_step(np.zeros(2), np.array([3.0, -0.2]), np.zeros(2), np.zeros(2), 1, alpha=0.01)
    assert np.allclose(theta, [-0.01, 0.01]), f"expected [-0.01, 0.01], got {theta}"


def test_u_uses_the_absolute_value():
    _, _, u = adamax_step(np.zeros(1), np.array([-4.0]), np.zeros(1), np.array([1.0]), 2)
    assert np.allclose(u, [4.0]), f"expected u = max(0.999, 4) = 4, got {u}"


def test_u_is_a_decayed_maximum_not_an_average():
    _, _, u = adamax_step(np.zeros(1), np.array([0.5]), np.zeros(1), np.array([2.0]), 3)
    assert np.allclose(u, [1.998]), f"expected u = max(0.999 * 2, 0.5) = 1.998, got {u}"


def test_steps_never_exceed_alpha():
    rng = np.random.default_rng(0)
    steps, _ = run(rng.normal(size=500) * rng.choice([0.01, 1, 100], size=500))
    assert max(abs(s) for s in steps) <= 1 + 1e-9, f"expected |step| <= alpha, got max {max(abs(s) for s in steps):.4f} * alpha"


def test_inputs_are_not_modified():
    theta, g, m, u = np.ones(2), np.array([1.0, -2.0]), np.full(2, 0.3), np.full(2, 1.5)
    copies = [a.copy() for a in (theta, g, m, u)]
    adamax_step(theta, g, m, u, 4)
    for name, before, after in zip("theta g m u".split(), copies, (theta, g, m, u)):
        assert np.array_equal(before, after), f"{name} was modified: before {before}, after {after}"


def test_minimizes_a_quadratic():
    out = adamax(lambda x: 2 * x, np.array([3.0, -2.0]), 400, alpha=0.1)
    assert np.all(np.abs(out) < 0.05), f"expected both coordinates within 0.05 of 0, got {out}"
