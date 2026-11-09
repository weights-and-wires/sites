import numpy as np
from submission import adam, rosenbrock, rosenbrock_grad, sgd_momentum

START = np.array([-1.5, 1.5])


def test_gradient_is_zero_at_the_minimum():
    got = np.asarray(rosenbrock_grad(np.array([1.0, 1.0])), dtype=float)
    assert np.allclose(got, 0), f"expected [0, 0] at (1, 1), got {got}"


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = rng.normal(size=2)
        h = 1e-6
        want = np.array([(rosenbrock(p + h * e) - rosenbrock(p - h * e)) / (2 * h) for e in np.eye(2)])
        got = np.asarray(rosenbrock_grad(p), dtype=float)
        assert got.shape == (2,), f"expected shape (2,), got {got.shape}"
        assert np.allclose(got, want, rtol=1e-4, atol=1e-4), f"at {p}: expected {want}, got {got}"


def test_gradient_at_the_start_point():
    got = np.asarray(rosenbrock_grad(START), dtype=float)
    want = [-2 * 2.5 - 400 * (-1.5) * (1.5 - 2.25), 200 * (1.5 - 2.25)]
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_momentum_two_steps_by_hand():
    out1 = sgd_momentum(lambda x: x, [1.0], 1, lr=0.1, mu=0.9)
    out2 = sgd_momentum(lambda x: x, [1.0], 2, lr=0.1, mu=0.9)
    assert np.allclose(out1, [0.9]), f"after 1 step expected [0.9], got {out1}"
    assert np.allclose(out2, [0.72]), f"after 2 steps expected [0.72] (v = 0.9 * 1 + 0.9 = 1.8), got {out2}"


def test_momentum_does_not_modify_x0():
    x0 = np.array([1.0, 2.0])
    sgd_momentum(lambda x: x, x0, 3, lr=0.1)
    assert np.array_equal(x0, [1.0, 2.0]), f"x0 was modified, now {x0}"


def test_momentum_converges_with_a_tuned_rate():
    x = sgd_momentum(rosenbrock_grad, START, 2000, lr=1e-3)
    loss = rosenbrock(x)
    assert loss < 1e-6, f"expected f < 1e-6 after 2000 steps at lr 1e-3, got f = {loss:.3e} at {x}"


def test_adam_converges_with_alpha_0_1():
    x = adam(rosenbrock_grad, START, 2000, alpha=0.1)
    loss = rosenbrock(x)
    assert loss < 1e-6, f"expected f < 1e-6, got f = {loss:.3e} at {x}"


def test_rescaling_the_gradient_breaks_momentum_but_not_adam():
    scaled = lambda p: 1000 * rosenbrock_grad(p)
    a = adam(scaled, START, 2000, alpha=0.1)
    assert rosenbrock(a) < 1e-6, f"Adam should ignore the scale; expected f < 1e-6, got {rosenbrock(a):.3e}"
    s = sgd_momentum(scaled, START, 2000, lr=1e-3)
    bad = (not np.all(np.isfinite(s))) or rosenbrock(s) > 1.0
    assert bad, f"momentum with lr 1e-3 should diverge on a 1000x gradient, got f = {rosenbrock(s):.3e}"
