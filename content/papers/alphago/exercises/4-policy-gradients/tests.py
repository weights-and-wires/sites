import numpy as np
from submission import rl_grad, sl_grad, softmax


def log_prob(x, a):
    p = softmax(x)
    return np.log(p[a])


def numeric_grad(x, a, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = eps
        g[i] = (log_prob(x + d, a) - log_prob(x - d, a)) / (2 * eps)
    return g


def test_softmax_sums_to_one_and_is_stable():
    p = np.asarray(softmax(np.array([1000.0, 1000.0, 999.0])))
    assert np.all(np.isfinite(p)), f"softmax overflowed on large logits, got {p}"
    assert np.isclose(p.sum(), 1.0), f"probabilities must sum to 1, got {p.sum()}"
    assert np.allclose(p, [0.4223, 0.4223, 0.1554], atol=1e-3), f"expected [0.4223, 0.4223, 0.1554], got {p}"


def test_sl_grad_on_page_example():
    g = np.asarray(sl_grad(np.zeros(3), 1))
    want = np.array([-1 / 3, 2 / 3, -1 / 3])
    assert np.allclose(g, want), f"expected {want}, got {g}"


def test_sl_grad_matches_finite_differences():
    x = np.array([0.5, -1.0, 2.0, 0.1])
    for a in range(4):
        got = np.asarray(sl_grad(x, a))
        want = numeric_grad(x, a)
        assert np.allclose(got, want, atol=1e-5), f"action {a}: expected {want}, got {got}"


def test_sl_grad_sums_to_zero():
    g = np.asarray(sl_grad(np.array([0.3, 0.2, -0.7]), 2))
    assert np.isclose(g.sum(), 0.0), f"probabilities sum to 1, so the gradient sums to 0, got {g.sum()}"


def test_rl_grad_sign_follows_the_outcome():
    x = np.array([0.5, -1.0, 2.0])
    win = np.asarray(rl_grad(x, 1, 1.0))
    loss = np.asarray(rl_grad(x, 1, -1.0))
    assert win[1] > 0, f"a win must raise the played move's logit, got {win}"
    assert loss[1] < 0, f"a loss must lower the played move's logit, got {loss}"
    assert np.allclose(win, -loss), f"win and loss gradients must be opposite, got {win} and {loss}"


def test_rl_grad_with_zero_baseline_equals_z_times_sl():
    x = np.array([0.5, -1.0, 2.0])
    got = np.asarray(rl_grad(x, 0, -1.0))
    want = -np.asarray(sl_grad(x, 0))
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_baseline_is_subtracted_from_z():
    x = np.array([0.0, 0.0])
    got = np.asarray(rl_grad(x, 0, 1.0, baseline=0.25))
    want = 0.75 * np.array([0.5, -0.5])
    assert np.allclose(got, want), f"expected (1 - 0.25) * [0.5, -0.5] = {want}, got {got}"
    got = np.asarray(rl_grad(x, 0, 1.0, baseline=1.0))
    assert np.allclose(got, [0.0, 0.0]), f"when the baseline equals z there is no update, got {got}"


def test_ascent_step_raises_probability_of_a_won_move():
    x = np.array([0.0, 0.0, 0.0])
    before = softmax(x)[2]
    x2 = x + 0.5 * np.asarray(rl_grad(x, 2, 1.0))
    after = softmax(x2)[2]
    assert after > before, f"a gradient-ascent step after a win must raise p, got {before} then {after}"
