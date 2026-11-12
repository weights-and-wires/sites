import numpy as np
from submission import az_grads, az_loss

LOGITS = np.array([1.0, 0.5, 0.0, -1.0])
PI = np.array([0.6, 0.3, 0.1, 0.0])


def test_page_example_total():
    got = az_loss(LOGITS, 0.3, PI, -1.0, np.full(50, 1.0), 1e-4)
    assert abs(got - 2.6916) < 1e-3, f"expected 2.6916 (1.6900 + 0.9966 + 0.0050), got {got}"


def test_value_term_is_squared_and_signed_by_z():
    base = az_loss(LOGITS, 0.0, PI, 0.0, np.zeros(1), 0.0)
    got = az_loss(LOGITS, 0.5, PI, 0.0, np.zeros(1), 0.0) - base
    assert abs(got - 0.25) < 1e-9, f"expected (0 - 0.5)^2 = 0.25 added to the loss, got {got}"
    got = az_loss(LOGITS, 0.5, PI, 1.0, np.zeros(1), 0.0) - base
    assert abs(got - 0.25) < 1e-9, f"expected (1 - 0.5)^2 = 0.25 added to the loss, got {got}"
    got = az_loss(LOGITS, -0.5, PI, 1.0, np.zeros(1), 0.0) - base
    assert abs(got - 2.25) < 1e-9, f"expected (1 + 0.5)^2 = 2.25 added to the loss, got {got}"


def test_policy_term_is_cross_entropy():
    uniform = az_loss(np.zeros(4), 0.0, np.array([1.0, 0, 0, 0]), 0.0, np.zeros(1), 0.0)
    assert abs(uniform - np.log(4)) < 1e-9, f"expected -log(1/4) = {np.log(4):.4f}, got {uniform}"
    # minimized when p == pi, with floor equal to the entropy of pi
    logits = np.log(PI + 1e-300)
    floor = az_loss(logits, 0.0, PI, 0.0, np.zeros(1), 0.0)
    entropy = -float(np.sum(PI[PI > 0] * np.log(PI[PI > 0])))
    assert abs(floor - entropy) < 1e-6, f"expected the entropy of pi = {entropy:.4f} when p == pi, got {floor}"
    assert az_loss(LOGITS, 0.0, PI, 0.0, np.zeros(1), 0.0) > floor, "a mismatched p must cost more than p == pi"


def test_l2_term():
    theta = np.array([3.0, 4.0])
    d = az_loss(LOGITS, 0.0, PI, 0.0, theta, 0.01) - az_loss(LOGITS, 0.0, PI, 0.0, theta, 0.0)
    assert abs(d - 0.25) < 1e-9, f"expected c * ||theta||^2 = 0.01 * 25 = 0.25, got {d}"


def test_zero_target_with_underflowed_probability_is_finite():
    got = az_loss(np.array([0.0, -2000.0]), 0.0, np.array([1.0, 0.0]), 0.0, np.zeros(1), 0.0)
    assert np.isfinite(got), f"expected a finite loss (use log-softmax, not log(softmax)), got {got}"
    assert abs(got - np.log(1 + np.exp(-2000.0))) < 1e-9, f"expected about 0, got {got}"
    big = az_loss(np.array([1000.0, 999.0]), 0.0, np.array([0.5, 0.5]), 0.0, np.zeros(1), 0.0)
    assert np.isfinite(big), f"expected a finite loss for logits near 1000, got {big}"


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=5)
    pi = rng.dirichlet(np.ones(5))
    theta = rng.normal(size=7)
    v, z, c = 0.2, -1.0, 0.05
    g_logits, g_v, g_theta = az_grads(logits, v, pi, z, theta, c)
    eps = 1e-6
    for i in range(5):
        d = np.zeros(5)
        d[i] = eps
        fd = (az_loss(logits + d, v, pi, z, theta, c) - az_loss(logits - d, v, pi, z, theta, c)) / (2 * eps)
        assert abs(g_logits[i] - fd) < 1e-6, f"d l / d logits[{i}]: expected {fd:.6f}, got {g_logits[i]:.6f}"
    fd = (az_loss(logits, v + eps, pi, z, theta, c) - az_loss(logits, v - eps, pi, z, theta, c)) / (2 * eps)
    assert abs(g_v - fd) < 1e-6, f"d l / d v: expected {fd:.6f}, got {g_v:.6f}"
    for i in range(7):
        d = np.zeros(7)
        d[i] = eps
        fd = (az_loss(logits, v, pi, z, theta + d, c) - az_loss(logits, v, pi, z, theta - d, c)) / (2 * eps)
        assert abs(g_theta[i] - fd) < 1e-6, f"d l / d theta[{i}]: expected {fd:.6f}, got {g_theta[i]:.6f}"


def test_page_gradient_values():
    g_logits, g_v, _ = az_grads(LOGITS, 0.3, PI, -1.0, np.zeros(2), 0.0)
    assert np.allclose(g_logits, [-0.1260, -0.0125, 0.0744, 0.0641], atol=1e-4), f"expected p - pi = [-0.1260, -0.0125, 0.0744, 0.0641], got {g_logits}"
    assert abs(g_v - 2.6) < 1e-9, f"expected -2 (z - v) = 2.6, got {g_v}"
