import numpy as np
from submission import log_sigmoid, discriminator_loss, generator_loss, generator_grad


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def numeric_grad(f, a, eps=1e-6):
    g = np.zeros_like(a)
    for i in range(a.size):
        up, down = a.copy(), a.copy()
        up[i] += eps
        down[i] -= eps
        g[i] = (f(up) - f(down)) / (2 * eps)
    return g


def test_log_sigmoid_values():
    got = log_sigmoid(np.array([0.0, 2.0, -2.0]))
    want = np.log(sigmoid(np.array([0.0, 2.0, -2.0])))
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_log_sigmoid_is_stable():
    got = log_sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(got)), f"log_sigmoid([-1000, 1000]) = {got}; use logaddexp"
    assert np.allclose(got, [-1000.0, 0.0]), f"expected [-1000, 0], got {got}"


def test_discriminator_loss_value():
    real, fake = np.array([2.0, 0.0]), np.array([-1.0, 2.0])
    want = -np.mean(np.log(sigmoid(real))) - np.mean(np.log(1 - sigmoid(fake)))
    got = discriminator_loss(real, fake)
    assert np.isclose(got, want), f"expected {want:.5f}, got {got:.5f}; fake logits must enter as log(1 - D)"


def test_discriminator_loss_at_chance():
    got = discriminator_loss(np.zeros(5), np.zeros(5))
    assert np.isclose(got, 2 * np.log(2)), f"expected 2 log 2 = {2 * np.log(2):.4f} when D = 1/2 everywhere, got {got}"


def test_generator_loss_values():
    a = np.array([-2.0, 1.0, 3.0])
    want_sat = np.mean(np.log(1 - sigmoid(a)))
    want_ns = -np.mean(np.log(sigmoid(a)))
    assert np.isclose(generator_loss(a, False), want_sat), f"saturating: expected {want_sat:.5f}, got {generator_loss(a, False):.5f}"
    assert np.isclose(generator_loss(a, True), want_ns), f"non-saturating: expected {want_ns:.5f}, got {generator_loss(a, True):.5f}"


def test_gradients_match_finite_differences():
    a = np.array([-3.0, -0.5, 0.7, 2.0])
    for ns in (False, True):
        want = numeric_grad(lambda v: generator_loss(v, ns), a)
        got = generator_grad(a, ns)
        assert np.allclose(got, want, atol=1e-6), f"non_saturating={ns}: expected {want}, got {got}"


def test_saturation_when_discriminator_wins():
    a = np.array([-5.0])  # D(G(z)) = 0.0067, so D is nearly sure the sample is fake
    sat = generator_grad(a, False)[0]
    ns = generator_grad(a, True)[0]
    assert abs(sat) < 0.01, f"saturating gradient should be about -0.0067, got {sat}"
    assert abs(ns) > 0.99, f"non-saturating gradient should be about -0.9933, got {ns}"
