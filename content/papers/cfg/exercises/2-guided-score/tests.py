import numpy as np
from submission import classifier_guided_eps, guided_eps, implicit_classifier_grad

# Two classes with x | c ~ N(mu_c, 1), equal class probabilities, mu = (-2, 2).
# For z = alpha x + sigma eps, z | c ~ N(alpha mu_c, v) with v = alpha^2 + sigma^2.
MU = np.array([-2.0, 2.0])


def _logit_parts(z, lam):
    alpha = np.sqrt(1 / (1 + np.exp(-lam)))
    sigma = np.sqrt(1 - alpha**2)
    v = alpha**2 + sigma**2
    return alpha, sigma, v


def eps_c(z, c, lam):
    alpha, sigma, v = _logit_parts(z, lam)
    return sigma * (z - alpha * MU[c]) / v


def log_posterior(z, c, lam):
    alpha, _, v = _logit_parts(z, lam)
    logp = np.array([-0.5 * (z - alpha * m) ** 2 / v for m in MU])
    return logp[c] - np.logaddexp(logp[0], logp[1])


def eps_u(z, lam):
    alpha, sigma, v = _logit_parts(z, lam)
    r1 = np.exp(log_posterior(z, 1, lam))
    mean = alpha * (r1 * MU[1] + (1 - r1) * MU[0])
    return sigma * (z - mean) / v


def test_hand_value():
    got = guided_eps(0.5, 0.2, 2.0)
    assert np.isclose(got, 1.1), f"expected (1 + 2) * 0.5 - 2 * 0.2 = 1.1, got {got}"


def test_endpoints():
    ec, eu = np.array([0.5, -1.0]), np.array([0.2, 0.4])
    assert np.allclose(guided_eps(ec, eu, 0.0), ec), f"w = 0 must return the conditional prediction {ec}, got {guided_eps(ec, eu, 0.0)}"
    assert np.allclose(guided_eps(ec, eu, -1.0), eu), f"w = -1 must return the unconditional prediction {eu}, got {guided_eps(ec, eu, -1.0)}"


def test_w_is_not_the_scale():
    ec, eu = np.array([1.0]), np.array([0.0])
    got = guided_eps(ec, eu, 1.0)
    assert np.allclose(got, 2.0), f"the paper's w = 1 doubles the conditional score here; expected 2.0, got {got}"


def test_equal_predictions_are_unchanged():
    e = np.array([[0.3, -0.7], [1.0, 0.0]])
    assert np.allclose(guided_eps(e, e, 5.0), e), "when conditional and unconditional agree, guidance must change nothing"


def test_toy_values():
    z, lam = 1.0, 0.0
    ec, eu = eps_c(z, 1, lam), eps_u(z, lam)
    want = {1.0: -0.4045, 2.0: -0.5161, 4.0: -0.7394}
    for w, v in want.items():
        got = guided_eps(ec, eu, w)
        assert np.isclose(got, v, atol=1e-4), f"expected {v} at w = {w}, got {got}"


def test_implicit_gradient_matches_bayes():
    for z, lam in [(1.0, 0.0), (-0.3, 1.5), (2.0, -1.0)]:
        _, sigma, _ = _logit_parts(z, lam)
        h = 1e-5
        fd = (log_posterior(z + h, 1, lam) - log_posterior(z - h, 1, lam)) / (2 * h)
        got = implicit_classifier_grad(eps_c(z, 1, lam), eps_u(z, lam), sigma)
        assert np.isclose(got, fd, atol=1e-5), f"at z = {z}, lambda = {lam} expected grad log p(c|z) = {fd:.5f}, got {got:.5f}"


def test_classifier_guidance_with_implicit_classifier_is_equation_6():
    for z, lam, w in [(1.0, 0.0, 2.0), (-0.3, 1.5, 0.5), (2.0, -1.0, 4.0)]:
        _, sigma, _ = _logit_parts(z, lam)
        ec, eu = eps_c(z, 1, lam), eps_u(z, lam)
        grad = implicit_classifier_grad(ec, eu, sigma)
        via_classifier = classifier_guided_eps(ec, grad, sigma, w)
        direct = guided_eps(ec, eu, w)
        assert np.isclose(via_classifier, direct), f"expected classifier route to equal Equation 6 ({direct:.5f}), got {via_classifier:.5f}"


def test_classifier_guided_with_true_posterior():
    z, lam, w = 1.0, 0.0, 2.0
    _, sigma, _ = _logit_parts(z, lam)
    h = 1e-5
    grad = (log_posterior(z + h, 1, lam) - log_posterior(z - h, 1, lam)) / (2 * h)
    got = classifier_guided_eps(eps_c(z, 1, lam), grad, sigma, w)
    want = guided_eps(eps_c(z, 1, lam), eps_u(z, lam), w)
    assert np.isclose(got, want, atol=1e-5), f"expected {want:.5f} with the true posterior gradient, got {got:.5f}"
