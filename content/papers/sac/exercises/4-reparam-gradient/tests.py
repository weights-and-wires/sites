import numpy as np
from submission import reparam_grad, score_grad


def q(a):
    return -((a - 2.0) ** 2)


def dq(a):
    return -2.0 * (a - 2.0)


MU, SIGMA = 0.5, 1.0
TRUE = np.array([-2 * (MU - 2.0), -2 * SIGMA])
EPS4 = np.array([-1.0, 0.0, 1.0, 0.5])


def test_reparam_on_four_fixed_draws():
    # a = [-0.5, 0.5, 1.5, 1.0], Q' = [5, 3, 1, 2]
    got = np.array(reparam_grad(MU, SIGMA, EPS4, dq))
    want = np.array([2.75, -0.75])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_score_on_four_fixed_draws():
    # Q = [-6.25, -2.25, -0.25, -1], (a - mu) = [-1, 0, 1, 0.5]
    got = np.array(score_grad(MU, SIGMA, EPS4, q))
    want = np.array([1.375, 0.75])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_reparam_is_unbiased_for_mu_and_sigma():
    eps = np.random.default_rng(0).standard_normal(200000)
    got = np.array(reparam_grad(MU, SIGMA, eps, dq))
    assert np.allclose(got, TRUE, atol=0.02), f"expected about {TRUE}, got {got}"
    got = np.array(reparam_grad(1.0, 0.5, eps, dq))
    want = np.array([2.0, -1.0])
    assert np.allclose(got, want, atol=0.02), f"mu=1, sigma=0.5: expected about {want}, got {got}"


def test_score_is_unbiased_for_mu_and_sigma():
    eps = np.random.default_rng(1).standard_normal(400000)
    got = np.array(score_grad(MU, SIGMA, eps, q))
    assert np.allclose(got, TRUE, atol=0.1), f"expected about {TRUE}, got {got}"
    got = np.array(score_grad(1.0, 0.5, eps, q))
    want = np.array([2.0, -1.0])
    assert np.allclose(got, want, atol=0.15), f"mu=1, sigma=0.5: expected about {want}, got {got}"


def test_score_uses_the_sigma_and_mu_of_the_policy():
    # With sigma = 2 the score for mu is (a - mu) / 4, not (a - mu).
    eps = np.random.default_rng(2).standard_normal(400000)
    got = np.array(score_grad(0.0, 2.0, eps, q))
    want = np.array([4.0, -4.0])
    assert np.allclose(got, want, atol=0.2), f"mu=0, sigma=2: expected about {want}, got {got}"


def test_reparameterization_has_lower_variance():
    rng = np.random.default_rng(3)
    batch, trials = 16, 2000
    rep, sco = [], []
    for _ in range(trials):
        eps = rng.standard_normal(batch)
        rep.append(reparam_grad(MU, SIGMA, eps, dq)[0])
        sco.append(score_grad(MU, SIGMA, eps, q)[0])
    vr, vs = np.var(rep), np.var(sco)
    assert np.isclose(vr, 4 * SIGMA**2 / batch, rtol=0.15), f"reparameterized variance should be 4 sigma^2 / B = 0.25, got {vr}"
    assert vs > 5 * vr, f"the score-function variance ({vs}) should be well above the reparameterized one ({vr})"


def test_inputs_are_not_modified():
    eps = EPS4.copy()
    reparam_grad(MU, SIGMA, eps, dq)
    score_grad(MU, SIGMA, eps, q)
    assert np.array_equal(eps, EPS4), f"eps must not change, got {eps}"
