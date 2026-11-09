import numpy as np
from submission import advantages, policy_value, surrogate, visitation

GAMMA = 0.9
R = np.array([0.0, 1.0])
RHO0 = np.array([1.0, 0.0])
P = np.zeros((2, 2, 2))
P[0, 0, 0] = P[1, 0, 1] = 1.0  # action 0 stays
P[0, 1, 1] = P[1, 1, 0] = 1.0  # action 1 switches


def policy(p_switch):
    p = np.asarray(p_switch, dtype=float)
    return np.stack([1 - p, p], axis=1)


def eta(pi):
    return RHO0 @ policy_value(P, R, pi, GAMMA)[1]


def random_mdp(seed, n_s=4, n_a=3):
    rng = np.random.default_rng(seed)
    P_ = rng.dirichlet(np.ones(n_s), size=(n_s, n_a))
    r_ = rng.normal(size=n_s)
    rho0_ = rng.dirichlet(np.ones(n_s))
    pi_ = rng.dirichlet(np.ones(n_a), size=n_s)
    return P_, r_, rho0_, pi_


def test_visitation_page_example():
    rho = visitation(P, RHO0, policy([0.5, 0.2]), GAMMA)
    want = np.array([3.8356, 6.1644])
    assert np.allclose(rho, want, atol=1e-3), f"expected {want}, got {rho}"


def test_visitation_sums_to_geometric_series():
    P_, _, rho0_, pi_ = random_mdp(1)
    rho = visitation(P_, rho0_, pi_, GAMMA)
    assert np.isclose(rho.sum(), 1 / (1 - GAMMA)), f"expected sum {1 / (1 - GAMMA)}, got {rho.sum()}"


def test_visitation_matches_simulated_series():
    P_, _, rho0_, pi_ = random_mdp(2)
    P_pi = np.einsum("sa,sat->st", pi_, P_)
    want, d = np.zeros(4), rho0_.copy()
    for t in range(400):
        want += GAMMA**t * d
        d = d @ P_pi
    got = visitation(P_, rho0_, pi_, GAMMA)
    assert np.allclose(got, want, atol=1e-8), f"expected {want}, got {got}"


def test_advantages_page_example():
    A = advantages(P, R, policy([0.5, 0.2]), GAMMA)
    want = np.array([[-0.6164, 0.6164], [0.2466, -0.9863]])
    assert np.allclose(A, want, atol=1e-3), f"expected\n{want}\ngot\n{A}"


def test_advantages_average_to_zero_under_pi():
    P_, r_, _, pi_ = random_mdp(3)
    A = advantages(P_, r_, pi_, GAMMA)
    avg = np.sum(pi_ * A, axis=1)
    assert A.shape == (4, 3), f"expected shape (4, 3), got {A.shape}"
    assert np.allclose(avg, 0, atol=1e-9), f"sum_a pi(a|s) A(s, a) must be 0 in every state, got {avg}"


def test_surrogate_page_example():
    L = surrogate(P, R, RHO0, GAMMA, policy([0.5, 0.2]), policy([0.9, 0.2]))
    assert np.isclose(L, 8.0559, atol=1e-3), f"expected 8.0559, got {L}"


def test_surrogate_equals_eta_at_old_policy():
    P_, r_, rho0_, pi_ = random_mdp(4)
    L = surrogate(P_, r_, rho0_, GAMMA, pi_, pi_)
    want = rho0_ @ policy_value(P_, r_, pi_, GAMMA)[1]
    assert np.isclose(L, want), f"L_pi(pi) must equal eta(pi) = {want}, got {L}"


def test_equation_1_identity_with_new_visitation():
    # eta(new) = eta(old) + sum_s rho_new(s) sum_a new(a|s) A_old(s, a)   (Equations 1 and 2)
    P_, r_, rho0_, pi_ = random_mdp(5)
    new = np.random.default_rng(6).dirichlet(np.ones(3), size=4)
    rhs = rho0_ @ policy_value(P_, r_, pi_, GAMMA)[1] + np.sum(
        visitation(P_, rho0_, new, GAMMA)[:, None] * new * advantages(P_, r_, pi_, GAMMA)
    )
    want = rho0_ @ policy_value(P_, r_, new, GAMMA)[1]
    assert np.isclose(rhs, want), f"expected eta(new) = {want}, the identity gave {rhs}"


def test_first_order_match_equation_4():
    # L matches eta to first order: the gradients agree, and the gap shrinks like step^2.
    P_, r_, rho0_, pi_ = random_mdp(7)
    direction = np.random.default_rng(8).normal(size=(4, 3))
    logits0 = np.log(pi_)

    def softmax_rows(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def gap(t):
        new = softmax_rows(logits0 + t * direction)
        true = rho0_ @ policy_value(P_, r_, new, GAMMA)[1]
        return true - surrogate(P_, r_, rho0_, GAMMA, pi_, new)

    g1, g2 = abs(gap(1e-2)), abs(gap(2e-2))
    assert g1 > 0, "the gap between eta and L should be nonzero away from the old policy"
    ratio = g2 / g1
    assert 3.0 < ratio < 5.0, f"the gap eta - L should scale as step^2 (ratio about 4), got {ratio}"
