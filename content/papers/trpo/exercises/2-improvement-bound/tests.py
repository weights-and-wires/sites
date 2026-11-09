import numpy as np
from submission import improvement_bound, max_kl, max_tv, penalty_constant, policy_value

GAMMA = 0.9
R = np.array([0.0, 1.0])
RHO0 = np.array([1.0, 0.0])
P = np.zeros((2, 2, 2))
P[0, 0, 0] = P[1, 0, 1] = 1.0
P[0, 1, 1] = P[1, 1, 0] = 1.0


def policy(p_switch):
    p = np.asarray(p_switch, dtype=float)
    return np.stack([1 - p, p], axis=1)


PI = policy([0.5, 0.2])
PI_NEW = policy([0.9, 0.2])


def test_max_tv_page_example():
    got = max_tv(PI, PI_NEW)
    assert np.isclose(got, 0.4), f"expected 0.4, got {got}"


def test_max_tv_takes_max_over_states_not_mean():
    old, new = policy([0.5, 0.2]), policy([0.6, 0.9])
    got = max_tv(old, new)
    assert np.isclose(got, 0.7), f"expected 0.7 (state 1 differs most), got {got}"


def test_max_kl_page_example():
    got = max_kl(PI, PI_NEW)
    assert np.isclose(got, 0.5108, atol=1e-4), f"expected 0.5108, got {got}"


def test_max_kl_is_not_symmetric():
    got = max_kl(PI_NEW, PI)
    assert np.isclose(got, 0.3681, atol=1e-4), f"expected KL(new || old) = 0.3681 when the arguments are swapped, got {got}"


def test_pinsker_relation():
    rng = np.random.default_rng(0)
    for _ in range(100):
        a = rng.dirichlet(np.ones(3), size=4)
        b = rng.dirichlet(np.ones(3), size=4)
        assert max_tv(a, b) ** 2 <= max_kl(a, b) + 1e-12, f"D_TV^2 = {max_tv(a, b) ** 2} must not exceed D_KL = {max_kl(a, b)}"


def test_penalty_constant():
    got = penalty_constant(0.9863, 0.9)
    assert np.isclose(got, 355.07, atol=0.05), f"expected 355.07, got {got}"
    got2 = penalty_constant(1.0, 0.5)
    assert np.isclose(got2, 8.0), f"for epsilon=1, gamma=0.5 expected 8.0, got {got2}"


def test_bounds_page_example():
    b_kl = improvement_bound(P, R, RHO0, GAMMA, PI, PI_NEW, divergence="kl")
    b_tv = improvement_bound(P, R, RHO0, GAMMA, PI, PI_NEW, divergence="tv")
    assert np.isclose(b_kl, -173.32, atol=0.05), f"KL bound expected -173.32, got {b_kl}"
    assert np.isclose(b_tv, -48.76, atol=0.05), f"TV bound expected -48.76, got {b_tv}"


def test_default_divergence_is_kl():
    a = improvement_bound(P, R, RHO0, GAMMA, PI, PI_NEW)
    b = improvement_bound(P, R, RHO0, GAMMA, PI, PI_NEW, divergence="kl")
    assert np.isclose(a, b), f"the default must be 'kl': expected {b}, got {a}"


def test_bound_is_tight_at_the_old_policy():
    eta = RHO0 @ policy_value(P, R, PI, GAMMA)[1]
    for div in ("kl", "tv"):
        b = improvement_bound(P, R, RHO0, GAMMA, PI, PI, divergence=div)
        assert np.isclose(b, eta), f"{div}: with pi_new = pi the bound must equal eta(pi) = {eta}, got {b}"


def test_bounds_hold_on_random_mdps():
    rng = np.random.default_rng(0)
    for trial in range(200):
        n_s, n_a = 3, 3
        P_ = rng.dirichlet(np.ones(n_s), size=(n_s, n_a))
        r_ = rng.normal(size=n_s)
        rho0_ = rng.dirichlet(np.ones(n_s))
        gamma = rng.choice([0.5, 0.9, 0.95])
        old = rng.dirichlet(np.ones(n_a), size=n_s)
        new = 0.5 * old + 0.5 * rng.dirichlet(np.ones(n_a) * 0.3, size=n_s) if trial % 2 else rng.dirichlet(np.ones(n_a), size=n_s)
        new = new / new.sum(axis=1, keepdims=True)
        eta_new = rho0_ @ policy_value(P_, r_, new, gamma)[1]
        for div in ("tv", "kl"):
            b = improvement_bound(P_, r_, rho0_, gamma, old, new, divergence=div)
            assert eta_new >= b - 1e-9, f"trial {trial}, {div}: eta(new) = {eta_new} fell below the bound {b}"
