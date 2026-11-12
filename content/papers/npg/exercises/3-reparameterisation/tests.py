import numpy as np
from submission import fisher, policy_gradient, probs, scaled_step


def make_problem():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(3, 3, 3))
    theta = rng.normal(size=3)
    rho = np.array([0.5, 0.3, 0.2])
    Q = rng.normal(size=(3, 3))
    return phi, theta, rho, Q


ONES = np.ones(3)
SCALE = np.array([10.0, 0.1, 3.0])


def test_unit_scale_is_the_original_coordinates():
    phi, theta, rho, Q = make_problem()
    g = policy_gradient(phi, theta, rho, Q)
    F = fisher(phi, theta, rho)
    plain = scaled_step(phi, theta, rho, Q, ONES, 0.1, False)
    nat = scaled_step(phi, theta, rho, Q, ONES, 0.1, True)
    assert np.allclose(plain, 0.1 * g), f"expected {0.1 * g}, got {plain}"
    want = 0.1 * np.linalg.solve(F, g)
    assert np.allclose(nat, want), f"expected {want}, got {nat}"


def test_natural_step_does_not_depend_on_the_units():
    phi, theta, rho, Q = make_problem()
    base = scaled_step(phi, theta, rho, Q, ONES, 0.1, True)
    for c in (SCALE, np.array([0.5, 0.5, 0.5]), np.array([100.0, 1.0, 0.01])):
        got = scaled_step(phi, theta, rho, Q, c, 0.1, True)
        assert np.allclose(got, base, atol=1e-8), f"c = {c}: expected {base}, got {got}"


def test_plain_step_scales_with_the_square_of_the_units():
    phi, theta, rho, Q = make_problem()
    g = policy_gradient(phi, theta, rho, Q)
    got = scaled_step(phi, theta, rho, Q, SCALE, 0.1, False)
    want = 0.1 * SCALE**2 * g
    assert np.allclose(got, want), f"expected 0.1 * c^2 * g = {want}, got {got}"
    base = scaled_step(phi, theta, rho, Q, ONES, 0.1, False)
    assert not np.allclose(got, base), "expected the plain gradient step to change when the units change"


def test_the_step_in_policy_space_is_the_same_under_both_parameterisations():
    phi, theta, rho, Q = make_problem()
    F = fisher(phi, theta, rho)
    a = scaled_step(phi, theta, rho, Q, ONES, 0.5, True)
    b = scaled_step(phi, theta, rho, Q, SCALE, 0.5, True)
    kl_a = 0.5 * a @ F @ a
    kl_b = 0.5 * b @ F @ b
    assert np.isclose(kl_a, kl_b), f"expected the same quadratic KL, got {kl_a} vs {kl_b}"
    pa = probs(phi, theta + a)
    pb = probs(phi, theta + b)
    assert np.allclose(pa, pb), f"expected the same new policy, got\n{pa}\nvs\n{pb}"


def test_plain_step_in_scaled_units_moves_the_policy_more():
    phi, theta, rho, Q = make_problem()
    old = probs(phi, theta)

    def mean_kl(step):
        new = probs(phi, theta + step)
        return float(np.sum(rho * np.sum(old * np.log(old / new), axis=1)))

    kl_unit = mean_kl(scaled_step(phi, theta, rho, Q, ONES, 0.1, False))
    kl_scaled = mean_kl(scaled_step(phi, theta, rho, Q, SCALE, 0.1, False))
    assert kl_scaled > 10 * kl_unit, f"expected the rescaled plain step to move the policy much more, got KL {kl_scaled} vs {kl_unit}"
