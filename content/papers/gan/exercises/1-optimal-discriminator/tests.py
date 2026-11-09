import numpy as np
from submission import optimal_discriminator, value, jsd, virtual_criterion

P = np.array([0.4, 0.4, 0.2, 0.0])
Q = np.array([0.1, 0.2, 0.3, 0.4])


def test_optimal_discriminator_values():
    got = optimal_discriminator(P, Q)
    want = np.array([0.8, 2 / 3, 0.4, 0.0])
    assert np.allclose(got, want), f"expected {want}, got {got}; D* = p_data / (p_data + p_g)"


def test_optimal_discriminator_edge_cases():
    got = optimal_discriminator(np.array([0.5, 0.0, 0.5]), np.array([0.0, 0.0, 0.5]))
    want = np.array([1.0, 0.5, 0.5])
    assert np.all(np.isfinite(got)), f"got {got}; handle the point where both densities are 0"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_value_of_constant_half_is_minus_log_four():
    got = value(P, Q, np.full(4, 0.5))
    assert np.isclose(got, -np.log(4)), f"expected {-np.log(4):.4f}, got {got}; every D = 1/2 gives -log 4"


def test_value_zero_log_zero():
    got = value(np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    assert np.isclose(got, 0.0), f"expected 0.0, got {got}; a point with probability 0 must add nothing"


def test_d_star_maximizes_value():
    d_star = optimal_discriminator(P, Q)
    best = value(P, Q, d_star)
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = np.clip(d_star + 0.05 * rng.normal(size=4), 0.01, 0.99)
        d[3] = d_star[3] if d_star[3] in (0.0, 1.0) else d[3]
        assert value(P, Q, d) <= best + 1e-12, f"a perturbed D scored {value(P, Q, d)}, above V(G, D*) = {best}"


def test_jsd_values():
    assert np.isclose(jsd(P, Q), 0.2088, atol=1e-4), f"expected 0.2088, got {jsd(P, Q)}"
    assert np.isclose(jsd(P, P), 0.0), f"expected 0 for identical inputs, got {jsd(P, P)}"
    assert np.isclose(jsd(P, Q), jsd(Q, P)), "the JS divergence is symmetric"


def test_jsd_disjoint_is_log_two():
    a, b = np.array([0.5, 0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.5, 0.5])
    assert np.isclose(jsd(a, b), np.log(2)), f"expected log 2 = {np.log(2):.4f}, got {jsd(a, b)}"


def test_virtual_criterion_matches_equation_six():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        p, q = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
        want = -np.log(4) + 2 * jsd(p, q)
        got = virtual_criterion(p, q)
        assert np.isclose(got, want), f"expected C(G) = -log 4 + 2 JSD = {want:.5f}, got {got:.5f}"


def test_virtual_criterion_minimum_and_disjoint():
    assert np.isclose(virtual_criterion(P, P), -np.log(4)), f"expected -log 4 at p_g = p_data, got {virtual_criterion(P, P)}"
    a, b = np.array([0.5, 0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.5, 0.5])
    got = virtual_criterion(a, b)
    assert np.isfinite(got) and np.isclose(got, 0.0), f"disjoint supports: expected C(G) = 0, got {got}"
