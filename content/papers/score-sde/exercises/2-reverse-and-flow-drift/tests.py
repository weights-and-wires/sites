import numpy as np
from submission import beta, gaussian_score, int_beta, pf_ode_drift, reverse_sde_drift


def test_gaussian_score_value():
    s = gaussian_score(1.0, 0.5, 2.0, 0.5)
    assert np.isclose(s, -0.4652, atol=1e-4), f"expected -0.4652 at x = 1, t = 0.5, got {s}"


def test_gaussian_score_at_t_zero_is_the_data_score():
    # At t = 0, p_0 = N(2, 0.25), so the score is -(x - 2) / 0.25.
    s = gaussian_score(np.array([1.0, 2.0, 3.0]), 0.0, 2.0, 0.5)
    assert np.allclose(s, [4.0, 0.0, -4.0]), f"expected [4, 0, -4], got {s}"


def test_gaussian_score_for_standard_normal_data_is_time_independent():
    # N(0, 1) data stays N(0, 1) under the VP-SDE, so the score is -x at every t.
    x = np.array([-1.0, 0.3, 2.0])
    for t in (0.1, 0.5, 1.0):
        s = gaussian_score(x, t, 0.0, 1.0)
        assert np.allclose(s, -x), f"at t = {t} expected {-x}, got {s}"


def test_reverse_sde_drift_value():
    got = reverse_sde_drift(1.0, 0.5, -0.4652)
    assert np.isclose(got, -0.3495, atol=1e-3), f"expected -0.3495, got {got}"


def test_pf_ode_drift_value():
    got = pf_ode_drift(1.0, 0.5, -0.4652)
    assert np.isclose(got, -2.6873, atol=1e-3), f"expected -2.6873, got {got}"


def test_zero_score_leaves_the_forward_drift():
    x = np.array([0.5, -1.0])
    f = -0.5 * beta(0.3) * x
    assert np.allclose(reverse_sde_drift(x, 0.3, np.zeros(2)), f), "with score 0 the reverse drift is f"
    assert np.allclose(pf_ode_drift(x, 0.3, np.zeros(2)), f), "with score 0 the flow drift is f"


def test_ode_takes_half_of_the_sde_score_term():
    x = np.array([0.5, -1.0, 2.0])
    s = np.array([0.7, -0.2, 1.1])
    f = -0.5 * beta(0.6) * x
    a = reverse_sde_drift(x, 0.6, s) - f
    b = pf_ode_drift(x, 0.6, s) - f
    assert np.allclose(b, 0.5 * a), f"expected the ODE score term {0.5 * a}, got {b}"
    assert np.allclose(a, -beta(0.6) * s), f"expected the SDE score term {-beta(0.6) * s}, got {a}"


def test_probability_flow_satisfies_the_continuity_equation():
    # For x(0) ~ N(2, 0.25), check d p_t / dt = -d/dx [ drift * p_t ] by finite differences.
    m0, s0 = 2.0, 0.5

    def p(x, t):
        m = np.exp(-0.5 * int_beta(t))
        v = m**2 * s0**2 + 1 - m**2
        return np.exp(-0.5 * (x - m * m0) ** 2 / v) / np.sqrt(2 * np.pi * v)

    x = np.linspace(-4, 5, 2001)
    h = x[1] - x[0]
    t, dt = 0.3, 1e-5
    dpdt = (p(x, t + dt) - p(x, t - dt)) / (2 * dt)
    flux = pf_ode_drift(x, t, gaussian_score(x, t, m0, s0)) * p(x, t)
    rhs = -np.gradient(flux, h)
    err = np.max(np.abs(dpdt - rhs)[50:-50])
    assert err < 1e-3, f"expected the ODE drift to move p_t correctly (error below 1e-3), got max error {err:.4f}"
