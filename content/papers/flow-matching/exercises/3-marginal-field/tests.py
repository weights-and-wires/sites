import numpy as np
from submission import marginal_vf

SM = 0.1


def cond_u(x, t, x1):
    return (x1 - (1 - SM) * x) / (1 - (1 - SM) * t)


def test_single_point_is_the_conditional_field():
    data = np.array([[2.0, -1.0]])
    x = np.array([[0.3, 0.7], [-1.0, 2.0]])
    got = marginal_vf(x, 0.5, data, SM)
    want = cond_u(x, 0.5, data[0])
    assert got.shape == (2, 2), f"expected shape (2, 2), got {got.shape}"
    assert np.allclose(got, want), f"with one data point expected Equation 21: {want}, got {got}"


def test_page_example_at_t_half():
    data = np.array([[-2.0], [2.0]])
    got = marginal_vf(np.array([[0.5]]), 0.5, data, SM)
    assert np.allclose(got, [[2.5609]], atol=1e-3), f"expected 2.5609 from the page, got {got}"


def test_symmetric_data_gives_zero_at_the_origin():
    data = np.array([[-2.0], [2.0]])
    got = marginal_vf(np.array([[0.0]]), 0.7, data, SM)
    assert np.allclose(got, 0.0, atol=1e-12), f"by symmetry the field at x = 0 is 0, got {got}"


def test_at_t_zero_the_field_is_the_mean_direction_from_noise():
    # At t = 0 every p_0(x | x1) is N(0, I), so all weights are equal.
    rng = np.random.default_rng(1)
    data = rng.normal(size=(5, 2))
    x = np.array([[0.4, -0.2]])
    got = marginal_vf(x, 0.0, data, SM)
    want = data.mean(axis=0, keepdims=True) - (1 - SM) * x
    assert np.allclose(got, want), f"with equal weights expected {want}, got {got}"


def test_weights_follow_the_variance_not_the_standard_deviation():
    data = np.array([[0.0], [1.0]])
    t = 0.5
    s = 1 - (1 - SM) * t
    x = np.array([[0.4]])
    logw = np.array([-0.5 * (0.4 - 0.0) ** 2 / s**2, -0.5 * (0.4 - t * 1.0) ** 2 / s**2])
    w = np.exp(logw) / np.exp(logw).sum()
    want = w[0] * cond_u(x, t, 0.0) + w[1] * cond_u(x, t, 1.0)
    got = marginal_vf(x, t, data, SM)
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_continuity_equation():
    # dp/dt + d(p u)/dx = 0 on a grid, with p the mixture of Equation 6.
    data = np.array([[-1.5], [0.5], [2.0]])
    xs = np.linspace(-3, 3, 25)

    def p(t, z):
        s = 1 - (1 - SM) * t
        return np.mean([np.exp(-0.5 * ((z - t * d) / s) ** 2) / (s * np.sqrt(2 * np.pi)) for d in data[:, 0]], axis=0)

    def flux(t, z):
        return p(t, z) * marginal_vf(z[:, None], t, data, SM)[:, 0]

    h = 1e-5
    for t in (0.2, 0.6):
        dpdt = (p(t + h, xs) - p(t - h, xs)) / (2 * h)
        dflux = (flux(t, xs + h) - flux(t, xs - h)) / (2 * h)
        worst = np.max(np.abs(dpdt + dflux))
        assert worst < 1e-5, f"t = {t}: continuity residual should be near 0, got {worst:.2e}"


def test_stable_far_from_the_data_near_t_one():
    data = np.array([[-2.0], [2.0]])
    got = marginal_vf(np.array([[50.0], [-50.0]]), 0.99, data, SM)
    assert np.all(np.isfinite(got)), f"the field must stay finite for large |x|, got {got}"
    want = np.array([[cond_u(50.0, 0.99, 2.0)], [cond_u(-50.0, 0.99, -2.0)]])
    assert np.allclose(got, want, rtol=1e-6), f"expected the nearest point's field {want}, got {got}"
