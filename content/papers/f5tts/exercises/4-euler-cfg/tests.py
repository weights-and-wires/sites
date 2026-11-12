import numpy as np
from submission import cfg_velocity, euler_cfg

S = 0.5
MU = np.array([[-2.0, 0.0], [2.0, 0.0]])


def toy_field(x, t, cond):
    """Exact OT-path field for data N(mu_c, S^2 I); the unconditional field is the mixture."""
    var = (1 - t) ** 2 + t**2 * S**2

    def comp(c):
        return MU[c] + (t * S**2 - (1 - t)) / var * (x - t * MU[c])

    if cond:
        return comp(1)
    logw = np.array([-((x - t * MU[c]) ** 2).sum(-1) / (2 * var) for c in range(2)])
    w = np.exp(logw - logw.max(0))
    w = w / w.sum(0)
    return sum(w[c][..., None] * comp(c) for c in range(2))


def test_cfg_velocity_formula():
    got = cfg_velocity(np.array([1.5, 0.0]), np.array([-0.5, 0.0]), 2.0)
    want = np.array([5.5, 0.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_cfg_w_zero_is_conditional_and_w_minus_one_is_unconditional():
    vc, vu = np.array([3.0, -1.0]), np.array([1.0, 2.0])
    assert np.allclose(cfg_velocity(vc, vu, 0.0), vc), "w = 0 must return v_cond"
    assert np.allclose(cfg_velocity(vc, vu, -1.0), vu), "w = -1 must return v_uncond"


def test_constant_fields_give_a_closed_form():
    fn = lambda x, t, cond: np.array([4.0, 0.0]) if cond else np.array([1.0, 0.0])
    out = euler_cfg(fn, np.zeros(2), np.linspace(0, 1, 5), 2.0)
    want = np.array([4.0 + 2.0 * 3.0, 0.0])
    assert np.allclose(out, want), f"expected x0 + v_cond + w (v_cond - v_uncond) = {want}, got {out}"


def test_uneven_steps_use_their_own_widths():
    fn = lambda x, t, cond: np.array([t])
    ts = np.array([0.0, 0.1, 0.4, 1.0])
    out = euler_cfg(fn, np.zeros(1), ts, 0.0)
    want = 0.0 * 0.1 + 0.1 * 0.3 + 0.4 * 0.6
    assert np.allclose(out, want), f"expected {want} (left-endpoint Euler, widths 0.1, 0.3, 0.6), got {out}"


def test_velocity_is_evaluated_at_the_current_state():
    fn = lambda x, t, cond: -x
    out = euler_cfg(fn, np.array([1.0]), np.linspace(0, 1, 5), 0.0)
    want = 0.75**4
    assert np.allclose(out, want), f"expected (1 - 0.25)^4 = {want}, got {out}"


def test_w_zero_never_calls_the_unconditional_branch():
    calls = []

    def fn(x, t, cond):
        calls.append(cond)
        return np.zeros_like(x)

    euler_cfg(fn, np.zeros(2), np.linspace(0, 1, 5), 0.0)
    assert calls == [True] * 4, f"expected 4 conditional calls and no unconditional call, got {calls}"
    calls.clear()
    euler_cfg(fn, np.zeros(2), np.linspace(0, 1, 5), 1.0)
    assert calls.count(True) == 4 and calls.count(False) == 4, f"expected 4 calls of each kind with w = 1, got {calls}"


def test_toy_guidance_pushes_away_from_the_other_cluster():
    x0 = np.random.default_rng(0).normal(size=(2000, 2))
    ts = np.linspace(0, 1, 17)
    plain = euler_cfg(toy_field, x0, ts, 0.0)
    guided = euler_cfg(toy_field, x0, ts, 2.0)
    assert abs(plain[:, 0].mean() - 2.0) < 0.05, f"expected mean x about 2.0 for w = 0, got {plain[:, 0].mean():.3f}"
    assert abs(guided[:, 0].mean() - 2.556) < 0.02, f"expected mean x about 2.556 for w = 2, got {guided[:, 0].mean():.3f}"
    assert guided[:, 0].std() < plain[:, 0].std(), f"expected guidance to shrink the spread, got {guided[:, 0].std():.3f} vs {plain[:, 0].std():.3f}"


def test_x0_is_not_modified():
    x0 = np.ones((3, 2))
    euler_cfg(toy_field, x0, np.linspace(0, 1, 5), 1.0)
    assert np.array_equal(x0, np.ones((3, 2))), f"x0 was modified: {x0}"
