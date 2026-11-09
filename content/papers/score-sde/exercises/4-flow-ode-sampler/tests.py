import numpy as np
from submission import MU, S, W, beta, pf_ode_sample, toy_score


def test_single_step_matches_hand_computation():
    x_T = np.array([[0.5, -1.0]])
    got = pf_ode_sample(lambda x, t: -x, x_T, 1, eps=0.1)
    # One Euler step from t = 1 to t = 0.1 with score -x: drift = -b x / 2 + b x / 2 = 0.
    assert np.allclose(got, x_T), f"with score -x the drift is 0, so expected {x_T}, got {got}"
    got2 = pf_ode_sample(lambda x, t: np.zeros_like(x), x_T, 1, eps=0.1)
    want = x_T + (-0.5 * beta(1.0) * x_T) * (0.1 - 1.0)
    assert np.allclose(got2, want), f"expected one Euler step {want}, got {got2}"


def test_score_term_has_coefficient_one_half():
    x_T = np.array([[0.0, 0.0]])
    const = np.array([[1.0, 2.0]])
    got = pf_ode_sample(lambda x, t: const, x_T, 1, eps=0.5)
    want = -0.5 * beta(1.0) * const * (0.5 - 1.0)
    assert np.allclose(got, want), f"expected {want}, got {got}; the score term takes 1/2 of beta"


def test_uses_the_time_at_the_start_of_each_step():
    seen = []

    def spy(x, t):
        seen.append(float(t))
        return np.zeros_like(x)

    pf_ode_sample(spy, np.zeros((1, 2)), 4, eps=0.2)
    want = list(np.linspace(1.0, 0.2, 5)[:4])
    assert np.allclose(seen, want), f"expected score calls at t = {want}, got {seen}"


def test_deterministic_and_does_not_modify_input():
    rng = np.random.default_rng(0)
    x_T = rng.standard_normal((20, 2))
    keep = x_T.copy()
    a = pf_ode_sample(toy_score, x_T, 50)
    b = pf_ode_sample(toy_score, x_T, 50)
    assert np.array_equal(a, b), "the ODE is deterministic: the same x_T must give the same output"
    assert np.array_equal(x_T, keep), "the input array must not be modified in place"
    assert a.shape == (20, 2), f"expected shape (20, 2), got {a.shape}"


def test_samples_recover_the_mixture():
    rng = np.random.default_rng(0)
    x = pf_ode_sample(toy_score, rng.standard_normal((4000, 2)), 200)
    k = np.argmin(np.sum((x[:, None, :] - MU[None]) ** 2, axis=-1), axis=1)
    frac = np.bincount(k, minlength=3) / len(x)
    assert np.allclose(frac, W, atol=0.04), f"expected mode weights {W}, got {np.round(frac, 3)}"
    for j in range(3):
        c = x[k == j]
        assert np.allclose(c.mean(axis=0), MU[j], atol=0.06), f"mode {j}: expected mean {MU[j]}, got {np.round(c.mean(axis=0), 3)}"
        assert np.allclose(c.std(axis=0), S, atol=0.05), f"mode {j}: expected std {S}, got {np.round(c.std(axis=0), 3)}"


def test_few_steps_still_land_near_the_modes():
    rng = np.random.default_rng(1)
    x = pf_ode_sample(toy_score, rng.standard_normal((2000, 2)), 25)
    d = np.min(np.sqrt(np.sum((x[:, None, :] - MU[None]) ** 2, axis=-1)), axis=1)
    assert np.mean(d < 1.0) > 0.97, f"expected over 97% of 25-step samples within 1.0 of a mode, got {np.mean(d < 1.0):.3f}"
