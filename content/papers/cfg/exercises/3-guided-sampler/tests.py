import numpy as np
from submission import NULL, alpha_sigma, lambdas, sample_cfg, toy_eps_model


def run(w, c=1, n=6000, T=64, seed=0, v=0.0):
    return sample_cfg(toy_eps_model, c, w, n, T, np.random.default_rng(seed), v=v)


def test_unguided_matches_the_class():
    x = run(0.0)
    assert abs(x.mean() - 2.0) < 0.08, f"w = 0 must sample the class N(2, 1); expected mean 2.0, got {x.mean():.3f}"
    assert abs(x.std() - 1.0) < 0.06, f"w = 0 must sample the class N(2, 1); expected std 1.0, got {x.std():.3f}"


def test_other_class_is_symmetric():
    x = run(0.0, c=0)
    assert abs(x.mean() + 2.0) < 0.08, f"expected mean -2.0 for class 0, got {x.mean():.3f}"


def test_guidance_moves_away_and_concentrates():
    x0, x2, x4 = run(0.0), run(2.0), run(4.0)
    assert x2.mean() > x0.mean() + 0.3, f"w = 2 must move samples away from the other class; means {x0.mean():.3f} then {x2.mean():.3f}"
    assert x4.mean() > x2.mean(), f"the mean must keep moving with w; got {x2.mean():.3f} at w = 2 and {x4.mean():.3f} at w = 4"
    assert x2.std() < x0.std() - 0.05, f"w = 2 must reduce the spread; stds {x0.std():.3f} then {x2.std():.3f}"
    assert x4.std() < x0.std() - 0.1, f"w = 4 must reduce the spread; stds {x0.std():.3f} then {x4.std():.3f}"


def test_guided_values_are_close_to_reference():
    x = run(2.0, n=20000, T=256)
    assert abs(x.mean() - 2.479) < 0.07, f"expected mean about 2.479 at w = 2 and T = 256, got {x.mean():.3f}"
    assert abs(x.std() - 0.853) < 0.04, f"expected std about 0.853 at w = 2 and T = 256, got {x.std():.3f}"


def test_model_calls_use_null_and_increasing_lambda():
    calls = []

    def spy(z, c, lam):
        calls.append((c, float(lam)))
        return toy_eps_model(z, c, lam)

    sample_cfg(spy, 1, 1.5, 100, 16, np.random.default_rng(0))
    assert len(calls) == 32, f"expected 2 model calls per step for 16 steps = 32, got {len(calls)}"
    conds = [c for c, _ in calls if c == 1]
    nulls = [c for c, _ in calls if c == NULL]
    assert len(conds) == 16 and len(nulls) == 16, f"expected 16 conditional and 16 unconditional calls, got {len(conds)} and {len(nulls)}"
    lam = np.array([l for _, l in calls])[::2]
    assert np.allclose(lam, lambdas(16)), "the sampler must walk the lambda sequence lambdas(T) in order, from lambda_min up to lambda_max"
    assert np.all(np.diff(lam) > 0), "lambda must increase at every step"


def test_noise_draws_stop_at_the_last_step():
    class Counting:
        def __init__(self):
            self.g = np.random.default_rng(0)
            self.n = 0

        def standard_normal(self, size=None):
            self.n += 1
            return self.g.standard_normal(size)

    rng = Counting()
    sample_cfg(toy_eps_model, 1, 1.0, 50, 12, rng)
    assert rng.n == 12, f"expected 12 draws (z_1 and 11 steps), got {rng.n}; no noise is added at the final step"


def test_last_step_returns_denoised_estimate():
    x = run(0.0, n=20000, T=256)
    assert abs(x.std() - 1.0) < 0.03, f"adding noise at the last step would inflate the spread; expected std near 1.0 at w = 0, got {x.std():.3f}"


def test_v_changes_the_sampler_but_not_the_target():
    a = run(0.0, v=0.0, seed=5)
    b = run(0.0, v=0.3, seed=5)
    assert not np.allclose(a, b), "v interpolates the variance, so v = 0.3 must give different samples from v = 0 under the same seed"
    assert abs(b.mean() - 2.0) < 0.1 and abs(b.std() - 1.0) < 0.08, f"v = 0.3 should still sample N(2, 1); got mean {b.mean():.3f}, std {b.std():.3f}"


def test_seed_reproducible():
    a, b = run(1.0, seed=7), run(1.0, seed=7)
    assert np.array_equal(a, b), "the same seed must give the same samples"


def test_matches_a_step_by_step_reference():
    def model(z, c, lam):
        return 0.3 * z + 0.1 * (c + 2) + 0.01 * lam

    T, w, v, n = 5, 1.5, 0.2, 4
    got = sample_cfg(model, 1, w, n, T, np.random.default_rng(11), v=v)
    g = np.random.default_rng(11)
    lam = lambdas(T)
    z = g.standard_normal(n)
    for t in range(T):
        a, s = alpha_sigma(lam[t])
        eps = (1 + w) * model(z, 1, lam[t]) - w * model(z, NULL, lam[t])
        x = (z - s * eps) / a
        if t == T - 1:
            want = x
            break
        a2, s2 = alpha_sigma(lam[t + 1])
        r = np.exp(lam[t] - lam[t + 1])
        var = ((1 - r) * s2**2) ** (1 - v) * ((1 - r) * s**2) ** v
        z = r * (a2 / a) * z + (1 - r) * a2 * x + np.sqrt(var) * g.standard_normal(n)
    assert np.allclose(got, want, rtol=1e-6, atol=1e-9), f"expected the Algorithm 2 recursion to give {want}, got {got}"
