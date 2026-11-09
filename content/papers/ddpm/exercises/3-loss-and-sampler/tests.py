import numpy as np
from submission import ddpm_loss, make_schedule, sample

# Data is x0 ~ N(m, s^2) in one dimension. Then x_t is Gaussian and
# E[eps | x_t] has a closed form, so the exact epsilon predictor is known.
M, S = 2.0, 0.5


def make_oracle(alpha_bars, m=M, s=S):
    def oracle(x, t):
        ab = alpha_bars[np.asarray(t) - 1]
        ab = np.reshape(ab, (-1,) + (1,) * (np.ndim(x) - 1)) if np.ndim(ab) else ab
        return np.sqrt(1 - ab) * (x - np.sqrt(ab) * m) / (ab * s**2 + 1 - ab)
    return oracle


def test_loss_zero_for_exact_predictor():
    _, _, ab = make_schedule(10, 0.05, 0.3)
    rng = np.random.default_rng(0)
    x0 = np.full((6, 2), 1.5)  # a point mass at 1.5 per coordinate
    eps = rng.standard_normal((6, 2))
    t = rng.integers(1, 11, size=6)

    def exact(x_t, t):
        a = ab[np.asarray(t) - 1][:, None]
        return (x_t - np.sqrt(a) * 1.5) / np.sqrt(1 - a)

    loss = ddpm_loss(exact, x0, t, eps, ab)
    assert abs(loss) < 1e-12, f"an exact epsilon predictor must give loss 0, got {loss}"


def test_loss_of_zero_predictor():
    _, _, ab = make_schedule(10, 0.05, 0.3)
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((8, 3))
    eps = rng.standard_normal((8, 3))
    t = np.arange(1, 9)
    got = ddpm_loss(lambda x, t: np.zeros_like(x), x0, t, eps, ab)
    want = float(np.mean(np.sum(eps**2, axis=1)))
    assert np.isclose(got, want), f"expected sum over dim, mean over batch = {want:.4f}, got {got:.4f}"


def test_loss_is_sum_over_dim_not_mean():
    _, _, ab = make_schedule(10, 0.05, 0.3)
    eps = np.ones((4, 5))
    got = ddpm_loss(lambda x, t: np.zeros_like(x), np.zeros((4, 5)), np.ones(4, dtype=int), eps, ab)
    assert np.isclose(got, 5.0), f"each row has squared norm 5, so expected 5.0, got {got}"


def test_model_receives_noisy_input_and_timesteps():
    _, _, ab = make_schedule(10, 0.05, 0.3)
    seen = {}

    def spy(x, t):
        seen["x"], seen["t"] = x, t
        return np.zeros_like(x)

    x0 = np.array([[1.0], [2.0]])
    eps = np.array([[0.5], [-0.5]])
    t = np.array([1, 10])
    ddpm_loss(spy, x0, t, eps, ab)
    want = np.sqrt(ab[t - 1])[:, None] * x0 + np.sqrt(1 - ab[t - 1])[:, None] * eps
    assert np.allclose(seen["x"], want), f"the model must see x_t = {want.ravel()}, got {seen['x'].ravel()}"
    assert np.array_equal(np.asarray(seen["t"]), t), f"the model must see t = {t}, got {seen['t']}"


def test_sampler_shape_and_determinism():
    betas, _, ab = make_schedule(50, 1e-4, 0.05)
    o = make_oracle(ab)
    a = sample(o, (7, 1), betas, np.random.default_rng(3))
    b = sample(o, (7, 1), betas, np.random.default_rng(3))
    assert a.shape == (7, 1), f"expected shape (7, 1), got {a.shape}"
    assert np.array_equal(a, b), "the same seed must give the same samples"


def test_no_noise_at_the_last_step():
    # With T = 1 and an exact point-mass predictor, one step returns x0 exactly.
    betas, _, ab = make_schedule(1, 0.3, 0.3)

    def exact(x, t):
        return (x - np.sqrt(ab[t - 1]) * 1.5) / np.sqrt(1 - ab[t - 1])

    x = sample(exact, (2000, 1), betas, np.random.default_rng(0))
    assert np.allclose(x, 1.5), f"at t = 1 the sampler must add no noise; expected all 1.5, got spread {x.std():.4f}"


def test_oracle_sampler_recovers_the_data_distribution():
    betas, _, ab = make_schedule(1000, 1e-4, 0.02)
    x = sample(make_oracle(ab), (5000, 1), betas, np.random.default_rng(0), sigma="beta")
    assert abs(x.mean() - M) < 0.06, f"expected mean {M}, got {x.mean():.3f}"
    assert abs(x.std() - S) < 0.06, f"expected std {S}, got {x.std():.3f}"


def test_sigma_beta_tilde_also_works():
    betas, _, ab = make_schedule(1000, 1e-4, 0.02)
    x = sample(make_oracle(ab), (5000, 1), betas, np.random.default_rng(0), sigma="beta_tilde")
    assert abs(x.mean() - M) < 0.06, f"expected mean {M}, got {x.mean():.3f}"
    assert abs(x.std() - S) < 0.06, f"expected std {S}, got {x.std():.3f}"


def test_variance_choice_changes_the_noise():
    # A model that predicts eps = 0 turns each step into x_{t-1} = x_t / sqrt(alpha_t) + sigma_t z.
    # With T = 2 and no noise at t = 1: Var(x_0) = (1 / alpha_1) * (1 / alpha_2 + sigma_2^2).
    betas, alphas, ab = make_schedule(2, 0.1, 0.4)
    zero = lambda x, t: np.zeros_like(x)
    beta_tilde_2 = (1 - ab[0]) / (1 - ab[1]) * betas[1]
    for name, s2 in (("beta", betas[1]), ("beta_tilde", beta_tilde_2)):
        x = sample(zero, (200_000, 1), betas, np.random.default_rng(0), sigma=name)
        want = (1 / alphas[0]) * (1 / alphas[1] + s2)
        assert abs(x.var() - want) < 0.02, f"sigma={name!r}: expected Var(x_0) = {want:.4f}, got {x.var():.4f}"
