import numpy as np
from submission import sample_latent, kl_to_standard_normal, estimate_scale, rescale


def test_sample_latent_values():
    mu = np.array([1.0, -2.0])
    logvar = np.array([0.0, np.log(4.0)])
    eps = np.array([0.5, -1.0])
    got = sample_latent(mu, logvar, eps)
    assert np.allclose(got, [1.5, -4.0]), f"expected [1.5, -4.0], got {got}"


def test_sample_latent_zero_noise_is_mean():
    mu = np.arange(6.0).reshape(1, 2, 1, 3)
    got = sample_latent(mu, np.full_like(mu, 2.0), np.zeros_like(mu))
    assert np.allclose(got, mu), f"eps = 0 must return mu, got {got}"


def test_kl_per_component():
    mu = np.array([1.0, -2.0]).reshape(1, 2, 1, 1)
    logvar = np.array([0.0, np.log(4.0)]).reshape(1, 2, 1, 1)
    got = kl_to_standard_normal(mu, logvar)
    assert got.shape == (1,), f"expected shape (1,), got {got.shape}"
    assert np.allclose(got, [0.5 + 2.806852819], atol=1e-6), f"expected [3.3069], got {got}"


def test_kl_zero_at_standard_normal():
    z = np.zeros((3, 4, 2, 2))
    got = kl_to_standard_normal(z, z)
    assert got.shape == (3,) and np.allclose(got, 0), f"expected zeros of shape (3,), got {got}"


def test_kl_sums_over_chw_not_batch():
    mu = np.ones((2, 3, 2, 2))
    got = kl_to_standard_normal(mu, np.zeros_like(mu))
    assert np.allclose(got, [6.0, 6.0]), f"expected [6.0, 6.0] (12 components of 0.5), got {got}"


def test_estimate_scale_is_pooled():
    ch0 = 10 + np.array([-1.0, 1.0, -1.0, 1.0])
    ch1 = np.array([-3.0, 3.0, -3.0, 3.0])
    z = np.stack([ch0, ch1]).reshape(1, 2, 2, 2)
    got = estimate_scale(z)
    assert abs(got - 30**0.5) < 1e-6, f"expected {30**0.5:.4f} (one mean and one std over all axes), got {got}"


def test_estimate_scale_is_std_not_variance():
    rng = np.random.default_rng(0)
    z = 5.0 * rng.standard_normal((4, 3, 8, 8)) + 1.0
    got = estimate_scale(z)
    assert abs(got - 5.0) < 0.5, f"expected about 5.0, got {got}"


def test_rescale_gives_unit_std():
    rng = np.random.default_rng(1)
    z = 4.0 * rng.standard_normal((2, 4, 8, 8))
    out = rescale(z, estimate_scale(z))
    assert abs(out.std() - 1.0) < 1e-9, f"expected std 1.0, got {out.std()}"
