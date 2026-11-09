import numpy as np
from submission import kl_closed_form, kl_monte_carlo

MU = np.array([[0.5, -1.0]])
LOGVAR = np.log(np.array([[0.8, 1.5]]) ** 2)


def test_page_example():
    got = kl_closed_form(MU, LOGVAR)
    assert got.shape == (1,), f"expected shape (1,), got {got.shape}"
    assert np.allclose(got, 0.8877, atol=1e-4), f"expected 0.8877, got {got}"


def test_prior_has_zero_kl():
    got = kl_closed_form(np.zeros((3, 4)), np.zeros((3, 4)))
    assert np.allclose(got, 0.0), f"q equal to the prior must give 0, got {got}"


def test_variance_term_and_mean_term():
    # mu only: 0.5 * sum(mu^2). sigma only: 0.5 * (sigma^2 - 1 - log sigma^2).
    got = kl_closed_form(np.array([[1.0, 2.0]]), np.zeros((1, 2)))
    assert np.allclose(got, 2.5), f"mean-only KL expected 2.5, got {got}"
    got = kl_closed_form(np.zeros((1, 1)), np.array([[np.log(4.0)]]))
    want = 0.5 * (4.0 - 1 - np.log(4.0))
    assert np.allclose(got, want), f"variance-only KL expected {want:.4f}, got {got}"


def test_sums_over_dimensions_not_batch():
    mu = np.array([[1.0, 0.0], [0.0, 2.0]])
    lv = np.zeros((2, 2))
    got = kl_closed_form(mu, lv)
    assert np.allclose(got, [0.5, 2.0]), f"expected per-row [0.5, 2.0], got {got}"


def test_non_negative():
    rng = np.random.default_rng(1)
    got = kl_closed_form(rng.normal(size=(50, 3)), rng.normal(size=(50, 3)))
    assert np.all(got >= -1e-12), f"KL must be non-negative, got min {got.min()}"


def test_monte_carlo_matches_closed_form():
    rng = np.random.default_rng(0)
    eps = rng.standard_normal((200_000, 1, 2))
    mc = kl_monte_carlo(MU, LOGVAR, eps)
    want = kl_closed_form(MU, LOGVAR)
    assert mc.shape == (1,), f"expected shape (1,), got {mc.shape}"
    assert np.allclose(mc, want, atol=0.01), f"expected about {want}, got {mc}"


def test_monte_carlo_batch_rows():
    rng = np.random.default_rng(2)
    mu = rng.normal(size=(3, 2))
    lv = 0.3 * rng.normal(size=(3, 2))
    eps = rng.standard_normal((100_000, 3, 2))
    mc = kl_monte_carlo(mu, lv, eps)
    want = kl_closed_form(mu, lv)
    assert np.allclose(mc, want, atol=0.02), f"expected {want}, got {mc}"
