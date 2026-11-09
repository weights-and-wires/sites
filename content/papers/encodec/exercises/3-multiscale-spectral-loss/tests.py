import numpy as np
from submission import spectral_loss, time_l1, stft_mag

FS = 24000
T = np.arange(4096) / FS
X = np.sin(2 * np.pi * 440 * T) + 0.3 * np.sin(2 * np.pi * 3000 * T)


def reference(x, xh, exps):
    out = 0.0
    for i in exps:
        d = stft_mag(x, 2**i) - stft_mag(xh, 2**i)
        out += np.abs(d).sum() + np.sqrt((d**2).sum())
    return out / len(exps)


def test_identical_signals_cost_nothing():
    got = spectral_loss(X, X)
    assert abs(got) < 1e-9, f"identical signals must give 0, got {got}"


def test_matches_reference_on_all_scales():
    rng = np.random.default_rng(0)
    xh = X + 0.05 * rng.normal(size=X.shape)
    want = reference(X, xh, range(5, 12))
    got = spectral_loss(X, xh)
    assert np.isclose(got, want), f"expected {want:.6f}, got {got:.6f}"


def test_l2_term_is_a_norm_not_a_square():
    got = spectral_loss(X, 0.9 * X)
    want = 0.1 * spectral_loss(X, np.zeros_like(X))
    assert np.isclose(got, want), f"scaling x_hat by 0.9 scales the difference by 0.1, so both norms scale by 0.1; expected {want:.6f}, got {got:.6f}"


def test_average_over_scales():
    xh = 0.5 * X
    one = spectral_loss(X, xh, exponents=[7])
    two = spectral_loss(X, xh, exponents=[7, 9])
    nine = spectral_loss(X, xh, exponents=[9])
    assert np.isclose(two, (one + nine) / 2), f"two scales should average: expected {(one + nine) / 2:.6f}, got {two:.6f}"


def test_scale_uses_window_two_to_the_i():
    xh = 0.5 * X
    want = reference(X, xh, [6])
    got = spectral_loss(X, xh, exponents=[6])
    assert np.isclose(got, want), f"exponent 6 means a window of 64 samples; expected {want:.6f}, got {got:.6f}"


def test_spectral_loss_ignores_the_sign_but_time_loss_does_not():
    s = spectral_loss(X, -X)
    t = time_l1(X, -X)
    assert abs(s) < 1e-9, f"a magnitude STFT cannot see a sign flip; expected spectral loss 0, got {s}"
    want = 2 * np.abs(X).sum()
    assert np.isclose(t, want), f"time L1 of x against -x should be 2 * sum|x| = {want:.4f}, got {t}"


def test_time_l1_is_a_sum():
    got = time_l1(np.array([1.0, -2.0, 3.0]), np.array([0.0, 0.0, 0.0]))
    assert np.isclose(got, 6.0), f"expected 6.0 (a sum, not a mean), got {got}"
