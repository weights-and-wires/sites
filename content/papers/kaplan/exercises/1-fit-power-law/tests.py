import numpy as np
from submission import fit_power_law, predict, extrapolate

ALPHA, X_C = 0.076, 8.8e13
X = 10.0 ** np.arange(4, 9)  # 1e4 ... 1e8 parameters
CLEAN = (X_C / X) ** ALPHA
NOISE = np.array([0.012, -0.008, 0.005, -0.011, 0.009])
NOISY = CLEAN * np.exp(NOISE)


def test_recovers_exponent_from_clean_data():
    alpha, _ = fit_power_law(X, CLEAN)
    assert abs(alpha - ALPHA) < 1e-6, f"expected alpha = {ALPHA}, got {alpha}"


def test_recovers_scale_from_clean_data():
    _, x_c = fit_power_law(X, CLEAN)
    assert abs(np.log(x_c / X_C)) < 1e-4, f"expected x_c = {X_C:.3e}, got {x_c:.3e}"


def test_noisy_fit():
    alpha, x_c = fit_power_law(X, NOISY)
    assert abs(alpha - 0.07639) < 2e-4, f"expected alpha near 0.07639, got {alpha}"
    assert abs(np.log(x_c / 8.162e13)) < 0.02, f"expected x_c near 8.162e13, got {x_c:.4e}"


def test_alpha_is_positive_for_falling_loss():
    alpha, _ = fit_power_law(X, NOISY)
    assert alpha > 0, f"loss falls with x, so alpha must be positive, got {alpha}"


def test_predict_matches_the_data():
    got = predict(ALPHA, X_C, X)
    assert np.allclose(got, CLEAN), f"expected {CLEAN}, got {got}"
    assert np.isclose(predict(ALPHA, X_C, 2 * X[0]) / predict(ALPHA, X_C, X[0]), 2 ** -ALPHA), \
        "doubling x must multiply y by 2 ** -alpha"


def test_extrapolate_prediction():
    pred, _ = extrapolate(X, NOISY, np.array([1e9, 1e10]))
    assert np.allclose(pred, [2.3726, 1.9899], atol=1e-3), f"expected [2.3726, 1.9899], got {pred}"


def test_extrapolate_uncertainty_values():
    _, sd = extrapolate(X, NOISY, np.array([1e9, 1e10, 1e11]))
    assert np.allclose(sd, [0.01709, 0.01973, 0.02268], atol=2e-4), f"expected [0.01709, 0.01973, 0.02268], got {sd}"


def test_uncertainty_grows_with_distance():
    _, sd = extrapolate(X, NOISY, np.array([1e6, 1e8, 1e9, 1e12]))
    assert sd[0] < sd[1] < sd[2] < sd[3], f"expected sd to rise as x leaves the data, got {sd}"
    assert sd[0] > 0.0, f"a new observation always carries the scatter of single measurements, got {sd[0]}"


def test_clean_data_has_no_uncertainty():
    pred, sd = extrapolate(X, CLEAN, 1e10)
    assert abs(float(sd)) < 1e-9, f"noise-free data should give sd near 0, got {sd}"
    assert np.isclose(float(pred), (X_C / 1e10) ** ALPHA), f"expected {(X_C / 1e10) ** ALPHA}, got {pred}"


def test_extrapolation_stays_on_the_power_law():
    pred, _ = extrapolate(X, NOISY, 1e10)
    assert 1.9 < float(pred) < 2.1, f"a log-log fit predicts about 1.99 at 1e10, got {pred}"
