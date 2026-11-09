import numpy as np
from submission import kl_categorical, klpen_objective, update_beta


def test_kl_value_and_direction():
    p_old = np.array([[0.5, 0.5]])
    p_new = np.array([[0.6, 0.4]])
    got = kl_categorical(p_old, p_new)
    assert abs(got - 0.02041) < 1e-4, f"KL[old, new] expected 0.02041, got {got}"
    rev = kl_categorical(p_new, p_old)
    assert abs(got - rev) > 1e-4, "the two KL directions differ here; check you did not swap or symmetrize"


def test_kl_zero_for_identical():
    p = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    got = kl_categorical(p, p)
    assert abs(got) < 1e-12, f"expected 0, got {got}"


def test_kl_is_mean_over_rows_not_sum_or_columns():
    p_old = np.array([[0.5, 0.5], [0.9, 0.1]])
    p_new = np.array([[0.6, 0.4], [0.5, 0.5]])
    row0 = 0.5 * np.log(0.5 / 0.6) + 0.5 * np.log(0.5 / 0.4)
    row1 = 0.9 * np.log(0.9 / 0.5) + 0.1 * np.log(0.1 / 0.5)
    want = (row0 + row1) / 2
    got = kl_categorical(p_old, p_new)
    assert abs(got - want) < 1e-9, f"expected mean of rows {want}, got {got}"


def test_klpen_objective():
    logp_old = np.log(np.array([0.5, 0.5]))
    logp_new = np.log(np.array([0.6, 0.4]))
    adv = np.array([1.0, -1.0])
    p_old = np.array([[0.5, 0.5], [0.5, 0.5]])
    p_new = np.array([[0.6, 0.4], [0.6, 0.4]])
    surrogate = (1.2 * 1.0 + 0.8 * -1.0) / 2
    want = surrogate - 3.0 * 0.020411
    got = klpen_objective(logp_new, logp_old, adv, p_old, p_new, 3.0)
    assert abs(got - want) < 1e-5, f"expected {want}, got {got}"
    got0 = klpen_objective(logp_new, logp_old, adv, p_old, p_new, 0.0)
    assert abs(got0 - surrogate) < 1e-9, f"beta=0 leaves only the surrogate {surrogate}, got {got0}"


def test_beta_rule_each_branch():
    assert update_beta(1.0, 0.004, 0.01) == 0.5, f"d below d_targ/1.5 must halve beta, got {update_beta(1.0, 0.004, 0.01)}"
    assert update_beta(1.0, 0.03, 0.01) == 2.0, f"d above 1.5*d_targ must double beta, got {update_beta(1.0, 0.03, 0.01)}"
    assert update_beta(1.0, 0.012, 0.01) == 1.0, f"d inside the band must keep beta, got {update_beta(1.0, 0.012, 0.01)}"


def test_beta_rule_thresholds():
    # 0.0067 > 0.01/1.5 = 0.006667: no change. 0.0066 < 0.006667: halve.
    assert update_beta(1.0, 0.0067, 0.01) == 1.0, "0.0067 is above d_targ/1.5 = 0.006667, so beta stays 1"
    assert update_beta(1.0, 0.0066, 0.01) == 0.5, "0.0066 is below d_targ/1.5 = 0.006667, so beta halves"
    assert update_beta(1.0, 0.0149, 0.01) == 1.0, "0.0149 is below 1.5*d_targ = 0.015, so beta stays 1"
    assert update_beta(1.0, 0.0151, 0.01) == 2.0, "0.0151 is above 1.5*d_targ = 0.015, so beta doubles"


def test_beta_sequence_from_page():
    beta = 1.0
    seen = []
    for d in [0.004, 0.012, 0.030, 0.0095, 0.0066]:
        beta = update_beta(beta, d, 0.01)
        seen.append(beta)
    assert seen == [0.5, 0.5, 1.0, 1.0, 0.5], f"expected [0.5, 0.5, 1.0, 1.0, 0.5], got {seen}"
