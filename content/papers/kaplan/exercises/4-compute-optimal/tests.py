import numpy as np
from submission import optimal_loss, compute_for_loss, optimal_model_size, optimal_tokens, critical_batch_size

PF_DAY = 8.64e19


def test_optimal_loss_at_one_pf_day():
    got = float(optimal_loss(1.0))
    assert abs(got - 2.6581) < 1e-3, f"expected 2.6581, got {got}"


def test_ten_times_compute_scales_loss_by_10_to_minus_alpha():
    got = float(optimal_loss(10.0) / optimal_loss(1.0))
    want = 10**-0.050
    assert abs(got - want) < 1e-9, f"expected {want}, got {got}"


def test_compute_for_loss_inverts_the_law():
    for c in (1e-3, 1.0, 250.0):
        back = float(compute_for_loss(optimal_loss(c)))
        assert abs(back / c - 1) < 1e-9, f"compute_for_loss(optimal_loss({c})) must return {c}, got {back}"


def test_compute_for_loss_two():
    got = float(compute_for_loss(2.0))
    assert abs(got / 295.64 - 1) < 1e-3, f"expected about 295.64 PF-days for a loss of 2.0, got {got}"


def test_model_size_exponent():
    assert abs(float(optimal_model_size(1.0)) - 1.3e9) < 1, f"expected 1.3e9 at 1 PF-day, got {optimal_model_size(1.0)}"
    got = float(optimal_model_size(10.0) / optimal_model_size(1.0))
    assert abs(got - 5.370) < 1e-3, f"10x compute must give 10**0.73 = 5.370x parameters, got {got}"


def test_token_exponent():
    got = float(optimal_tokens(1.0))
    assert abs(got / 2e10 - 1) < 1e-9, f"expected 2e10 tokens at 1 PF-day, got {got}"
    ratio = float(optimal_tokens(10.0) / optimal_tokens(1.0))
    assert abs(ratio - 1.862) < 1e-3, f"10x compute must give 10**0.27 = 1.862x tokens, got {ratio}"


def test_compute_budget_is_consistent():
    # Training at B_crit uses C = 2 * C_min = 6 N D, so 6 N D / (C_min in FLOPs) should be about 2
    # and the same at every budget, because the exponents of N and D sum to 1.
    ratios = [float(6 * optimal_model_size(c) * optimal_tokens(c) / (PF_DAY * c)) for c in (1e-3, 1.0, 1e2)]
    assert all(1.6 < r < 2.0 for r in ratios), f"expected 6ND / C_min near 1.8, got {ratios}"
    assert max(ratios) - min(ratios) < 1e-6, f"the ratio must not change with the budget, got {ratios}"


def test_critical_batch_size():
    got = [float(critical_batch_size(l)) for l in (2.0, 3.0)]
    want = [7.740e6, 1.1226e6]
    assert np.allclose(got, want, rtol=1e-3), f"expected {want}, got {got}"
    assert float(critical_batch_size(2.0)) > float(critical_batch_size(3.0)), "B_crit must grow as the loss falls"
