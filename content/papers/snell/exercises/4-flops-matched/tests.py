import math
from submission import pretrain_flops, inference_flops, matched_multiplier


def test_flop_formulas():
    got = pretrain_flops(10, 100)
    assert got == 6000, f"expected 6 * 10 * 100 = 6000, got {got}"
    got = inference_flops(10, 100)
    assert got == 2000, f"expected 2 * 10 * 100 = 2000, got {got}"


def test_small_cases_by_hand():
    got = matched_multiplier(2, 1.0)
    assert math.isclose(got, 5.0), f"M=2, R=1: expected 2 + 3 = 5, got {got}"
    got = matched_multiplier(2, 3.0)
    assert math.isclose(got, 3.0), f"M=2, R=3: expected 2 + 1 = 3, got {got}"
    got = matched_multiplier(1, 0.5)
    assert math.isclose(got, 1.0), f"M=1 needs no extra compute: expected 1, got {got}"


def test_paper_values_for_a_14x_model():
    for r, want in ((0.16, 257.75), (0.79, 63.3671), (22.0, 15.7727)):
        got = matched_multiplier(14, r)
        assert math.isclose(got, want, rel_tol=1e-4), f"M=14, R={r}: expected {want}, got {got}"


def test_closed_form_matches_the_definitions():
    n, d_pre = 3.0e9, 2.0e12
    for m in (2, 5, 14):
        for r in (0.1, 1.0, 20.0):
            d_inf = r * d_pre
            x = pretrain_flops(n, d_pre)
            y = inference_flops(n, d_inf)
            want = (m * (x + y) - x) / y
            got = matched_multiplier(m, r)
            assert math.isclose(got, want, rel_tol=1e-9), f"M={m}, R={r}: expected {want}, got {got}"


def test_large_r_approaches_m_and_small_r_explodes():
    got = matched_multiplier(14, 1e9)
    assert math.isclose(got, 14.0, rel_tol=1e-6), f"expected about 14 for R=1e9, got {got}"
    assert matched_multiplier(14, 0.01) > 1000, "expected a huge factor when inference is rare (R = 0.01)"


def test_larger_r_means_a_smaller_factor():
    a, b = matched_multiplier(14, 0.16), matched_multiplier(14, 22.0)
    assert a > b, f"expected the factor to fall as R rises, got {a} then {b}"
