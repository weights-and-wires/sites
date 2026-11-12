import numpy as np
from submission import p_star, best_bits

GBAR = 1.9475  # mean of gamma_w, gamma_i, and gamma_kv in Appendix K, Table 2


def test_p_star_value():
    v = p_star(GBAR)
    assert abs(v - 3.708) < 5e-3, f"gamma_bar = {GBAR}: expected P* = 3.708, got {v}"


def test_p_star_scales_with_gamma():
    a, b = p_star(1.0), p_star(4.0)
    assert abs(a - 1.9038) < 2e-3, f"gamma_bar = 1: expected 1.9038, got {a}"
    assert abs(b - 7.615) < 5e-3, f"gamma_bar = 4: expected 7.615, got {b}"


def test_p_star_is_not_the_trivial_root():
    # exp(x) - 1 = 3x also has the root x = 0, which is not an optimum.
    assert p_star(1.0) > 1.0, f"expected the positive root x = 1.9038, got {p_star(1.0)}"


def test_best_bits_is_independent_of_budget():
    got = [best_bits(c, range(2, 17), GBAR)[0] for c in (1e19, 1e21, 1e23)]
    assert got == [4, 4, 4], f"expected the same best width 4 at every budget, got {got}"


def test_best_bits_loss():
    p, loss = best_bits(1e21, range(2, 17), GBAR)
    assert abs(loss - 2.8969) < 2e-3, f"C = 1e21: expected loss 2.8969, got {loss}"
    _, loss23 = best_bits(1e23, range(2, 17), GBAR)
    assert abs(loss23 - 2.8069) < 2e-3, f"C = 1e23: expected loss 2.8069, got {loss23}"


def test_cost_grows_with_bits():
    # 16 bits has almost no precision penalty, but it costs twice as much as 8 bits.
    p, _ = best_bits(1e21, [8, 16], GBAR)
    assert p == 8, f"expected 8 bits to beat 16 bits at equal cost budget, got {p}"
    q, _ = best_bits(1e21, [4, 16], GBAR)
    assert q == 4, f"expected 4 bits to beat 16 bits, got {q}"


def test_sensitive_models_need_more_bits():
    p, _ = best_bits(1e21, range(2, 17), 4.0)
    assert p == 8, f"gamma_bar = 4: expected 8 bits (P* = 7.6), got {p}"
