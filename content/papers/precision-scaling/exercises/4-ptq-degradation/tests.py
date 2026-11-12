import numpy as np
from submission import delta_ptq, d_crit

B, BETA = 1.806e4, 0.4965


def test_delta_values():
    a = float(delta_ptq(30e6, 26e9, 4))
    b = float(delta_ptq(30e6, 1.5e9, 4))
    c = float(delta_ptq(220e6, 26e9, 4))
    assert abs(a - 0.0349) < 2e-4, f"N=30M, D=26B, 4 bits: expected 0.0349, got {a}"
    assert abs(b - 0.0082) < 2e-4, f"N=30M, D=1.5B, 4 bits: expected 0.0082, got {b}"
    assert abs(c - 0.0176) < 2e-4, f"N=220M, D=26B, 4 bits: expected 0.0176, got {c}"


def test_delta_grows_with_data_and_shrinks_with_size():
    base = float(delta_ptq(100e6, 10e9, 5))
    more_d = float(delta_ptq(100e6, 100e9, 5)) / base
    more_n = float(delta_ptq(1000e6, 10e9, 5)) / base
    assert abs(more_d - 3.212) < 5e-3, f"10x tokens: expected a factor 3.212, got {more_d}"
    assert abs(more_n - 0.453) < 5e-3, f"10x parameters: expected a factor 0.453, got {more_n}"


def test_delta_falls_with_bits():
    r = float(delta_ptq(100e6, 10e9, 5) / delta_ptq(100e6, 10e9, 4))
    assert abs(r - 0.1840) < 1e-3, f"one more bit: expected a factor 0.1840, got {r}"


def test_d_crit_values():
    a = float(d_crit(30e6, 5))
    b = float(d_crit(8e9, 4))
    assert abs(a / 4.787e11 - 1) < 2e-3, f"N=30M, 5 bits: expected 4.787e11, got {a:.4e}"
    assert abs(b / 6.009e11 - 1) < 2e-3, f"N=8B, 4 bits: expected 6.009e11, got {b:.4e}"


def test_d_crit_is_a_stationary_point():
    # The pretraining loss falls with D and the PTQ penalty rises. They balance at D_crit.
    n, p = 30e6, 5
    dc = float(d_crit(n, p))
    f = lambda d: B * d**-BETA + float(delta_ptq(n, d, p))
    eps = 1e-4
    slope = (f(dc * (1 + eps)) - f(dc * (1 - eps))) / (2 * eps * dc)
    scale = BETA * B * dc ** (-BETA - 1)
    assert abs(slope) < 1e-4 * scale, f"slope of B D^-beta + delta at D_crit should be 0, got {slope:.3e} (scale {scale:.3e})"
    below = f(dc * 0.5) > f(dc) < f(dc * 2)
    assert below, f"D_crit must be a minimum of B D^-beta + delta: {f(dc * 0.5)}, {f(dc)}, {f(dc * 2)}"


def test_d_crit_per_bit():
    r = float(d_crit(30e6, 6) / d_crit(30e6, 5))
    assert abs(r - 5.405) < 1e-2, f"one more bit: expected D_crit to grow by 5.405, got {r}"
