import numpy as np
from submission import eff_factor, n_eff, loss_hat


def test_single_factor():
    v = float(eff_factor(4, 2.6745))
    assert abs(v - 0.7759) < 1e-4, f"1 - exp(-4/2.6745): expected 0.7759, got {v}"
    w = eff_factor([2, 4], 2.6745, 0.3037)
    assert np.allclose(w, [0.3586, 0.6964], atol=1e-4), f"with offset 0.3037 expected [0.3586, 0.6964], got {w}"


def test_factor_saturates():
    lo, hi = float(eff_factor(1, 2.0)), float(eff_factor(40, 2.0))
    assert lo < 0.5 and abs(hi - 1.0) < 1e-6, f"expected a value below 0.5 at 1 bit and about 1 at 40 bits, got {lo} and {hi}"


def test_n_eff_plain():
    v = float(n_eff(1e9, 4, 4, 4))
    assert abs(v / 1e9 - 0.6389) < 2e-4, f"N_eff/N at 4/4/4 bits: expected 0.6389, got {v / 1e9}"


def test_components_differ():
    # The three gammas differ, so the order of the precisions matters.
    a = float(n_eff(1e9, 8, 4, 8)) / 1e9
    b = float(n_eff(1e9, 4, 8, 8)) / 1e9
    c = float(n_eff(1e9, 8, 8, 4)) / 1e9
    assert abs(a - 0.7941) < 2e-4, f"(P_w, P_a, P_kv) = (8, 4, 8): expected 0.7941, got {a}"
    assert abs(b - 0.7549) < 2e-4, f"(4, 8, 8): expected 0.7549, got {b}"
    assert abs(c - 0.9101) < 2e-4, f"(8, 8, 4): expected 0.9101, got {c}"


def test_n_eff_offsets():
    v = float(n_eff(1e9, 4, 4, 4, offsets=True)) / 1e9
    w = float(n_eff(220e6, 8, 8, 8, offsets=True))
    assert abs(v - 0.1910) < 2e-4, f"with offsets, N_eff/N at 4 bits: expected 0.1910, got {v}"
    assert abs(w / 1e8 - 1.8211) < 2e-3, f"with offsets, N_eff for N=220M at 8 bits: expected 1.8211e8, got {w}"


def test_loss_hat():
    v = float(loss_hat(220e6, 26e9, 8, 8, 8, offsets=True))
    assert abs(v - 3.2271) < 2e-4, f"L(220M, 26B, 8 bits, offsets): expected 3.2271, got {v}"
    u = float(loss_hat(220e6, 26e9, 6, 6, 6))
    assert abs(u - 3.2260) < 2e-4, f"L(220M, 26B, 6 bits, no offsets): expected 3.2260, got {u}"


def test_loss_limit_and_asymmetry():
    big = 1e6
    a = float(loss_hat(1e9, 2e10, big, big, big))
    b = float(loss_hat(2e10, 1e9, big, big, big))
    assert abs(a - 3.0497) < 2e-4, f"full precision, N=1e9, D=2e10: expected 3.0497, got {a}"
    assert abs(b - 3.4119) < 2e-4, f"full precision, N=2e10, D=1e9: expected 3.4119, got {b}"


def test_vectorised():
    v = loss_hat(np.array([1e9, 2e10]), np.array([2e10, 1e9]), 1e6, 1e6, 1e6)
    assert v.shape == (2,), f"expected shape (2,), got {v.shape}"
    assert np.allclose(v, [3.0497, 3.4119], atol=2e-4), f"expected [3.0497, 3.4119], got {v}"
