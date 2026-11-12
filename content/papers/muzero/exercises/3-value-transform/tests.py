import numpy as np
from submission import h, h_inv, phi, to_target, from_logits


def test_h_values():
    got = np.asarray(h(np.array([0.0, 3.0, -3.0, 100.0])))
    want = np.array([0.0, 1.0 + 0.003, -1.0 - 0.003, 9.0498756 + 0.1])
    assert np.allclose(got, want, atol=1e-6), f"expected {want}, got {got}"


def test_h_is_odd_and_increasing():
    x = np.linspace(-50.0, 50.0, 101)
    y = np.asarray(h(x))
    assert np.allclose(y, -np.asarray(h(-x))), "h must be an odd function"
    assert np.all(np.diff(y) > 0), "h must be strictly increasing"


def test_h_inv_round_trip():
    x = np.array([-5000.0, -100.0, -3.7, -0.01, 0.0, 0.5, 3.7, 100.0, 5000.0])
    got = np.asarray(h_inv(h(x)))
    assert np.allclose(got, x, rtol=1e-9, atol=1e-9), f"expected {x}, got {got}"


def test_h_inv_is_not_just_the_quadratic_without_the_linear_term():
    # Without the epsilon term h_inv(y) would be sign(y) * (y^2 + 2|y|). The exact inverse differs for large y.
    y = 100.0
    naive = y * y + 2.0 * y
    got = float(h_inv(y))
    assert abs(got - naive) > 1.0, f"the epsilon term must change the inverse, got {got} vs naive {naive}"
    assert abs(float(h(got)) - y) < 1e-9, f"h(h_inv(100)) should be 100, got {float(h(got))}"


def test_phi_paper_example():
    got = np.asarray(phi(3.7, 0, 5))
    want = np.array([0.0, 0.0, 0.0, 0.3, 0.7, 0.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_phi_integer_is_one_hot():
    got = np.asarray(phi(2.0, 0, 5))
    want = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_phi_negative_and_offset_support():
    # -1.25 lies between -2 and -1: weight 0.25 on -2 (index 1) and 0.75 on -1 (index 2).
    got = np.asarray(phi(-1.25, -3, 3))
    want = np.zeros(7)
    want[1] = 0.25
    want[2] = 0.75
    assert got.shape == (7,), f"expected length 7, got {got.shape}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_phi_sums_to_one_and_recovers_x():
    x = 4.2
    p = np.asarray(phi(x, -10, 10))
    support = np.arange(-10, 11)
    assert abs(p.sum() - 1.0) < 1e-12, f"weights must sum to 1, got {p.sum()}"
    assert abs(float(p @ support) - x) < 1e-12, f"expected the mean to be {x}, got {float(p @ support)}"


def test_phi_clips_values_outside_the_support():
    top = np.asarray(phi(50.0, -5, 5))
    bottom = np.asarray(phi(-50.0, -5, 5))
    assert top[-1] == 1.0 and top.sum() == 1.0, f"expected all weight on the top support, got {top}"
    assert bottom[0] == 1.0 and bottom.sum() == 1.0, f"expected all weight on the bottom support, got {bottom}"


def test_to_target_transforms_before_encoding():
    # h(3.7) = 1.1716..., so the weight lands on supports 1 and 2.
    got = np.asarray(to_target(3.7, 0, 5))
    hx = float(h(3.7))
    want = np.zeros(6)
    want[1] = 2.0 - hx
    want[2] = hx - 1.0
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_default_support_has_601_entries():
    got = np.asarray(to_target(0.0))
    assert got.shape == (601,), f"expected shape (601,), got {got.shape}"
    assert got[300] == 1.0, f"zero belongs to the middle support, got {got[300]}"


def test_from_logits_inverts_a_one_hot_target():
    logits = np.full(11, -30.0)
    logits[7] = 30.0  # support value 2 when lo = -5
    got = from_logits(logits, lo=-5)
    want = float(h_inv(2.0))
    assert abs(got - want) < 1e-6, f"expected {want}, got {got}"


def test_from_logits_takes_the_mean_before_inverting():
    # Half the mass on support 0 and half on support 10: mean 5, then h_inv(5).
    logits = np.full(11, -50.0)
    logits[0] = 0.0
    logits[10] = 0.0
    got = from_logits(logits, lo=0)
    want = float(h_inv(5.0))
    wrong = 0.5 * (float(h_inv(0.0)) + float(h_inv(10.0)))
    assert abs(got - want) < 1e-6, f"expected h_inv(5) = {want}, got {got} (inverting each support first gives {wrong})"


def test_from_logits_is_stable_for_large_logits():
    logits = np.zeros(601)
    logits[400] = 1000.0
    got = from_logits(logits)
    want = float(h_inv(100.0))
    assert np.isfinite(got), f"expected a finite value, got {got}"
    assert abs(got - want) < 1e-6, f"expected {want}, got {got}"
