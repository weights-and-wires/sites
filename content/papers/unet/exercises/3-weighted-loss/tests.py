import numpy as np
from submission import border_weight, weighted_pixel_loss


def test_weight_values():
    cases = {(1, 1): 10.2312, (2, 3): 7.0653, (5, 5): 2.3534, (10, 10): 1.0034}
    for (d1, d2), want in cases.items():
        got = float(border_weight(d1, d2))
        assert abs(got - want) < 1e-3, f"d1={d1}, d2={d2}: expected {want}, got {got}"


def test_far_from_borders_is_class_weight():
    got = float(border_weight(40, 40))
    assert abs(got - 1.0) < 1e-6, f"far from any border w should be wc = 1, got {got}"


def test_sum_is_squared_as_a_whole():
    got = float(border_weight(2, 3))
    bug = 1.0 + 10.0 * np.exp(-(2**2 + 3**2) / 50)
    assert abs(got - bug) > 0.5, f"w must use (d1 + d2)^2 = 25, not d1^2 + d2^2 = 13; got {got}, which matches the bug {bug:.4f}"


def test_weight_arguments_and_arrays():
    got = border_weight(np.array([0.0, 0.0]), np.array([0.0, 10.0]), w0=4.0, sigma=10.0, wc=2.0)
    want = np.array([6.0, 2.0 + 4.0 * np.exp(-0.5)])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_loss_page_example():
    logits = np.array([[[2.0, 0.0]], [[0.0, 1.0]]])
    labels = np.array([[0, 0]])
    got = weighted_pixel_loss(logits, labels, np.array([[1.0, 7.0]]))
    assert abs(got - 9.3198) < 1e-3, f"expected 9.3198, got {got}"
    got1 = weighted_pixel_loss(logits, labels, np.ones((1, 2)))
    assert abs(got1 - 1.4402) < 1e-3, f"with unit weights expected 1.4402, got {got1}"


def test_loss_is_positive_and_scales_with_weight():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(3, 4, 4))
    labels = rng.integers(0, 3, size=(4, 4))
    a = weighted_pixel_loss(logits, labels, np.ones((4, 4)))
    b = weighted_pixel_loss(logits, labels, 3 * np.ones((4, 4)))
    assert a > 0, f"cross entropy is positive, got {a}"
    assert abs(b - 3 * a) < 1e-9, f"tripling every weight must triple the loss: expected {3 * a}, got {b}"


def test_softmax_runs_over_channels():
    logits = np.zeros((2, 1, 2))
    logits[0, 0, 0] = 5.0  # pixel 0 favors class 0; pixel 1 is a tie
    labels = np.array([[0, 0]])
    got = weighted_pixel_loss(logits, labels, np.array([[1.0, 1.0]]))
    want = np.log(1 + np.exp(-5.0)) + np.log(2.0)
    assert abs(got - want) < 1e-9, f"expected {want}, got {got}; the softmax must normalize over the K channels at each pixel"


def test_large_logits_do_not_overflow():
    logits = np.array([[[1000.0]], [[0.0]]])
    got = weighted_pixel_loss(logits, np.array([[0]]), np.array([[1.0]]))
    assert np.isfinite(got) and abs(got) < 1e-9, f"expected about 0 for a confident correct pixel, got {got}"
    wrong = weighted_pixel_loss(logits, np.array([[1]]), np.array([[1.0]]))
    assert abs(wrong - 1000.0) < 1e-6, f"expected 1000 for a confident wrong pixel, got {wrong}"


def test_weight_map_changes_which_pixel_counts():
    logits = np.array([[[2.0, 0.0]], [[0.0, 1.0]]])
    labels = np.array([[0, 0]])
    a = weighted_pixel_loss(logits, labels, np.array([[1.0, 0.0]]))
    assert abs(a - 0.1269) < 1e-3, f"weight 0 must remove pixel 1; expected 0.1269, got {a}"
