import numpy as np
from submission import masked_lm_loss

LOGITS = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 1.0, 2.0]])
LABELS = np.array([-100, 1, -100, 0])


def test_page_example():
    got = masked_lm_loss(LOGITS, LABELS)
    assert abs(got - 1.4076) < 1e-3, f"expected 1.4076, got {got}"


def test_unselected_positions_do_not_matter():
    other = LOGITS.copy()
    other[0] = [-50.0, 40.0, 7.0]
    other[2] = [9.0, -9.0, 0.0]
    a, b = masked_lm_loss(LOGITS, LABELS), masked_lm_loss(other, LABELS)
    assert abs(a - b) < 1e-12, f"changing logits at ignored positions changed the loss: expected {a}, got {b}"


def test_mean_over_selected_not_over_all():
    labels = np.array([-100, -100, -100, 0])
    got = masked_lm_loss(LOGITS, labels)
    want = float(-np.log(np.exp(0.0) / np.exp([0.0, 1.0, 2.0]).sum()))
    assert abs(got - want) < 1e-9, f"expected {want} for one selected position, got {got}"


def test_all_selected_matches_plain_cross_entropy():
    labels = np.array([0, 1, 2, 2])
    p = np.exp(LOGITS) / np.exp(LOGITS).sum(axis=1, keepdims=True)
    want = float(-np.log(p[np.arange(4), labels]).mean())
    got = masked_lm_loss(LOGITS, labels)
    assert abs(got - want) < 1e-9, f"expected {want}, got {got}"


def test_label_zero_is_a_real_label():
    got = masked_lm_loss(np.array([[0.0, 5.0], [0.0, 5.0]]), np.array([0, 0]))
    want = float(np.log1p(np.exp(5.0)))
    assert abs(got - want) < 1e-9, f"label 0 must count as a target: expected {want}, got {got}"


def test_stable_for_large_logits():
    big = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]])
    got = masked_lm_loss(big, np.array([0, 2]))
    assert np.isfinite(got), f"expected a finite loss, got {got}"
    assert abs(got - 500.0) < 1e-6, f"expected (0 + 1000) / 2 = 500, got {got}"


def test_no_selected_positions_returns_zero():
    got = masked_lm_loss(LOGITS, np.full(4, -100))
    assert got == 0.0, f"expected 0.0 with no selected positions, got {got}"


def test_softmax_runs_over_the_vocabulary_axis():
    logits = np.array([[3.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    got = masked_lm_loss(logits, np.array([0, -100, -100]))
    want = float(np.log1p(np.exp(-3.0)))
    assert abs(got - want) < 1e-9, f"expected {want}, got {got}"
