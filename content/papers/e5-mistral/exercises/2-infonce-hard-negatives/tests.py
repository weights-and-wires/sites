import numpy as np
from submission import info_nce


def unit_with_cos(c):
    """A 2-D unit vector whose cosine with (1, 0) is c."""
    return np.array([c, np.sqrt(1 - c * c)])


def reference(q, dp, dh, tau):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    cand = dp / np.linalg.norm(dp, axis=1, keepdims=True)
    if dh is not None:
        cand = np.vstack([cand, dh / np.linalg.norm(dh, axis=1, keepdims=True)])
    total = 0.0
    for i in range(len(q)):
        phi = [np.exp(float(q[i] @ c) / tau) for c in cand]
        total += -np.log(phi[i] / sum(phi))
    return total / len(q)


def test_hand_value_one_hard_negative():
    q = np.array([[1.0, 0.0]])
    dp = unit_with_cos(0.80)[None]
    dh = unit_with_cos(0.78)[None]
    got = info_nce(q, dp, dh, tau=0.02)
    want = np.log(1 + np.exp(-1.0))
    assert np.isclose(got, want, atol=1e-6), f"expected ln(1 + e^-1) = {want:.4f}, got {got}"


def test_in_batch_candidate_raises_the_loss_as_in_the_reference():
    # Query 0 sees cosines 0.80 (positive), 0.78 (its hard negative), and 0.70 (the
    # other pair's positive). Query 1 is the mirror image on the other axis.
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    dp = np.array([unit_with_cos(0.80), unit_with_cos(0.80)[::-1]])
    dh = np.array([unit_with_cos(0.78), unit_with_cos(0.78)[::-1]])
    want = reference(q, dp, dh, 0.02)
    got = info_nce(q, dp, dh, tau=0.02)
    assert np.isclose(got, want, atol=1e-8), f"expected {want}, got {got}"


def test_temperature_divides_the_cosine():
    q = np.array([[1.0, 0.0]])
    dp = unit_with_cos(0.80)[None]
    dh = unit_with_cos(0.70)[None]
    got = info_nce(q, dp, dh, tau=1.0)
    want = np.log(1 + np.exp(-0.10))
    assert np.isclose(got, want, atol=1e-6), f"tau=1: expected {want:.4f}, got {got}"
    small = info_nce(q, dp, dh, tau=0.02)
    assert small < got, f"a smaller tau sharpens the softmax and lowers this loss; got {small} vs {got}"


def test_in_batch_negatives_are_used():
    rng = np.random.default_rng(3)
    q, dp = rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
    want = reference(q, dp, None, 0.05)
    got = info_nce(q, dp, None, tau=0.05)
    assert np.isclose(got, want, atol=1e-8), f"in-batch only: expected {want}, got {got}"


def test_hard_negatives_of_all_queries_are_candidates():
    rng = np.random.default_rng(4)
    q, dp, dh = rng.normal(size=(4, 6)), rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    want = reference(q, dp, dh, 0.02)
    got = info_nce(q, dp, dh, tau=0.02)
    assert np.isclose(got, want, atol=1e-8), f"expected {want}, got {got}; use all 2B candidates for each query"


def test_hard_negatives_change_the_loss():
    rng = np.random.default_rng(5)
    q, dp, dh = rng.normal(size=(4, 6)), rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    a, b = info_nce(q, dp, None, 0.02), info_nce(q, dp, dh, 0.02)
    assert b >= a - 1e-12 and not np.isclose(a, b), f"extra candidates raise the loss; got {a} without and {b} with"


def test_inputs_are_normalized_inside():
    rng = np.random.default_rng(6)
    q, dp, dh = rng.normal(size=(3, 5)), rng.normal(size=(3, 5)), rng.normal(size=(3, 5))
    a = info_nce(q, dp, dh, 0.02)
    b = info_nce(q * [[2.0], [5.0], [0.3]], dp * [[7.0], [0.1], [3.0]], dh * [[4.0], [4.0], [0.2]], 0.02)
    assert np.isclose(a, b, atol=1e-9), f"scaling the vectors must not change a cosine loss; got {a} and {b}"


def test_mean_not_sum_and_finite_for_tiny_tau():
    q = np.eye(3)
    dp = np.eye(3)[[1, 2, 0]]   # every positive is orthogonal, every wrong one scores 1
    got = info_nce(q, dp, None, tau=0.001)
    assert np.isfinite(got), f"loss is {got} at tau=0.001; subtract the maximum before exp"
    assert np.isclose(got, 1000.0, rtol=1e-3), f"expected a mean of about 1000 (logit gap 1/tau), got {got}"
