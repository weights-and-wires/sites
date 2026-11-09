import numpy as np
from submission import kl_k3, grpo_objective

EPS, BETA = 0.2, 0.04


def page_batch():
    ratio = np.array([[1.3, 0.9, 1.0, 1.0], [1.1, 1.0, 0.7, 1.25]])
    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])
    logp_old = np.full((2, 4), -1.0)
    logp_new = logp_old + np.log(ratio)
    d = np.array([[0.1, -0.2, 0.0, 0.0], [0.0, 0.3, -0.1, 0.2]])
    return logp_new, logp_old, logp_new + d, np.array([1.0, -1.0]), mask


def test_k3_single_token():
    got = float(kl_k3(np.log(0.5), np.log(0.25)))
    assert abs(got - 0.1931) < 1e-3, f"pi=0.5, pi_ref=0.25: expected 0.1931, got {got}"


def test_k3_direction():
    got = float(kl_k3(np.log(0.25), np.log(0.5)))
    assert abs(got - 0.3069) < 1e-3, f"pi=0.25, pi_ref=0.5: expected 0.3069 (ratio is pi_ref/pi), got {got}"


def test_k3_nonnegative_and_zero_at_equal():
    x = np.log(np.array([0.05, 0.3, 0.9]))
    y = np.log(np.array([0.6, 0.3, 0.01]))
    k = kl_k3(x, y)
    assert np.all(k >= 0), f"the estimator must be non-negative, got {k}"
    assert abs(float(k[1])) < 1e-12, f"it must be 0 when pi == pi_ref, got {k[1]}"


def test_k3_is_unbiased_for_kl():
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.2, 0.5, 0.3])
    want = float(np.sum(p * np.log(p / q)))
    got = float(np.sum(p * kl_k3(np.log(p), np.log(q))))
    assert abs(got - want) < 1e-9, f"E_pi[k3] must equal KL(pi || pi_ref) = {want:.4f}, got {got:.4f}"


def test_objective_page_example():
    lp, lo, lr, adv, mask = page_batch()
    got = float(grpo_objective(lp, lo, lr, adv, mask, EPS, BETA))
    assert abs(got - 0.0056) < 1e-3, f"expected 0.0056, got {got:.4f}"


def test_objective_without_kl():
    lp, lo, lr, adv, mask = page_batch()
    got = float(grpo_objective(lp, lo, lr, adv, mask, EPS, 0.0))
    assert abs(got - 0.0062) < 1e-3, f"expected 0.0062 with beta=0, got {got:.4f}"


def test_length_normalization_is_per_output():
    # Ratio 1 and no KL: every token term equals the output's advantage, so the
    # objective is the plain mean of the advantages whatever the lengths are.
    lp = np.zeros((3, 5))
    mask = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    adv = np.array([3.0, 0.0, -1.0])
    got = float(grpo_objective(lp, lp, lp, adv, mask, EPS, BETA))
    assert abs(got - (3.0 + 0.0 - 1.0) / 3) < 1e-9, f"expected {2 / 3:.4f} (mean of advantages), got {got:.4f}"


def test_padding_is_ignored():
    lp, lo, lr, adv, mask = page_batch()
    lp2, lo2, lr2 = lp.copy(), lo.copy(), lr.copy()
    lp2[0, 2:], lo2[0, 2:], lr2[0, 2:] = 5.0, -3.0, 2.0
    a = float(grpo_objective(lp, lo, lr, adv, mask, EPS, BETA))
    b = float(grpo_objective(lp2, lo2, lr2, adv, mask, EPS, BETA))
    assert abs(a - b) < 1e-12, f"values at masked positions must not matter: expected {a}, got {b}"


def test_clipping_is_pessimistic():
    lo = np.zeros((1, 1))
    mask = np.ones((1, 1))
    up = np.log(np.full((1, 1), 1.5))
    down = np.log(np.full((1, 1), 0.5))
    cases = [
        (up, 1.0, 1.2, "A>0, r=1.5: gain is capped at (1+eps)A"),
        (up, -1.0, -1.5, "A<0, r=1.5: the full penalty stays"),
        (down, 1.0, 0.5, "A>0, r=0.5: the unclipped (lower) term stays"),
        (down, -1.0, -0.8, "A<0, r=0.5: the penalty is capped at (1-eps)A"),
    ]
    for lp, a, want, label in cases:
        got = float(grpo_objective(lp, lo, lp, np.array([a]), mask, EPS, 0.0))
        assert abs(got - want) < 1e-9, f"{label}: expected {want}, got {got}"


def test_kl_term_lowers_the_objective():
    lp, lo, lr, adv, mask = page_batch()
    j0 = float(grpo_objective(lp, lo, lr, adv, mask, EPS, 0.0))
    j1 = float(grpo_objective(lp, lo, lr, adv, mask, EPS, 1.0))
    assert j1 < j0, f"a positive beta must subtract the KL estimate: expected below {j0:.4f}, got {j1:.4f}"
