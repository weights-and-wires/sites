import numpy as np
from submission import reward_margin, dpo_loss

PI_W, PI_L = np.array([-12.0]), np.array([-15.0])
REF_W, REF_L = np.array([-13.0]), np.array([-14.0])


def test_margin_value():
    got = reward_margin(PI_W, PI_L, REF_W, REF_L, 0.1)
    assert np.allclose(got, [0.2]), f"expected margin [0.2] (0.1 * (1 - (-1))), got {got}"


def test_loss_value_beta_01():
    got = dpo_loss(PI_W, PI_L, REF_W, REF_L, 0.1)
    assert np.allclose(got, [0.5981], atol=1e-4), f"expected [0.5981], got {got}"


def test_loss_value_beta_05():
    got = dpo_loss(PI_W, PI_L, REF_W, REF_L, 0.5)
    assert np.allclose(got, [0.3133], atol=1e-4), f"expected [0.3133] (margin 1.0), got {got}"


def test_policy_equal_to_reference_gives_log_2():
    lp = np.array([-7.0, -3.0, -20.0])
    lq = np.array([-9.0, -4.0, -25.0])
    got = dpo_loss(lp, lq, lp, lq, 0.3)
    assert np.allclose(got, np.log(2)), f"expected log 2 = {np.log(2):.4f} for every pair, got {got}"


def test_returns_one_loss_per_pair():
    pi_w = np.array([-1.0, -2.0, -3.0])
    got = dpo_loss(pi_w, pi_w - 1, pi_w - 0.5, pi_w - 2, 0.1)
    assert np.shape(got) == (3,), f"expected shape (3,), got {np.shape(got)}; return per-pair losses, not a mean"


def test_reference_enters_with_opposite_sign():
    # Raising the reference log-probability of the winner makes the same policy look worse.
    base = dpo_loss(PI_W, PI_L, REF_W, REF_L, 0.1)
    worse = dpo_loss(PI_W, PI_L, REF_W + 5, REF_L, 0.1)
    better = dpo_loss(PI_W, PI_L, REF_W, REF_L + 5, 0.1)
    assert worse > base, f"raising ref_w should raise the loss: {worse} vs {base}"
    assert better < base, f"raising ref_l should lower the loss: {better} vs {base}"


def test_direction_of_policy_changes():
    base = dpo_loss(PI_W, PI_L, REF_W, REF_L, 0.1)
    assert dpo_loss(PI_W + 1, PI_L, REF_W, REF_L, 0.1) < base, "raising pi_w must lower the loss"
    assert dpo_loss(PI_W, PI_L + 1, REF_W, REF_L, 0.1) > base, "raising pi_l must raise the loss"


def test_wrong_order_costs_more_than_log_2():
    got = dpo_loss(np.array([-14.5]), np.array([-13.5]), REF_W, REF_L, 0.1)
    assert np.allclose(got, [0.7981], atol=1e-4), f"expected [0.7981] (margin -0.2), got {got}"


def test_common_shift_of_all_log_probs_changes_nothing():
    a = dpo_loss(PI_W, PI_L, REF_W, REF_L, 0.1)
    b = dpo_loss(PI_W - 40, PI_L - 40, REF_W - 40, REF_L - 40, 0.1)
    assert np.allclose(a, b), f"only the margin matters; got {a} and {b}"


def test_stable_for_huge_margins():
    lose = dpo_loss(np.array([-2000.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), 1.0)
    win = dpo_loss(np.array([0.0]), np.array([-2000.0]), np.array([0.0]), np.array([0.0]), 1.0)
    assert np.all(np.isfinite(lose)), f"margin -2000 must give a finite loss, got {lose}"
    assert np.allclose(lose, [2000.0]), f"expected [2000.0] for margin -2000, got {lose}"
    assert np.allclose(win, [0.0], atol=1e-12), f"expected [0.0] for margin +2000, got {win}"
