import numpy as np
from submission import ppo_ptx_objective

LP_RL = np.array([[-1.2, -0.8, -2.1, -0.5, -1.0], [-0.6, -0.9, -1.1, -0.7, -0.4]])
LP_SFT = np.array([[-1.5, -1.0, -1.9, -0.9, -1.4], [-0.9, -0.8, -1.3, -0.5, -0.6]])
RM = np.array([1.8, 0.4])
PRE = np.array([-2.3, -2.7, -2.0])


def test_one_sample_no_pretrain_term():
    got = ppo_ptx_objective(RM[:1], LP_RL[:1], LP_SFT[:1], PRE, 0.02, 0.0)
    assert abs(got - 1.778) < 1e-9, f"expected 1.8 - 0.02 * 1.1 = 1.778, got {got}"


def test_batch_mean():
    # Sample 2 log ratio: 0.3 - 0.1 + 0.2 - 0.2 + 0.2 = 0.4.
    got = ppo_ptx_objective(RM, LP_RL, LP_SFT, PRE, 0.1, 0.0)
    want = ((1.8 - 0.1 * 1.1) + (0.4 - 0.1 * 0.4)) / 2
    assert abs(got - want) < 1e-9, f"expected {want}, got {got}"


def test_log_ratio_is_summed_over_tokens():
    got = ppo_ptx_objective(RM, LP_RL, LP_SFT, PRE, 1.0, 0.0)
    want = ((1.8 - 1.1) + (0.4 - 0.4)) / 2
    assert abs(got - want) < 1e-9, f"expected {want} with the sum over tokens, got {got}"


def test_pretraining_term_added_and_scaled():
    base = ppo_ptx_objective(RM, LP_RL, LP_SFT, PRE, 0.02, 0.0)
    got = ppo_ptx_objective(RM, LP_RL, LP_SFT, PRE, 0.02, 27.8)
    want = base + 27.8 * (-2.3 - 2.7 - 2.0) / 3
    assert abs(got - want) < 1e-9, f"expected {want}, got {got}"


def test_pretraining_term_is_a_mean_not_a_sum():
    a = ppo_ptx_objective(RM, LP_RL, LP_SFT, PRE, 0.0, 1.0)
    b = ppo_ptx_objective(RM, LP_RL, LP_SFT, np.concatenate([PRE, PRE]), 0.0, 1.0)
    assert abs(a - b) < 1e-9, f"repeating the pretraining texts must not change the mean: {a} vs {b}"


def test_better_pretraining_likelihood_raises_objective():
    lo = ppo_ptx_objective(RM, LP_RL, LP_SFT, PRE - 1.0, 0.02, 27.8)
    hi = ppo_ptx_objective(RM, LP_RL, LP_SFT, PRE, 0.02, 27.8)
    assert hi > lo, f"a higher log-likelihood must raise the objective: {hi} vs {lo}"
    assert abs((hi - lo) - 27.8) < 1e-9, f"expected a difference of gamma = 27.8, got {hi - lo}"


def test_same_policy_removes_the_kl_term():
    got = ppo_ptx_objective(RM, LP_RL, LP_RL, PRE, 5.0, 0.0)
    assert abs(got - 1.1) < 1e-9, f"expected the mean score 1.1, got {got}"


def test_returns_python_float():
    got = ppo_ptx_objective(RM, LP_RL, LP_SFT, PRE, 0.02, 27.8)
    assert isinstance(got, float), f"expected a Python float, got {type(got)}"
