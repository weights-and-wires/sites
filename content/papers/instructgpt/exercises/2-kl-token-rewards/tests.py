import numpy as np
from submission import token_rewards

LP_RL = np.array([-1.2, -0.8, -2.1, -0.5, -1.0])
LP_SFT = np.array([-1.5, -1.0, -1.9, -0.9, -1.4])


def test_values_beta_0_1():
    got = np.asarray(token_rewards(LP_RL, LP_SFT, 1.8, 0.1))
    want = np.array([-0.03, -0.02, 0.02, -0.04, 1.76])
    assert got.shape == (5,), f"expected shape (5,), got {got.shape}"
    assert np.allclose(got, want, atol=1e-9), f"expected {want}, got {got}"


def test_score_only_at_last_token():
    got = np.asarray(token_rewards(LP_RL, LP_SFT, 100.0, 0.1))
    assert np.allclose(got[:-1], [-0.03, -0.02, 0.02, -0.04]), f"earlier tokens must not see the RM score, got {got[:-1]}"
    assert abs(got[-1] - 99.96) < 1e-9, f"expected 100 - 0.1 * 0.4 = 99.96 at the last token, got {got[-1]}"


def test_penalty_sign():
    got = np.asarray(token_rewards(np.array([-0.5, -0.5]), np.array([-1.5, -1.5]), 0.0, 1.0))
    assert np.allclose(got, [-1.0, -1.0]), f"a policy that is more likely than the SFT model must be penalized, expected [-1, -1], got {got}"


def test_sum_matches_equation_2():
    beta, score = 0.02, 1.8
    got = np.asarray(token_rewards(LP_RL, LP_SFT, score, beta)).sum()
    want = score - beta * (LP_RL.sum() - LP_SFT.sum())
    assert abs(got - want) < 1e-9, f"expected the sum r - beta * log(pi_RL(y|x) / pi_SFT(y|x)) = {want}, got {got}"


def test_same_policy_gives_only_the_score():
    got = np.asarray(token_rewards(LP_RL, LP_RL, 0.7, 0.5))
    assert np.allclose(got, [0, 0, 0, 0, 0.7]), f"expected [0, 0, 0, 0, 0.7], got {got}"


def test_beta_zero_removes_penalty():
    got = np.asarray(token_rewards(LP_RL, LP_SFT, 1.8, 0.0))
    assert np.allclose(got, [0, 0, 0, 0, 1.8]), f"expected [0, 0, 0, 0, 1.8], got {got}"


def test_single_token_response():
    got = np.asarray(token_rewards(np.array([-0.5]), np.array([-1.0]), 2.0, 0.2))
    assert got.shape == (1,) and abs(got[0] - 1.9) < 1e-9, f"expected [1.9] for one token, got {got}"


def test_inputs_not_modified():
    a, b = LP_RL.copy(), LP_SFT.copy()
    token_rewards(a, b, 1.8, 0.1)
    assert np.array_equal(a, LP_RL) and np.array_equal(b, LP_SFT), f"inputs changed: {a}, {b}"
