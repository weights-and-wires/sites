import numpy as np


def ppo_ptx_objective(rm_scores, logp_rl, logp_sft, pre_logp, beta, gamma):
    """Return the sample estimate of Equation 2 as a float.

    rm_scores: shape (B,), reward-model score of each sampled response.
    logp_rl, logp_sft: shape (B, T), log-probabilities of the sampled tokens.
    pre_logp: shape (P,), log pi_RL of each pretraining text.
    """
    log_ratio = (np.asarray(logp_rl, dtype=float) - np.asarray(logp_sft, dtype=float)).sum(axis=1)
    rl_term = np.mean(np.asarray(rm_scores, dtype=float) - beta * log_ratio)
    pre_term = np.mean(np.asarray(pre_logp, dtype=float))
    return float(rl_term + gamma * pre_term)
