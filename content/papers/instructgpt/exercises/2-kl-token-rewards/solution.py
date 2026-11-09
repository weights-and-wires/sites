import numpy as np


def token_rewards(logp_rl, logp_sft, rm_score, beta):
    """Return the per-token reward array of shape (T,).

    logp_rl, logp_sft: log-probabilities of the sampled tokens, shape (T,).
    rm_score: the reward model's scalar for the whole response.
    beta: the KL reward coefficient.
    """
    r = -beta * (np.asarray(logp_rl, dtype=float) - np.asarray(logp_sft, dtype=float))
    r[-1] += rm_score
    return r
