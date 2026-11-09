import numpy as np


def token_rewards(logp_rl, logp_sft, rm_score, beta):
    """Return the per-token reward array of shape (T,).

    logp_rl, logp_sft: log-probabilities of the sampled tokens, shape (T,).
    rm_score: the reward model's scalar for the whole response.
    beta: the KL reward coefficient.
    """
    raise NotImplementedError
