import numpy as np


def optimal_policy(ref_probs, r, beta):
    """Return (pi, log_Z) for Equation 4. pi sums to 1.

    ref_probs: 1-D array, the reference policy over the responses.
    r: 1-D array, the reward of each response.
    log_Z must be a float and the function must not overflow for large r / beta.
    """
    raise NotImplementedError


def implicit_reward(pi, ref_probs, beta):
    """Return beta * log(pi / ref_probs), the implicit reward without the log Z term."""
    raise NotImplementedError
