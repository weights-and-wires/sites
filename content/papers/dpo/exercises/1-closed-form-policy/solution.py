import numpy as np


def optimal_policy(ref_probs, r, beta):
    """Return (pi, log_Z) for Equation 4. pi sums to 1."""
    ref_probs = np.asarray(ref_probs, dtype=float)
    r = np.asarray(r, dtype=float)
    log_weights = np.log(ref_probs) + r / beta
    m = log_weights.max()
    log_Z = float(m + np.log(np.sum(np.exp(log_weights - m))))
    pi = np.exp(log_weights - log_Z)
    return pi, log_Z


def implicit_reward(pi, ref_probs, beta):
    """Return beta * log(pi / ref_probs)."""
    return beta * (np.log(np.asarray(pi, dtype=float)) - np.log(np.asarray(ref_probs, dtype=float)))
