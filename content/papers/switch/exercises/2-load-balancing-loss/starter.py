import numpy as np


def expert_fractions(p):
    """p: (T, N) router probabilities.

    Returns (f, P), each of shape (N,):
      f[i] fraction of tokens whose argmax is expert i (Equation 5)
      P[i] mean router probability of expert i (Equation 6)
    """
    raise NotImplementedError


def load_balancing_loss(p, alpha=1e-2):
    """Equation 4: alpha * N * sum_i f_i * P_i."""
    raise NotImplementedError
