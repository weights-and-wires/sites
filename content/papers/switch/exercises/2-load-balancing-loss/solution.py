import numpy as np


def expert_fractions(p):
    """p: (T, N) router probabilities.

    Returns (f, P), each of shape (N,):
      f[i] fraction of tokens whose argmax is expert i (Equation 5)
      P[i] mean router probability of expert i (Equation 6)
    """
    T, N = p.shape
    idx = np.argmax(p, axis=1)
    f = np.bincount(idx, minlength=N) / T
    P = p.mean(axis=0)
    return f, P


def load_balancing_loss(p, alpha=1e-2):
    """Equation 4: alpha * N * sum_i f_i * P_i."""
    N = p.shape[1]
    f, P = expert_fractions(p)
    return alpha * N * float(np.sum(f * P))
