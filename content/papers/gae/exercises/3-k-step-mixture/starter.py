import numpy as np


def k_step_advantage(rewards, values, gamma, t, k):
    """A_t^(k) = -V(s_t) + sum_{l<k} gamma^l r_{t+l} + gamma^k V(s_{t+k}) (Equation 14).

    The trajectory ends after len(rewards) steps with V = 0. If t + k runs past the end,
    use k = len(rewards) - t.
    """
    raise NotImplementedError


def gae_mixture(rewards, values, gamma, lam):
    """GAE at every t, as a weighted average of the k-step estimators (Section 3, before Equation 16).

    For n = T - t steps left: weight (1 - lam) * lam**(k - 1) on k = 1..n-1, and the leftover
    weight lam**(n - 1) on k = n.
    """
    raise NotImplementedError
