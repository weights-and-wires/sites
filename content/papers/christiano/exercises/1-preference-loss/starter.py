import numpy as np


def preference_prob(ret1, ret2, eps=0.0):
    """Return P_hat[sigma1 > sigma2] (Equation 1) with the eps adjustment of Section 2.2.3."""
    raise NotImplementedError


def preference_loss(ret1, ret2, mu, eps=0.0):
    """Return the mean cross-entropy between mu = mu(1) and P_hat. Stay finite for large gaps."""
    raise NotImplementedError
