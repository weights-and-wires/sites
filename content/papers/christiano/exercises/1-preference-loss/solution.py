import numpy as np


def _log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def preference_prob(ret1, ret2, eps=0.0):
    """Return P_hat[sigma1 > sigma2] (Equation 1) with the eps adjustment of Section 2.2.3."""
    diff = np.asarray(ret1, dtype=float) - np.asarray(ret2, dtype=float)
    p = np.exp(_log_sigmoid(diff))
    return (1.0 - eps) * p + eps / 2.0


def preference_loss(ret1, ret2, mu, eps=0.0):
    """Return the mean cross-entropy between mu = mu(1) and P_hat. Stay finite for large gaps."""
    ret1 = np.asarray(ret1, dtype=float)
    ret2 = np.asarray(ret2, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if eps == 0.0:
        lp1 = _log_sigmoid(ret1 - ret2)
        lp2 = _log_sigmoid(ret2 - ret1)
    else:
        lp1 = np.log(preference_prob(ret1, ret2, eps))
        lp2 = np.log(preference_prob(ret2, ret1, eps))
    return float(np.mean(-(mu * lp1 + (1.0 - mu) * lp2)))
