import numpy as np


def squashed_log_prob(u, mu, std):
    """Return log pi(tanh(u)) for u ~ N(mu, std^2), summed over the last axis."""
    raise NotImplementedError
