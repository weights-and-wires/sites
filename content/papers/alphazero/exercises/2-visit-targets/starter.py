import numpy as np


def visits_to_policy(N, tau):
    """Turn root visit counts into a policy target pi proportional to N ** (1 / tau).

    N: visit counts, shape (K,), may contain zeros. tau: temperature.
    When tau == 0, return a greedy target: equal mass on every move that has
    the maximum count, and 0 elsewhere. Returns a float array that sums to 1.
    """
    raise NotImplementedError
