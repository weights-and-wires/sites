import math

import numpy as np


def soft_update(target, source, tau):
    """Return a new dict with tau * source[k] + (1 - tau) * target[k] for every key k."""
    raise NotImplementedError


def updates_to_shrink(tau, frac):
    """Return the smallest integer n with (1 - tau) ** n <= frac."""
    raise NotImplementedError
