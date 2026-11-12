import math

import numpy as np


def soft_update(target, source, tau):
    """Return a new dict with tau * source[k] + (1 - tau) * target[k] for every key k."""
    return {k: tau * source[k] + (1.0 - tau) * target[k] for k in target}


def updates_to_shrink(tau, frac):
    """Return the smallest integer n with (1 - tau) ** n <= frac."""
    n = math.ceil(math.log(frac) / math.log(1.0 - tau))
    return int(n)
