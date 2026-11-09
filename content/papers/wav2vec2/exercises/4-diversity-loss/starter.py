import numpy as np


def diversity_loss(logits):
    """Return L_d from Equation 4.

    logits: array of shape (N, G, V). The softmax is over V, with no Gumbel noise
    and no temperature. Average it over N, then return
    sum_g sum_v pbar log pbar / (G * V), using the natural logarithm.
    """
    raise NotImplementedError
