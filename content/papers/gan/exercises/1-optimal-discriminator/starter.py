import numpy as np


def optimal_discriminator(p_data, p_g):
    """Return D*(x) = p_data / (p_data + p_g); return 0.5 where both are 0."""
    raise NotImplementedError


def value(p_data, p_g, D):
    """Return V(G, D) = sum p_data log D + p_g log(1 - D), with 0 log 0 = 0."""
    raise NotImplementedError


def jsd(p, q):
    """Return the Jensen-Shannon divergence (natural log) of two probability vectors."""
    raise NotImplementedError


def virtual_criterion(p_data, p_g):
    """Return C(G) = V(G, D*_G)."""
    raise NotImplementedError
