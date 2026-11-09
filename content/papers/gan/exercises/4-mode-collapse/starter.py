import numpy as np


def histogram(samples, edges):
    """Return the probability mass per bin; samples outside the edges are dropped."""
    raise NotImplementedError


def js_divergence(p, q):
    """Return the Jensen-Shannon divergence (natural log) of two histograms."""
    raise NotImplementedError


def modes_covered(samples, modes, radius, min_frac):
    """Return how many modes have at least min_frac of all samples within radius."""
    raise NotImplementedError
