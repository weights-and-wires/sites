import math


def replicas_needed(rate, goodput):
    """Smallest number of instances, each with this goodput, that serve `rate` requests per second."""
    raise NotImplementedError


def best_split(budget, g_prefill, g_decode):
    """Return (rate, n_p, n_d): the split of at most `budget` GPUs with the highest system rate."""
    raise NotImplementedError
