import math


def replicas_needed(rate, goodput):
    """Smallest number of instances, each with this goodput, that serve `rate` requests per second."""
    return math.ceil(rate / goodput - 1e-9)


def best_split(budget, g_prefill, g_decode):
    """Return (rate, n_p, n_d): the split of at most `budget` GPUs with the highest system rate."""
    best = (0.0, 0, 0)
    for n_p in range(1, budget):
        for n_d in range(1, budget - n_p + 1):
            rate = min(n_p * g_prefill, n_d * g_decode)
            if rate > best[0] + 1e-9 or (abs(rate - best[0]) <= 1e-9 and n_p + n_d < best[1] + best[2]):
                best = (rate, n_p, n_d)
    return best
