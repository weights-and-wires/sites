import math


def online_normalizer(xs):
    """Read the iterable xs once. Return (m, d) with m = max(xs) and d = sum(exp(x - m))."""
    raise NotImplementedError


def online_softmax(x):
    """Finished for you: one pass for (m, d), then one pass for the outputs."""
    x = list(x)
    m, d = online_normalizer(iter(x))
    return [math.exp(v - m) / d for v in x]
