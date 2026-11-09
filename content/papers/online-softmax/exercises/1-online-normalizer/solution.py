import math


def online_normalizer(xs):
    """Read the iterable xs once. Return (m, d) with m = max(xs) and d = sum(exp(x - m))."""
    m, d = -math.inf, 0.0
    for x in xs:
        m_new = max(m, x)
        d = d * math.exp(m - m_new) + math.exp(x - m_new)
        m = m_new
    return m, d


def online_softmax(x):
    """Finished for you: one pass for (m, d), then one pass for the outputs."""
    x = list(x)
    m, d = online_normalizer(iter(x))
    return [math.exp(v - m) / d for v in x]
