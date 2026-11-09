import math


def online_softmax_topk(xs, k):
    """One pass over xs. Return (v, z): top-k softmax probabilities and their indices."""
    m, d = -math.inf, 0.0
    u = [-math.inf] * (k + 1)
    p = [-1] * (k + 1)
    for j, x in enumerate(xs):
        m_new = max(m, x)
        d = d * math.exp(m - m_new) + math.exp(x - m_new)
        m = m_new
        u[k], p[k] = x, j
        i = k - 1
        while i >= 0 and u[i] < u[i + 1]:
            u[i], u[i + 1] = u[i + 1], u[i]
            p[i], p[i + 1] = p[i + 1], p[i]
            i -= 1
    v = [math.exp(u[i] - m) / d for i in range(k)]
    return v, p[:k]
