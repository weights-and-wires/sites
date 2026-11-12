import numpy as np

# Appendix K, Table 2 (alpha = beta).
A, B, E, ALPHA = 4.299e3, 1.806e4, 2.7648, 0.4965


def p_star(gamma_bar):
    """Return the compute-optimal precision P* = x * gamma_bar.

    x is the positive root of exp(x) - 1 = 3 x, found by bisection on [1, 10].
    """
    lo, hi = 1.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.exp(mid) - 1.0 - 3.0 * mid < 0:
            lo = mid
        else:
            hi = mid
    return gamma_bar * 0.5 * (lo + hi)


def best_bits(c, bits, gamma_bar):
    """Return (p, loss): the precision in `bits` with the lowest loss at budget c.

    For each p, use the cost C = (6/16) N D p, so D = c / (6/16 * N * p), and
    the loss A (N (1 - exp(-p/gamma_bar))**3)**-ALPHA + B D**-ALPHA + E.
    Take the lowest loss over a grid of N, then the lowest over `bits`.
    """
    n = np.logspace(6, 13, 4000)
    best = (None, np.inf)
    for p in bits:
        d = c / (6.0 / 16.0 * n * p)
        loss = A * (n * (1.0 - np.exp(-p / gamma_bar)) ** 3) ** -ALPHA + B * d**-ALPHA + E
        if loss.min() < best[1]:
            best = (p, float(loss.min()))
    return best
