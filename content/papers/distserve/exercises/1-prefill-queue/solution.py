INF = float("inf")


def avg_ttft(rate, d):
    """Equation 1: average TTFT of one GPU, an M/D/1 queue. inf when rate * d >= 1."""
    rho = rate * d
    if rho >= 1:
        return INF
    return d + rate * d * d / (2 * (1 - rho))


def avg_ttft_inter(rate, d):
    """Equation 2: 2-way inter-op parallelism. inf when rate * d >= 2."""
    rho = rate * d
    if rho >= 2:
        return INF
    return d + rate * d * d / (4 * (2 - rho))


def avg_ttft_intra(rate, d, k):
    """Equation 3: 2-way intra-op parallelism with speedup k. inf when rate * d >= k."""
    rho = rate * d
    if rho >= k:
        return INF
    return d / k + rate * d * d / (2 * k * (k - rho))


def better_parallelism(rate, d, k):
    """Return "intra" or "inter", whichever has the lower average TTFT ("intra" on a tie)."""
    if avg_ttft_inter(rate, d) < avg_ttft_intra(rate, d, k):
        return "inter"
    return "intra"
