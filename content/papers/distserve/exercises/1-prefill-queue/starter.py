INF = float("inf")


def avg_ttft(rate, d):
    """Equation 1: average TTFT of one GPU, an M/D/1 queue. inf when rate * d >= 1."""
    raise NotImplementedError


def avg_ttft_inter(rate, d):
    """Equation 2: 2-way inter-op parallelism. inf when rate * d >= 2."""
    raise NotImplementedError


def avg_ttft_intra(rate, d, k):
    """Equation 3: 2-way intra-op parallelism with speedup k. inf when rate * d >= k."""
    raise NotImplementedError


def better_parallelism(rate, d, k):
    """Return "intra" or "inter", whichever has the lower average TTFT ("intra" on a tie)."""
    raise NotImplementedError
