def slo_attainment(ttft, tpot, ttft_slo, tpot_slo):
    """Fraction of requests that meet both the TTFT and the TPOT SLO."""
    raise NotImplementedError


def max_goodput(attainment_at, target=0.9, lo=0.0, hi=100.0, iters=40):
    """Largest rate in [lo, hi] whose attainment is at least target, by bisection."""
    raise NotImplementedError
