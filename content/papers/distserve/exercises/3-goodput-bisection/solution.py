def slo_attainment(ttft, tpot, ttft_slo, tpot_slo):
    """Fraction of requests that meet both the TTFT and the TPOT SLO."""
    good = sum(1 for a, b in zip(ttft, tpot) if a <= ttft_slo and b <= tpot_slo)
    return good / len(ttft)


def max_goodput(attainment_at, target=0.9, lo=0.0, hi=100.0, iters=40):
    """Largest rate in [lo, hi] whose attainment is at least target, by bisection."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if attainment_at(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo
