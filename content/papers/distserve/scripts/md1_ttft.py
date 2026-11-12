"""M/D/1 model of a prefill instance and the effect of parallelism (Section 3.1, Equations 1 to 3).

Run from the paper folder: python3 scripts/md1_ttft.py

Notation is the paper's: R is the Poisson arrival rate, D is the execution time
of one prefill on one GPU, and K is the speedup of 2-way intra-op parallelism,
with 1 < K < 2. The script also runs a queue simulation to check Equation 1.
"""

import numpy as np


def avg_ttft(r, d):
    """Equation 1: Avg_TTFT = D + R D^2 / (2 (1 - R D)), valid for R D < 1."""
    if r * d >= 1:
        return float("inf")
    return d + r * d * d / (2 * (1 - r * d))


def avg_ttft_inter(r, d):
    """Equation 2: 2-way inter-op, Avg_TTFT = D + R D^2 / (4 (2 - R D))."""
    if r * d >= 2:
        return float("inf")
    return d + r * d * d / (4 * (2 - r * d))


def avg_ttft_intra(r, d, k):
    """Equation 3: 2-way intra-op, Avg_TTFT = D/K + R D^2 / (2 K (K - R D))."""
    if r * d >= k:
        return float("inf")
    return d / k + r * d * d / (2 * k * (k - r * d))


def crossover(d, k):
    """Rate at which Equation 2 and Equation 3 are equal (bisection)."""
    lo, hi = 1e-9, min(2.0, k) / d - 1e-9
    f = lambda r: avg_ttft_intra(r, d, k) - avg_ttft_inter(r, d)
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return lo


def simulate_fcfs(r, d, n=200000, seed=0):
    """Mean sojourn time of a single FCFS server with Poisson arrivals and a
    constant service time d (Lindley recursion)."""
    rng = np.random.RandomState(seed)
    gaps = rng.exponential(1.0 / r, n)
    wait, total = 0.0, 0.0
    for g in gaps:
        wait = max(0.0, wait + d - g)
        total += wait + d
    return total / n


if __name__ == "__main__":
    D = 1.0  # time unit: one prefill on one GPU
    print("Equation 1 vs. a queue simulation (D = 1, 200000 requests)")
    for rho in (0.2, 0.5, 0.8, 0.9):
        print(f"  R D = {rho}: formula {avg_ttft(rho, D):.4f}  simulation {simulate_fcfs(rho, D):.4f}")

    # Equation 2 follows from Equation 1 with D_m = D / 2 and D_s = D.
    for rho in (0.5, 1.0, 1.5):
        via_eq1 = D + rho * (D / 2) ** 2 / (2 * (1 - rho * D / 2))
        print(f"Equation 2 at R D = {rho}: {avg_ttft_inter(rho, D):.4f} (from Equation 1 with D_m = D/2: {via_eq1:.4f})")

    print("Avg TTFT in units of D, for 2-way parallelism")
    print("   R D | single GPU | inter-op | intra-op K=1.6 | intra-op K=1.9 | best (K=1.6)")
    for rho in (0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5):
        one = avg_ttft(rho, D)
        inter = avg_ttft_inter(rho, D)
        a, b = avg_ttft_intra(rho, D, 1.6), avg_ttft_intra(rho, D, 1.9)
        best = "intra-op" if a < inter else "inter-op"
        print(f"  {rho:4.1f} | {one:10.3f} | {inter:8.3f} | {a:14.3f} | {b:14.3f} | {best}")

    for k in (1.5, 1.6, 1.7, 1.8, 1.9):
        print(f"crossover rate for K = {k}: R D = {crossover(D, k):.3f}")
