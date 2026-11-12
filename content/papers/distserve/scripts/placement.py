"""Place prefill and decoding instances under a GPU budget (Section 4.1, Algorithm 1).

Run from the paper folder: python3 scripts/placement.py

Algorithm 1 finds the best configuration of one prefill instance and one decoding
instance by simulation, then replicates each to match the traffic. This toy has
one GPU per instance, so the search is over the counts n_p and n_d. The goodput
of one prefill GPU and of one decoding GPU come from the toy simulator in
goodput_sim.py, each measured against its own SLO only.
"""

import heapq

import numpy as np

import goodput_sim as gs


def prefill_goodput():
    """Rate that one prefill GPU sustains with TTFT within the SLO for TARGET of requests."""
    def run(rate):
        ttft, _ = gs.disaggregated(rate, 1, 100)  # 100 decode GPUs: never the limit
        return ttft, np.zeros_like(ttft)  # TPOT is not checked
    return gs.goodput(run)


def decode_goodput():
    """Rate that one decoding GPU sustains with TPOT within the SLO for TARGET of requests."""
    def run(rate):
        # Requests reach the decoding GPU as soon as they arrive.
        arr = gs.arrivals(rate)
        slots = [0.0] * gs.B
        heapq.heapify(slots)
        tpot = []
        for a in arr:
            begin = max(a, heapq.heappop(slots))
            end = begin + (gs.L - 1) * gs.D
            heapq.heappush(slots, end)
            tpot.append((end - a) / (gs.L - 1))
        return np.zeros(len(arr)), np.array(tpot)  # TTFT is not checked
    return gs.goodput(run)


def best_split(budget, g_p, g_d):
    """Brute force over n_p + n_d <= budget. Returns (system goodput, n_p, n_d)."""
    best = (0.0, 0, 0)
    for n_p in range(1, budget):
        for n_d in range(1, budget - n_p + 1):
            rate = min(n_p * g_p, n_d * g_d)
            if rate > best[0] + 1e-12:
                best = (rate, n_p, n_d)
    return best


if __name__ == "__main__":
    g_p, g_d = prefill_goodput(), decode_goodput()
    print(f"one prefill GPU: {g_p:.2f} req/s; one decoding GPU: {g_d:.2f} req/s")
    for budget in (2, 3, 4, 5, 8):
        rate, n_p, n_d = best_split(budget, g_p, g_d)
        print(f"budget {budget}: {n_p} prefill + {n_d} decoding, system goodput {rate:.2f} req/s, {rate / (n_p + n_d):.2f} per GPU")
    rate, n_p, n_d = best_split(8, g_p, g_d)
    sim = gs.goodput(lambda r: gs.disaggregated(r, n_p, n_d))
    print(f"simulated goodput of {n_p}P+{n_d}D: {sim:.2f} req/s (min rule predicted {rate:.2f})")
    col = gs.goodput(gs.colocated)
    print(f"8 colocated GPUs: {8 * col:.2f} req/s; best 8-GPU split: {rate:.2f} req/s ({rate / (8 * col):.2f}x)")
