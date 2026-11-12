"""Goodput of a colocated GPU vs. a disaggregated prefill/decode split (Sections 1, 2.3, 4).

Run from the paper folder: python3 scripts/goodput_sim.py

This is a toy model, not the paper's simulator (Appendix A). Every request has
the same prompt and the same output length. A prefill takes `P` seconds on one
GPU and does not batch. A decode iteration takes `D` seconds and emits one token
for every request in the batch, whatever the batch size, up to `B` requests.

- Colocated: one GPU runs prefill-first continuous batching. When a request is
  waiting, the GPU runs its prefill, and every running decode waits.
- Disaggregated: `n_p` prefill GPUs and `n_d` decode GPUs. A request goes to
  prefill GPU `i mod n_p`, then waits `transfer` seconds for the KV cache, then
  takes a decode slot on the decode GPU that frees one first.

TTFT is the time from arrival to the first token. TPOT is the time from the
first token to the last, divided by the number of later tokens, so a wait for a
decode slot counts against TPOT, as in footnote 1 of the paper. Goodput is the
largest rate at which at least `TARGET` of requests meet both SLOs (Section 1),
found by bisection.
"""

import heapq

import numpy as np

P = 0.10  # s, prefill of one request
D = 0.03  # s, one decode iteration
L = 64  # output tokens per request (Figure 1 uses 64)
B = 32  # decode slots per GPU
TTFT_SLO = 0.40  # s
TPOT_SLO = 0.05  # s
TARGET = 0.90  # SLO attainment goal (Section 1)
N = 2000
SEED = 0


def arrivals(rate, n=N, seed=SEED):
    """Poisson arrivals. The same unit-rate gaps are scaled by 1/rate, so a
    higher rate compresses the same trace and attainment can only fall."""
    gaps = np.random.RandomState(seed).exponential(1.0, n)
    return np.cumsum(gaps) / rate


def colocated(rate):
    """Return (ttft, tpot) arrays for one GPU with prefill-first batching."""
    arr = arrivals(rate)
    n = len(arr)
    first = np.zeros(n)
    finish = np.zeros(n)
    waiting, running = [], []  # running holds [request, tokens left]
    t, nxt, done = 0.0, 0, 0
    while done < n:
        while nxt < n and arr[nxt] <= t:
            waiting.append(nxt)
            nxt += 1
        if not waiting and not running:
            t = arr[nxt]
            continue
        if waiting and len(running) < B:
            r = waiting.pop(0)
            t += P
            first[r] = t
            running.append([r, L - 1])
        else:
            t += D
            for item in running:
                item[1] -= 1
            for r, left in running:
                if left == 0:
                    finish[r] = t
                    done += 1
            running = [item for item in running if item[1] > 0]
    return first - arr, (finish - first) / (L - 1)


def disaggregated(rate, n_p, n_d, transfer=0.0):
    """Return (ttft, tpot) arrays for n_p prefill GPUs and n_d decode GPUs."""
    arr = arrivals(rate)
    n = len(arr)
    free = [0.0] * n_p
    first = np.zeros(n)
    for i in range(n):
        g = i % n_p
        start = max(arr[i], free[g])
        free[g] = start + P
        first[i] = free[g]
    slots = [0.0] * (n_d * B)  # time at which each decode slot frees up
    heapq.heapify(slots)
    finish = np.zeros(n)
    for i in np.argsort(first + transfer, kind="stable"):
        ready = first[i] + transfer
        begin = max(ready, heapq.heappop(slots))
        finish[i] = begin + (L - 1) * D
        heapq.heappush(slots, finish[i])
    return first - arr, (finish - first) / (L - 1)


def attainment(ttft, tpot, ttft_slo=TTFT_SLO, tpot_slo=TPOT_SLO):
    """Fraction of requests that meet both SLOs."""
    return float(np.mean((ttft <= ttft_slo) & (tpot <= tpot_slo)))


def goodput(run, lo=0.05, hi=40.0, iters=24):
    """Largest rate whose attainment is at least TARGET, by bisection."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if attainment(*run(mid)) >= TARGET:
            lo = mid
        else:
            hi = mid
    return lo


if __name__ == "__main__":
    print("Attainment of one colocated GPU")
    for rate in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        ttft, tpot = colocated(rate)
        print(
            f"  rate {rate:.1f}: both {attainment(ttft, tpot):.3f}"
            f"  TTFT-only {attainment(ttft, tpot, tpot_slo=1e9):.3f}"
            f"  TPOT-only {attainment(ttft, tpot, ttft_slo=1e9):.3f}"
        )
    ttft4, tpot4 = np.array([0.1, 0.5, 0.2, 0.2]), np.array([0.03, 0.03, 0.06, 0.04])
    print(f"hand example: attainment of four requests = {attainment(ttft4, tpot4):.2f}")
    print("Attainment of 2 prefill + 1 decode GPUs")
    for rate in (3.0, 6.0, 12.0, 17.0, 18.0, 20.0):
        ttft, tpot = disaggregated(rate, 2, 1)
        print(
            f"  rate {rate:.1f}: both {attainment(ttft, tpot):.3f}"
            f"  TTFT-only {attainment(ttft, tpot, tpot_slo=1e9):.3f}"
            f"  TPOT-only {attainment(ttft, tpot, ttft_slo=1e9):.3f}"
        )
    print("First bisection steps for the colocated GPU")
    lo, hi = 0.05, 40.0
    for step in range(1, 7):
        mid = (lo + hi) / 2
        a = attainment(*colocated(mid))
        print(f"  step {step}: lo {lo:.3f} hi {hi:.3f} mid {mid:.3f} attainment {a:.3f}")
        lo, hi = (mid, hi) if a >= TARGET else (lo, mid)
    g_col = goodput(colocated)
    print(f"colocated goodput: {g_col:.2f} req/s on 1 GPU = {g_col:.2f} per GPU")
    for n_p, n_d in ((1, 1), (2, 1), (3, 1), (3, 2), (4, 1)):
        g = goodput(lambda r: disaggregated(r, n_p, n_d))
        print(
            f"disaggregated {n_p}P+{n_d}D: goodput {g:.2f} req/s on {n_p + n_d} GPUs"
            f" = {g / (n_p + n_d):.2f} per GPU ({g / (n_p + n_d) / g_col:.2f}x colocated)"
        )

    # KV transfer for a 512-token OPT-66B request (scripts/kv_transfer.py).
    request = 1207959552
    for name, bw in (("NVLink 600 GB/s", 600e9), ("Ethernet 25 Gbit/s", 25e9 / 8)):
        t = request / bw
        g = goodput(lambda r: disaggregated(r, 2, 1, transfer=t))
        print(f"2P+1D with {name} ({t * 1e3:.1f} ms per request): goodput {g:.2f} req/s")
