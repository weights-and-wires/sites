"""Prefill-prioritizing vs stall-free scheduling on a toy trace (Sections 3.2, 4.2).

Run from the paper folder: python3 scripts/stall_free_sim.py

The simulator runs D requests that are already decoding and one long prompt
that arrives. Each iteration emits one token for every request that is in its
batch. The iteration time comes from a one-line roofline: it costs `base`
until the batch passes `ridge` tokens, then grows linearly.
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import roofline_cost as rc


def simulate(policy, d, steps, prompt, arrive, budget, iter_time):
    """Return (max time between tokens of the d decodes, TTFT of the prompt)."""
    left = [steps] * d
    times = [[] for _ in range(d)]
    t, it, remaining, ttft, t_arrive = 0.0, 0, prompt, None, None
    while any(left) or remaining > 0:
        waiting = it >= arrive and remaining > 0
        if waiting and t_arrive is None:
            t_arrive = t
        active = [i for i in range(d) if left[i] > 0]
        if policy == "prefill_first":
            if waiting:
                n, include, chunk = remaining, False, remaining
            else:
                n, include, chunk = len(active), True, 0
        else:
            chunk = min(remaining, max(0, budget - len(active))) if waiting else 0
            n, include = len(active) + chunk, True
        t += iter_time(n)
        if include:
            for i in active:
                left[i] -= 1
                times[i].append(t)
        remaining -= chunk
        if waiting and remaining == 0:
            ttft = t - t_arrive
        it += 1
    gaps = [b - a for ts in times for a, b in zip(ts, ts[1:])]
    return max(gaps), ttft


def toy(n):
    return 10.0 * max(1.0, n / 100)  # ms: ridge 100 tokens, floor 10 ms


def real(n):
    return rc.iteration_time(n)[0] * 1e3  # ms


if __name__ == "__main__":
    print("toy model: D=4, steps=12, prompt=1000, arrives at iteration 3")
    for name, pol, b in (("prefill_first", "prefill_first", 0), ("stall_free b=200", "stall_free", 200),
                         ("stall_free b=100", "stall_free", 100), ("stall_free b=1000", "stall_free", 1000)):
        tbt, ttft = simulate(pol, 4, 12, 1000, 3, b, toy)
        print(f"{name:18s} max TBT = {tbt:6.1f} ms, TTFT = {ttft:6.1f} ms")
    print()
    print("roofline model: D=32, steps=40, prompt=4096, arrives at iteration 5")
    print(f"decode-only iteration: {real(32):.2f} ms")
    for name, pol, b in (("prefill_first", "prefill_first", 0), ("stall_free b=153", "stall_free", 153),
                         ("stall_free b=512", "stall_free", 512), ("stall_free b=2048", "stall_free", 2048)):
        tbt, ttft = simulate(pol, 32, 40, 4096, 5, b, real)
        print(f"{name:18s} max TBT = {tbt:6.2f} ms, TTFT = {ttft:7.2f} ms")
