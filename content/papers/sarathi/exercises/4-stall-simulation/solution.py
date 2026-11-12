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
