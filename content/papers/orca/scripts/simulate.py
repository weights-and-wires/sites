"""Reference simulators for request-level and iteration-level scheduling.

Time is counted in model iterations. A request with length g needs g iterations:
the first is the initiation phase (Section 2) and the other g - 1 are increment
phases. Every iteration takes one time unit, whatever the batch holds. That is a
simplification: a real iteration with more tokens costs more.
"""

import numpy as np


def static_batching(arrivals, lengths, max_bs):
    """Request-level scheduling. Returns (finish_times, useful_per_iteration)."""
    n = len(arrivals)
    order = sorted(range(n), key=lambda i: (arrivals[i], i))
    finish = [0] * n
    useful = []
    t = 0
    nxt = 0
    while nxt < n:
        if arrivals[order[nxt]] > t:
            useful += [0] * (arrivals[order[nxt]] - t)
            t = arrivals[order[nxt]]
        batch = []
        while nxt < n and len(batch) < max_bs and arrivals[order[nxt]] <= t:
            batch.append(order[nxt])
            nxt += 1
        steps = max(lengths[i] for i in batch)
        for s in range(steps):
            useful.append(sum(1 for i in batch if lengths[i] > s))
        t += steps
        for i in batch:
            finish[i] = t
    return finish, useful


def continuous_batching(arrivals, lengths, max_bs):
    """Iteration-level scheduling. Returns (finish_times, useful_per_iteration)."""
    n = len(arrivals)
    order = sorted(range(n), key=lambda i: (arrivals[i], i))
    remaining = list(lengths)
    finish = [0] * n
    useful = []
    t = 0
    done = 0
    while done < n:
        ready = [i for i in order if arrivals[i] <= t and remaining[i] > 0][:max_bs]
        for i in ready:
            remaining[i] -= 1
            if remaining[i] == 0:
                finish[i] = t + 1
                done += 1
        useful.append(len(ready))
        t += 1
    return finish, useful


def utilization(useful, max_bs):
    """Fraction of slot-iterations that did useful work."""
    return sum(useful) / (max_bs * len(useful))


def make_workload(n, rate, seed):
    """Lengths ~ U(1, 128) as in Section 6.2; Poisson arrivals at `rate` per iteration.
    rate=None puts every request in the queue at time 0."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 129, size=n).tolist()
    if rate is None:
        arrivals = [0] * n
    else:
        arrivals = np.floor(np.cumsum(rng.exponential(1 / rate, size=n))).astype(int).tolist()
    return arrivals, lengths
