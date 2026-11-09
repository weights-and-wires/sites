def continuous_batching(arrivals, lengths, max_bs):
    """Iteration-level scheduling.

    Return (finish, useful): finish[i] is the time request i returns, and
    useful[t] counts the requests that ran in iteration t.
    """
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
