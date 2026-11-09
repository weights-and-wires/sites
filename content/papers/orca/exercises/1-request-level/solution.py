def static_batching(arrivals, lengths, max_bs):
    """Request-level scheduling.

    Return (finish, useful): finish[i] is the time request i returns, and
    useful[t] counts the requests that still needed iteration t.
    """
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
