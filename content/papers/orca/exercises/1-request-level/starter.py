def static_batching(arrivals, lengths, max_bs):
    """Request-level scheduling.

    Return (finish, useful): finish[i] is the time request i returns, and
    useful[t] counts the requests that still needed iteration t.
    """
    raise NotImplementedError
