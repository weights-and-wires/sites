def continuous_batching(arrivals, lengths, max_bs):
    """Iteration-level scheduling.

    Return (finish, useful): finish[i] is the time request i returns, and
    useful[t] counts the requests that ran in iteration t.
    """
    raise NotImplementedError
