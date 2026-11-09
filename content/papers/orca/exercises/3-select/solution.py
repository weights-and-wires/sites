def select(pool, n_rsrv, max_bs, n_slots):
    """Algorithm 1, Select (lines 17 to 28).

    pool: list of dicts with "arrival", "state", and "max_tokens".
    Return (batch, n_rsrv): batch is a list of indices into pool.
    """
    candidates = sorted(
        (i for i, r in enumerate(pool) if r["state"] != "RUNNING"),
        key=lambda i: pool[i]["arrival"],
    )
    batch = []
    for i in candidates:
        if len(batch) == max_bs:
            break
        if pool[i]["state"] == "INITIATION":
            new_n_rsrv = n_rsrv + pool[i]["max_tokens"]
            if new_n_rsrv > n_slots:
                break
            n_rsrv = new_n_rsrv
        batch.append(i)
    return batch, n_rsrv
