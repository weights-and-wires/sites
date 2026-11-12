def stall_free_batch(n_decodes, waiting, budget):
    """Return the prefill chunk size for each entry of `waiting`."""
    left = max(0, budget - n_decodes)
    chunks = []
    for remaining in waiting:
        c = min(remaining, left)
        chunks.append(c)
        left -= c
    return chunks
