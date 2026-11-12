def kept_indices(n, sinks, window):
    """Sorted token indices in the cache after n tokens: the first `sinks` and the last `window`."""
    if n <= sinks + window:
        return list(range(n))
    return list(range(sinks)) + list(range(n - window, n))


def evicted_indices(n, sinks, window):
    """Sorted token indices that the cache has dropped after n tokens."""
    kept = set(kept_indices(n, sinks, window))
    return [i for i in range(n) if i not in kept]


def cache_step(cache, new_index, sinks, window):
    """Return a new cache list after adding new_index. Don't modify `cache` in place."""
    out = list(cache) + [new_index]
    if len(out) > sinks + window:
        del out[sinks]
    return out
