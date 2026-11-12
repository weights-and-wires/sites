def kept_indices(n, sinks, window):
    """Sorted token indices in the cache after n tokens: the first `sinks` and the last `window`."""
    raise NotImplementedError


def evicted_indices(n, sinks, window):
    """Sorted token indices that the cache has dropped after n tokens."""
    raise NotImplementedError


def cache_step(cache, new_index, sinks, window):
    """Return a new cache list after adding new_index. Don't modify `cache` in place."""
    raise NotImplementedError
