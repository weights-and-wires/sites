import math


def blocks_needed(n_tokens, block_size):
    """Number of blocks that hold n_tokens tokens."""
    raise NotImplementedError


def paged_cost(n_tokens, block_size):
    """Slots a request occupies when its KV cache is stored in blocks."""
    raise NotImplementedError


def contiguous_waste(lengths, max_len):
    """Total unused slots when every request reserves max_len slots."""
    raise NotImplementedError


def paged_waste(lengths, block_size):
    """Total unused slots when each request holds only the blocks it needs."""
    raise NotImplementedError


def admit_count(lengths, capacity, cost):
    """How many requests, taken in order, fit in capacity slots. Stop at the first that doesn't."""
    raise NotImplementedError
