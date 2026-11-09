import math


def blocks_needed(n_tokens, block_size):
    """Number of blocks that hold n_tokens tokens."""
    return math.ceil(n_tokens / block_size)


def paged_cost(n_tokens, block_size):
    """Slots a request occupies when its KV cache is stored in blocks."""
    return blocks_needed(n_tokens, block_size) * block_size


def contiguous_waste(lengths, max_len):
    """Total unused slots when every request reserves max_len slots."""
    return sum(max_len - n for n in lengths)


def paged_waste(lengths, block_size):
    """Total unused slots when each request holds only the blocks it needs."""
    return sum(paged_cost(n, block_size) - n for n in lengths)


def admit_count(lengths, capacity, cost):
    """How many requests, taken in order, fit in capacity slots. Stop at the first that doesn't."""
    used = 0
    count = 0
    for n in lengths:
        c = cost(n)
        if used + c > capacity:
            break
        used += c
        count += 1
    return count
