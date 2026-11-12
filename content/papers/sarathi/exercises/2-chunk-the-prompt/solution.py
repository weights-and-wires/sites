def chunk_sizes(prompt_len, budget):
    """Split prompt_len tokens into chunks of at most `budget` tokens."""
    sizes = []
    left = prompt_len
    while left > 0:
        c = min(budget, left)
        sizes.append(c)
        left -= c
    return sizes


def kv_tokens_reread(chunks):
    """Total earlier-chunk KV tokens read across all chunks."""
    return sum(sum(chunks[:i]) for i in range(len(chunks)))
