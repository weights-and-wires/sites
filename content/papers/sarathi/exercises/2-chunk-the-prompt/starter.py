def chunk_sizes(prompt_len, budget):
    """Split prompt_len tokens into chunks of at most `budget` tokens."""
    raise NotImplementedError


def kv_tokens_reread(chunks):
    """Total earlier-chunk KV tokens read across all chunks."""
    raise NotImplementedError
