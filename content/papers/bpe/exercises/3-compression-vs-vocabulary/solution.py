def encode(word, merges):
    """Finished for you: apply merges in learned order."""
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        out = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


def corpus_stats(word_freqs, merges):
    """Return (n_tokens, vocab_size, ratio) as described in the goal."""
    chars = {c for word in word_freqs for c in word}
    n_tokens = sum(freq * len(encode(word, merges)) for word, freq in word_freqs.items())
    initial = sum(freq * (len(word) + 1) for word, freq in word_freqs.items())
    return n_tokens, len(chars) + 1 + len(merges), initial / n_tokens
