import collections
import re


def get_stats(vocab):
    """Finished for you: pair counts weighted by word frequency."""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, vocab):
    """Finished for you: merge a pair of whole symbols."""
    out = {}
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word, freq in vocab.items():
        out[pattern.sub("".join(pair), word)] = freq
    return out


def learn_bpe(word_freqs, num_merges):
    """Return the list of merged pairs, in the order they were learned.

    word_freqs maps a plain word such as "low" to its count. Ties go to the
    lexicographically smallest pair. Stop early if no pair is left.
    """
    vocab = {" ".join(w) + " </w>": f for w, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    return merges


def encode(word, merges):
    """Segment one word. Return a list of symbols, ending with the </w> symbol.

    Apply the merges in the order they were learned.
    """
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
