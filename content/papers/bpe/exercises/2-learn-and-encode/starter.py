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
    raise NotImplementedError


def encode(word, merges):
    """Segment one word. Return a list of symbols, ending with the </w> symbol.

    Apply the merges in the order they were learned.
    """
    raise NotImplementedError
