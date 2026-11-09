import collections
import re


def get_stats(vocab):
    """Count adjacent symbol pairs, weighted by word frequency.

    vocab maps "s y m b o l s </w>" to its frequency.
    Return a dict that maps (left, right) to a count.
    """
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, vocab):
    """Replace each whole-symbol occurrence of pair with the joined symbol.

    Return a new dict; keep each word's frequency.
    """
    out = {}
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word, freq in vocab.items():
        out[pattern.sub("".join(pair), word)] = freq
    return out
