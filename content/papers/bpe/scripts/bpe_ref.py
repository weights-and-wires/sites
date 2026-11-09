"""Reference BPE code, following Algorithm 1 of Sennrich et al. (2016).

The only differences from the paper's listing: learn_bpe returns the merges,
ties between equally frequent pairs go to the lexicographically smallest pair
(the paper's `max` over a dict takes the pair it saw first), and encode applies
the learned merges to a new word.
"""

import collections
import re


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in v_in:
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def to_vocab(word_freqs):
    return {" ".join(w) + " </w>": f for w, f in word_freqs.items()}


def best_pair(pairs):
    return min(pairs, key=lambda p: (-pairs[p], p))


def learn_bpe(word_freqs, num_merges):
    vocab = to_vocab(word_freqs)
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = best_pair(pairs)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    return merges


def encode(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        out, i = [], 0
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
    chars = {c for w in word_freqs for c in w}
    tokens = sum(f * len(encode(w, merges)) for w, f in word_freqs.items())
    initial = sum(f * (len(w) + 1) for w, f in word_freqs.items())
    return tokens, len(chars) + 1 + len(merges), initial / tokens
