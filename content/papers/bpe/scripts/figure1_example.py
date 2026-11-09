"""Trace Algorithm 1 on the toy dictionary from the paper (Figure 1).

The caption of Figure 1 says the dictionary is {low, lowest, newer, wider}. The
listing in Algorithm 1 uses a different dictionary (lower, newest, widest). This
script runs both, so you can see which one produces the merges printed in the
figure.

Run from the paper folder: python3 scripts/figure1_example.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import bpe_ref as B  # noqa: E402


def trace(vocab, n, rule):
    for i in range(n):
        pairs = B.get_stats(vocab)
        best = rule(pairs)
        print(f"  merge {i + 1}: {best[0]} {best[1]} -> {''.join(best)}  (count {pairs[best]})")
        vocab = B.merge_vocab(best, vocab)
    return vocab


paper_rule = lambda pairs: max(pairs, key=pairs.get)  # noqa: E731

print("Algorithm 1 listing dictionary, first 4 merges, paper's rule:")
listing = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}
trace(listing, 4, paper_rule)

print("Figure 1 caption dictionary, first 4 merges, paper's rule:")
fig = {"l o w </w>": 5, "l o w e s t </w>": 2, "n e w e r </w>": 6, "w i d e r </w>": 3}
print("  pair counts:", sorted(B.get_stats(fig).items(), key=lambda kv: (-kv[1], kv[0]))[:5])
trace(fig, 4, paper_rule)

print("Same dictionary, lexicographic tie-break, 10 merges:")
words = {"low": 5, "lowest": 2, "newer": 6, "wider": 3}
merges = B.learn_bpe(words, 10)
for i, m in enumerate(merges, 1):
    print(f"  merge {i}: {m[0]} {m[1]} -> {m[0] + m[1]}")
for w in ["lower", "lowest", "newer", "wider", "slow", "xyz"]:
    print(f"  encode({w!r}) =", B.encode(w, merges))

print("Corpus stats (tokens, vocabulary, compression) for the training words:")
for k in [0, 4, 10]:
    t, v, r = B.corpus_stats(words, merges[:k])
    print(f"  {k:2d} merges: {t} tokens, vocabulary {v}, compression {r:.3f}")

print("Naive str.replace bug, merging (b, c):")
v = {"a b c </w>": 1, "ab c </w>": 1}
print("  correct :", B.merge_vocab(("b", "c"), v))
print("  naive   :", {k.replace("b c", "bc"): f for k, f in v.items()})
