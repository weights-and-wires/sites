"""Sequence length against vocabulary size for BPE on a toy corpus.

Run from the paper folder: python3 scripts/compression.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import bpe_ref as B  # noqa: E402
import toy_corpus  # noqa: E402

corpus = toy_corpus.build()
n_words = sum(corpus.values())
n_types = len(corpus)
chars = len({c for w in corpus for c in w})
print(f"corpus: {n_types} word types, {n_words} word tokens, {chars} distinct characters")
print(f"word level: {n_words} tokens, vocabulary {n_types}")

t0 = time.time()
all_merges = B.learn_bpe(corpus, 2000)
print(f"learned {len(all_merges)} merges (limit 2000) in {time.time() - t0:.1f} s")
print("first 8 merges:", all_merges[:8])
print()
print("merges  vocab  tokens  tokens/word  compression  (characters+</w> per token)")
for k in [0, 100, 500, 2000]:
    tokens, vocab, ratio = B.corpus_stats(corpus, all_merges[:k])
    print(f"{k:6d} {vocab:6d} {tokens:7d} {tokens / n_words:11.3f} {ratio:12.3f}")

# A word that is not in the corpus: the learned merges still segment it.
unseen = "rewatchingly"
print()
print(f"unseen word {unseen!r} in corpus: {unseen in corpus}")
for k in [0, 100, 500, 2000]:
    print(f"  {k:4d} merges ->", B.encode(unseen, all_merges[:k]))

# Table 1 of the paper, German side (tokens in millions).
print()
print("Paper Table 1 (German training data): tokens in millions / types")
rows = [("none", 100, 1_750_000), ("characters", 550, 3000), ("BPE", 112, 63_000), ("BPE (joint)", 111, 82_000)]
for name, tok, types in rows:
    print(f"  {name:12s} {tok:4d} m  {types:9d} types  length vs. words: {tok / 100:.2f}x")
