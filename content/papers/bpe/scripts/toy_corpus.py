"""A small synthetic word-frequency table with English-like morphology.

Frequencies follow a Zipf-like rule, freq = 20000 // rank, over a seeded shuffle
of every prefix + stem + suffix combination. The numbers are not real-language
statistics; they only give the compression example a corpus with shared
subwords and a long tail of rare words.
"""

import random

STEMS = """walk talk jump play work look call move help learn build train read write open close
start stop test check light power water sun system form act port print link lock mark name
point place plan run set sort store track use value view wait watch wind wire word yield
zone break catch change charge clean clear count cover cut draw drive drop fill find fix
flow fold follow frame gain grow guard hand hold join keep kick lead lift list load""".split()
SUFFIXES = ["", "s", "ed", "ing", "er", "ers", "ly", "ness", "able", "ment"]
PREFIXES = ["", "re", "un"]


def build():
    words = [p + s + x for p in PREFIXES for s in STEMS for x in SUFFIXES]
    words = sorted(set(words))
    random.Random(0).shuffle(words)
    return {w: max(1, 20000 // (rank + 1)) for rank, w in enumerate(words)}


if __name__ == "__main__":
    t = build()
    print(len(t), "word types,", sum(t.values()), "tokens")
