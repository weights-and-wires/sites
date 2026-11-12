import math


def recall_at_k(ranked, relevant, k):
    """Fraction of `relevant` found in ranked[:k]. 0.0 if relevant is empty."""
    raise NotImplementedError


def ndcg_at_k(ranked, relevant, k):
    """DCG@k / IDCG@k with binary gains and a log2(rank + 1) discount."""
    raise NotImplementedError
