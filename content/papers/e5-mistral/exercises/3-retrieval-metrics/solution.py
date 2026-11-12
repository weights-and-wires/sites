import math


def recall_at_k(ranked, relevant, k):
    """Fraction of `relevant` found in ranked[:k]. 0.0 if relevant is empty."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked, relevant, k):
    """DCG@k / IDCG@k with binary gains and a log2(rank + 1) discount."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(j + 2) for j, d in enumerate(ranked[:k]) if d in relevant)
    idcg = sum(1.0 / math.log2(j + 2) for j in range(min(k, len(relevant))))
    return dcg / idcg
