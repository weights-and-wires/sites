#!/usr/bin/env python3
"""Recall@k and nDCG@k on a four-document ranking.

Run from the paper folder: python3 scripts/metrics_example.py
"""

import math

ranked = ["d3", "d1", "d7", "d2"]
relevant = {"d1", "d2"}


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant)


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked, relevant, k):
    gains = [1.0 if d in relevant else 0.0 for d in ranked[:k]]
    ideal = [1.0] * min(k, len(relevant))
    return dcg(gains) / dcg(ideal)


for k in (1, 2, 3, 4):
    print(f"k={k}: recall={recall_at_k(ranked, relevant, k):.4f}, "
          f"ndcg={ndcg_at_k(ranked, relevant, k):.4f}")
print(f"DCG@3 = {dcg([0, 1, 0]):.4f}, IDCG@3 = {dcg([1, 1]):.4f}")
