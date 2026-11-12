from collections import Counter


def best_of_n(scores):
    """Index of the highest score. The first one wins a tie."""
    best = 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best


def majority_vote(answers):
    """The most common answer. Among tied answers, the one that appears first wins."""
    return Counter(answers).most_common(1)[0][0]


def weighted_vote(answers, scores):
    """The answer with the largest sum of scores. Among tied answers, the first one wins."""
    totals = {}
    for a, s in zip(answers, scores):
        totals[a] = totals.get(a, 0.0) + s
    best, best_total = None, None
    for a, t in totals.items():
        if best_total is None or t > best_total:
            best, best_total = a, t
    return best
