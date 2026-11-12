from collections import Counter


def best_of_n(scores):
    """Return the index of the highest score. The first index wins a tie."""
    raise NotImplementedError


def majority_vote(answers):
    """Return the most common answer. Among tied answers, the one that appears first wins."""
    raise NotImplementedError


def weighted_vote(answers, scores):
    """Return the answer with the largest sum of scores.

    Among answers with equal sums, the one that appears first wins.
    """
    raise NotImplementedError
