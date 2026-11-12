import numpy as np


def step_score(step_scores, how="last"):
    """Reduce per-step PRM scores to one float: "last", "min", or "prod"."""
    if how == "last":
        return float(step_scores[-1])
    elif how == "min":
        return float(np.min(step_scores))
    elif how == "prod":
        return float(np.prod(step_scores))
    else:
        raise ValueError(f"unknown aggregation: {how!r}")


def select_answer(answers, scores, method):
    """Pick a final answer: "best_of_n", "majority", or "weighted".

    Among tied choices, the answer that appears first in `answers` wins.
    """
    if method == "best_of_n":
        return answers[int(np.argmax(scores))]
    if method == "majority":
        totals = {}
        for a in answers:
            totals[a] = totals.get(a, 0) + 1
    elif method == "weighted":
        totals = {}
        for a, s in zip(answers, scores):
            totals[a] = totals.get(a, 0.0) + float(s)
    else:
        raise ValueError(f"unknown method: {method!r}")
    best = None
    for a, t in totals.items():
        if best is None or t > totals[best]:
            best = a
    return best
