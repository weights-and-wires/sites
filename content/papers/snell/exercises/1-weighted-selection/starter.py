import numpy as np


def step_score(step_scores, how="last"):
    """Reduce per-step PRM scores to one float: "last", "min", or "prod"."""
    raise NotImplementedError


def select_answer(answers, scores, method):
    """Pick a final answer: "best_of_n", "majority", or "weighted".

    Among tied choices, the answer that appears first in `answers` wins.
    """
    raise NotImplementedError
