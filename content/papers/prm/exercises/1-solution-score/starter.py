import numpy as np


def step_scores(p_pos, p_neu, neutral="positive"):
    """Per-step probability that the step counts as correct.

    p_pos and p_neu hold, for each step, the PRM's probability of the positive
    and the neutral label. Return a float array with one score per step.
    """
    raise NotImplementedError


def solution_score(p_pos, p_neu, neutral="positive", reduction="product"):
    """Reduce the step scores of one solution to a single float."""
    raise NotImplementedError
