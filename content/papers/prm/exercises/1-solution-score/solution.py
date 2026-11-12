import numpy as np


def step_scores(p_pos, p_neu, neutral="positive"):
    """Per-step probability that the step counts as correct."""
    p_pos = np.asarray(p_pos, dtype=float)
    p_neu = np.asarray(p_neu, dtype=float)
    if neutral == "positive":
        return p_pos + p_neu
    if neutral == "negative":
        return p_pos.copy()
    raise ValueError(f"neutral must be 'positive' or 'negative', got {neutral!r}")


def solution_score(p_pos, p_neu, neutral="positive", reduction="product"):
    """Reduce the step scores of one solution to a single score."""
    scores = step_scores(p_pos, p_neu, neutral)
    if scores.size == 0:
        raise ValueError("a solution needs at least one step")
    if reduction == "product":
        return float(np.prod(scores))
    if reduction == "min":
        return float(np.min(scores))
    raise ValueError(f"reduction must be 'product' or 'min', got {reduction!r}")
