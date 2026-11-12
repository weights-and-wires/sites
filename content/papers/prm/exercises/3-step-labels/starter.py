def process_labels(step_ok):
    """Return a list of 1 (correct) and 0 (incorrect) up to and including the first incorrect step.

    `step_ok` is a list of booleans, one per step. A solution with no incorrect step
    keeps all its labels.
    """
    raise NotImplementedError


def outcome_labels(n_steps, final_correct):
    """Return the outcome label (1 or 0) of the whole solution, repeated `n_steps` times."""
    raise NotImplementedError


def synthetic_step_ok(p_neg, threshold=0.2):
    """Return a list of booleans: is each step correct, given PRM_large's negative probabilities?"""
    raise NotImplementedError
