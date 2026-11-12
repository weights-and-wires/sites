def process_labels(step_ok):
    """Labels 1 (correct) or 0 (incorrect) up to and including the first incorrect step."""
    labels = []
    for ok in step_ok:
        labels.append(1 if ok else 0)
        if not ok:
            break
    return labels


def outcome_labels(n_steps, final_correct):
    """The outcome label of the whole solution, repeated once per step."""
    return [1 if final_correct else 0] * n_steps


def synthetic_step_ok(p_neg, threshold=0.2):
    """A step is correct unless PRM_large gives its negative label more than `threshold`."""
    return [p <= threshold for p in p_neg]
