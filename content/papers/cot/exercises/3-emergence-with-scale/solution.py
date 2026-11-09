def gains(standard, cot):
    """Return the chain-of-thought gain, cot minus standard, for each model size."""
    if len(standard) != len(cot):
        raise ValueError("standard and cot must have the same length")
    return [c - s for s, c in zip(standard, cot)]


def emergence_index(gain_list, threshold=0.0):
    """Return the smallest i such that every gain from i onward is above threshold.

    Return None when the last gain is not above threshold, or the list is empty.
    """
    start = None
    for i, g in enumerate(gain_list):
        if g > threshold:
            if start is None:
                start = i
        else:
            start = None
    return start


def emergence_size(sizes, standard, cot, threshold=0.0):
    """Return the model size at the emergence index, or None."""
    i = emergence_index(gains(standard, cot), threshold)
    return None if i is None else sizes[i]
