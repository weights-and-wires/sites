def gains(standard, cot):
    """Return the chain-of-thought gain, cot minus standard, for each model size.

    Raise ValueError when the lists differ in length.
    """
    raise NotImplementedError


def emergence_index(gain_list, threshold=0.0):
    """Return the smallest i such that every gain from i onward is above threshold.

    Return None when the last gain is not above threshold, or the list is empty.
    """
    raise NotImplementedError


def emergence_size(sizes, standard, cot, threshold=0.0):
    """Return the model size at the emergence index, or None."""
    raise NotImplementedError
