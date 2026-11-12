def soft_overlong_penalty(length, l_max=20480, l_cache=4096):
    """R_length(y) from Equation 13 for a response of `length` tokens."""
    if length <= l_max - l_cache:
        return 0.0
    if length <= l_max:
        return ((l_max - l_cache) - length) / l_cache
    return -1.0


def total_reward(correct, length, l_max=20480, l_cache=4096):
    """Rule-based reward (+1 or -1, Equation 7) plus the length penalty."""
    base = 1.0 if correct else -1.0
    return base + soft_overlong_penalty(length, l_max, l_cache)
