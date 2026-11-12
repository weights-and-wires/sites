def soft_overlong_penalty(length, l_max=20480, l_cache=4096):
    """R_length(y) from Equation 13 for a response of `length` tokens."""
    raise NotImplementedError


def total_reward(correct, length, l_max=20480, l_cache=4096):
    """Rule-based reward (+1 or -1, Equation 7) plus the length penalty."""
    raise NotImplementedError
