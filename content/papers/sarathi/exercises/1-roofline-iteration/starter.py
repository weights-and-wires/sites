import math


def arithmetic_intensity(n_tokens, bytes_per_param=2):
    """FLOPs per byte of weight traffic for a pass over n_tokens tokens."""
    raise NotImplementedError


def iteration_time(n_tokens, n_params, peak_flops, bandwidth, bytes_per_param=2):
    """Return (seconds, bound) with bound in {"memory", "compute"}."""
    raise NotImplementedError


def token_budget(slo, n_params, peak_flops, bandwidth, bytes_per_param=2):
    """Largest integer n with iteration_time(n) <= slo, or 0."""
    raise NotImplementedError
