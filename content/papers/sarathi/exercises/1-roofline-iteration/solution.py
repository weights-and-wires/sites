import math


def arithmetic_intensity(n_tokens, bytes_per_param=2):
    """FLOPs per byte of weight traffic for a pass over n_tokens tokens."""
    return 2.0 * n_tokens / bytes_per_param


def iteration_time(n_tokens, n_params, peak_flops, bandwidth, bytes_per_param=2):
    """Return (seconds, bound) with bound in {"memory", "compute"}."""
    t_math = 2.0 * n_params * n_tokens / peak_flops
    t_mem = n_params * bytes_per_param / bandwidth
    return max(t_math, t_mem), ("memory" if t_mem > t_math else "compute")


def token_budget(slo, n_params, peak_flops, bandwidth, bytes_per_param=2):
    """Largest integer n with iteration_time(n) <= slo, or 0."""
    if slo < n_params * bytes_per_param / bandwidth:
        return 0
    return int(math.floor(slo * peak_flops / (2.0 * n_params) + 1e-9))
