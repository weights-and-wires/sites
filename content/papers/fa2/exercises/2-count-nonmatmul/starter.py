import math


def nonmatmul_ops(N, d, Br, Bc, version):
    """Count scalar non-matmul operations for one head, forward pass.

    version is "fa1" or "fa2". The counting model is in the exercise goal.
    """
    raise NotImplementedError


def matmul_flops(N, d):
    """FLOPs of the two matrix multiplications, S = QK^T and O = PV."""
    raise NotImplementedError


def nonmatmul_time_share(N, d, Br, Bc, version, ratio=16):
    """Fraction of time spent on non-matmul work if each such operation
    costs `ratio` times a matmul FLOP."""
    raise NotImplementedError
