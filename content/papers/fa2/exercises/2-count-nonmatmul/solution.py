import math


def nonmatmul_ops(N, d, Br, Bc, version):
    """Count scalar non-matmul operations for one head, forward pass.

    version is "fa1" or "fa2". The counting model is in the exercise goal.
    """
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)
    shared = 4 * Br * Bc  # rowmax, subtract, exp, rowsum on the Br x Bc tile
    if version == "fa1":
        per_tile = shared + 9 * Br + 3 * Br * d
        return Tr * Tc * per_tile
    if version == "fa2":
        per_tile = shared + 5 * Br + Br * d
        final = Br * d + 2 * Br  # one divide per output element, plus log and add
        return Tr * (Tc * per_tile + final)
    raise ValueError(f"unknown version {version!r}")


def matmul_flops(N, d):
    """FLOPs of the two matrix multiplications, S = QK^T and O = PV."""
    return 4 * N * N * d


def nonmatmul_time_share(N, d, Br, Bc, version, ratio=16):
    """Fraction of time spent on non-matmul work if each such operation
    costs `ratio` times a matmul FLOP."""
    nm = ratio * nonmatmul_ops(N, d, Br, Bc, version)
    return nm / (nm + matmul_flops(N, d))
