import math


def block_sizes(d, M):
    """Algorithm 1, line 1. Returns (Bc, Br)."""
    Bc = math.ceil(M / (4 * d))
    return Bc, min(Bc, d)


def standard_hbm_accesses(N, d):
    """Algorithm 0: count element reads and writes to HBM."""
    line1 = 2 * N * d + N * N
    line2 = N * N + N * N
    line3 = N * N + N * d + N * d
    return line1 + line2 + line3


def flash_hbm_accesses(N, d, M):
    """Algorithm 1: count element reads and writes to HBM."""
    Bc, _ = block_sizes(d, M)
    Tc = math.ceil(N / Bc)
    load_kv = 2 * N * d
    loads = N * d + N * d + N + N
    writes = N * d + N + N
    return load_kv + Tc * (loads + writes)
