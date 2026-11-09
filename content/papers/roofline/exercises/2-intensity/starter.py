def matmul_intensity(m, k, n, b):
    """Return (flops, bytes, intensity) for C = A @ B with compulsory traffic only."""
    raise NotImplementedError


def softmax_intensity(rows, cols, b):
    """Row-wise softmax: 5 FLOPs per element, one read and one write."""
    raise NotImplementedError


def layernorm_intensity(rows, cols, b):
    """LayerNorm: 8 FLOPs per element; reads x, gamma, beta; writes y."""
    raise NotImplementedError
