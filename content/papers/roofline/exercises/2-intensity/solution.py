def matmul_intensity(m, k, n, b):
    """Return (flops, bytes, intensity) for C = A @ B with compulsory traffic only."""
    flops = 2 * m * k * n
    traffic = b * (m * k + k * n + m * n)
    return flops, traffic, flops / traffic


def softmax_intensity(rows, cols, b):
    """Row-wise softmax: 5 FLOPs per element, one read and one write."""
    flops = 5 * rows * cols
    traffic = b * 2 * rows * cols
    return flops, traffic, flops / traffic


def layernorm_intensity(rows, cols, b):
    """LayerNorm: 8 FLOPs per element; reads x, gamma, beta; writes y."""
    flops = 8 * rows * cols
    traffic = b * (2 * rows * cols + 2 * cols)
    return flops, traffic, flops / traffic
