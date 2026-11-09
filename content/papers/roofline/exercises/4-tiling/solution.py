def tiled_matmul_intensity(n, tile, b):
    """FLOPs per DRAM byte for an n x n matmul with T x T output tiles."""
    flops = 2 * n ** 3
    traffic = b * (2 * n ** 3 / tile + n ** 2)
    return flops / traffic


def min_tile_for_ridge(n, ridge, b):
    """Smallest tile in 1..n with intensity >= ridge, or None."""
    if tiled_matmul_intensity(n, n, b) < ridge:
        return None
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if tiled_matmul_intensity(n, mid, b) >= ridge:
            hi = mid
        else:
            lo = mid + 1
    return lo
