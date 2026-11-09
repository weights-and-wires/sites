def tiled_matmul_intensity(n, tile, b):
    """FLOPs per DRAM byte for an n x n matmul with T x T output tiles."""
    raise NotImplementedError


def min_tile_for_ridge(n, ridge, b):
    """Smallest tile in 1..n with intensity >= ridge, or None."""
    raise NotImplementedError
