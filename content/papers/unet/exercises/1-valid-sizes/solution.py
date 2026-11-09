def unet_output_size(n, levels=4):
    """Return the output side length for an n x n input, or None if n is invalid.

    Each contracting step: two 3x3 valid convolutions, then a 2x2 pool (needs an even size).
    The bottom: two 3x3 valid convolutions.
    Each expanding step: a 2x2 up-convolution that doubles the size, then two 3x3 valid convolutions.
    """
    s = n
    for _ in range(levels):
        s -= 4
        if s <= 0 or s % 2:
            return None
        s //= 2
    s -= 4
    if s <= 0:
        return None
    for _ in range(levels):
        s *= 2
        s -= 4
        if s <= 0:
            return None
    return s
