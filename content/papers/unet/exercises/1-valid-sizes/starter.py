def unet_output_size(n, levels=4):
    """Return the output side length for an n x n input, or None if n is invalid.

    Each contracting step: two 3x3 valid convolutions, then a 2x2 pool (needs an even size).
    The bottom: two 3x3 valid convolutions.
    Each expanding step: a 2x2 up-convolution that doubles the size, then two 3x3 valid convolutions.
    """
    raise NotImplementedError
