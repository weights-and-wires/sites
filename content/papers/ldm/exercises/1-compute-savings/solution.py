def latent_shape(H, W, f, c):
    """Return (h, w, c) for an H x W image downsampled by the factor f.

    Raise ValueError when f does not divide both H and W.
    """
    if H % f != 0 or W % f != 0:
        raise ValueError(f"f={f} does not divide {H}x{W}")
    return H // f, W // f, c


def compression_ratio(H, W, f, c):
    """Return (values in an H x W x 3 image) / (values in the latent)."""
    h, w, c = latent_shape(H, W, f, c)
    return (H * W * 3) / (h * w * c)


def position_ratio(H, W, f):
    """Return (pixel positions) / (latent positions)."""
    h, w, _ = latent_shape(H, W, f, 1)
    return (H * W) / (h * w)


def attention_ratio(H, W, f):
    """Return (entries of the pixel-space N x N attention matrix) / (entries of the latent one)."""
    h, w, _ = latent_shape(H, W, f, 1)
    return (H * W) ** 2 / (h * w) ** 2
