def latent_shape(H, W, f, c):
    """Return (h, w, c) for an H x W image downsampled by the factor f.

    Raise ValueError when f does not divide both H and W.
    """
    raise NotImplementedError


def compression_ratio(H, W, f, c):
    """Return (values in an H x W x 3 image) / (values in the latent)."""
    raise NotImplementedError


def position_ratio(H, W, f):
    """Return (pixel positions) / (latent positions)."""
    raise NotImplementedError


def attention_ratio(H, W, f):
    """Return (entries of the pixel-space N x N attention matrix) / (entries of the latent one)."""
    raise NotImplementedError
