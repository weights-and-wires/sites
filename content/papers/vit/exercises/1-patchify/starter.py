import numpy as np


def patchify(img, P):
    """Return an (N, P*P*C) array of flattened patches in raster order.

    img has shape (H, W, C); H and W are multiples of P.
    """
    raise NotImplementedError


def unpatchify(patches, H, W, C, P):
    """Invert patchify: return the (H, W, C) image."""
    raise NotImplementedError
