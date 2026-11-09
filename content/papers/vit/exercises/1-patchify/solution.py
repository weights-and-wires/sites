import numpy as np


def patchify(img, P):
    """Return an (N, P*P*C) array of flattened patches in raster order."""
    H, W, C = img.shape
    grid = img.reshape(H // P, P, W // P, P, C).transpose(0, 2, 1, 3, 4)
    return grid.reshape((H // P) * (W // P), P * P * C)


def unpatchify(patches, H, W, C, P):
    """Invert patchify: return the (H, W, C) image."""
    grid = patches.reshape(H // P, W // P, P, P, C).transpose(0, 2, 1, 3, 4)
    return grid.reshape(H, W, C)
