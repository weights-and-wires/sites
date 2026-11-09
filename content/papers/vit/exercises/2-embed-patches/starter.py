import numpy as np


def patchify(img, P):
    """Finished for you: (H, W, C) -> (N, P*P*C) in raster order."""
    H, W, C = img.shape
    grid = img.reshape(H // P, P, W // P, P, C).transpose(0, 2, 1, 3, 4)
    return grid.reshape((H // P) * (W // P), P * P * C)


def embed_patches(img, P, E, x_class, E_pos):
    """Equation 1: return z_0 with shape (N + 1, D).

    E: (P*P*C, D), x_class: (D,), E_pos: (N + 1, D). No bias.
    """
    raise NotImplementedError
