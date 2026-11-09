import numpy as np


def patchify(z, p):
    """(I, I, C) array -> (T, p*p*C) array, T = (I // p) ** 2, patches in raster order."""
    I, _, C = z.shape
    g = I // p
    return z.reshape(g, p, g, p, C).transpose(0, 2, 1, 3, 4).reshape(g * g, p * p * C)


def unpatchify(tokens, p, C):
    """(T, p*p*C) array -> (I, I, C) array. The inverse of patchify."""
    g = int(round(tokens.shape[0] ** 0.5))
    return tokens.reshape(g, g, p, p, C).transpose(0, 2, 1, 3, 4).reshape(g * p, g * p, C)
