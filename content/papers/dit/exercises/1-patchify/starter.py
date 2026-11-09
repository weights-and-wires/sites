import numpy as np


def patchify(z, p):
    """(I, I, C) array -> (T, p*p*C) array, T = (I // p) ** 2, patches in raster order."""
    raise NotImplementedError


def unpatchify(tokens, p, C):
    """(T, p*p*C) array -> (I, I, C) array. The inverse of patchify."""
    raise NotImplementedError
