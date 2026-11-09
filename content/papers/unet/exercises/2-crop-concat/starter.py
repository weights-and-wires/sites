import numpy as np


def crop_and_concat(skip, up):
    """Center-crop skip to up's height and width, then concatenate on the channel axis.

    skip: array (C1, H, W). up: array (C2, h, w) with h <= H and w <= W.
    Returns an array (C1 + C2, h, w) with the cropped skip channels first.
    """
    raise NotImplementedError
