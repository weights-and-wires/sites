import numpy as np


def resize_pos_embed(E_pos, g_new):
    """Keep row 0 (class token); resize the g x g grid in rows 1: to g_new x g_new.

    E_pos: (1 + g*g, D) in raster order. Return (1 + g_new*g_new, D).
    Bilinear interpolation with aligned corners: new index i reads source
    coordinate i * (g - 1) / (g_new - 1) along each axis.
    """
    raise NotImplementedError
