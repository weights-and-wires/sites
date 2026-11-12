import numpy as np


def info_nce(q, d_pos, d_hard=None, tau=0.02):
    """Mean InfoNCE loss over the batch with in-batch and hard negatives.

    q, d_pos: (B, d). d_hard: None or (B, d). Returns a float.
    """
    raise NotImplementedError
