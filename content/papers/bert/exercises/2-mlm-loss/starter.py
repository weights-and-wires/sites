import numpy as np


def masked_lm_loss(logits, labels, ignore_index=-100):
    """Mean cross-entropy over positions where labels != ignore_index."""
    raise NotImplementedError
