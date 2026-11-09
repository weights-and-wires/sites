import numpy as np


def attention_backward(alpha, H, G):
    """Return (dE, dH).

    alpha: (Ty, Tx) softmax weights, rows sum to 1
    H: (Tx, 2n) annotations
    G: (Ty, 2n) gradient of the loss with respect to the contexts C = alpha @ H
    dE: (Ty, Tx) gradient with respect to the scores
    dH: (Tx, 2n) gradient with respect to H through C
    """
    raise NotImplementedError
