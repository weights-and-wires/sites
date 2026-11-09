import numpy as np


def loss_and_grads(Xi, Xt, Wi, Wt, t):
    """Return (loss, dWi, dWt, dt) for the symmetric contrastive loss.

    Xi: (n, d_i), Xt: (n, d_t), Wi: (d_i, d_e), Wt: (d_t, d_e), t: float.
    I_f = Xi @ Wi and T_f = Xt @ Wt are L2-normalized to I_e and T_e,
    logits = exp(t) * I_e @ T_e.T, and the loss is
    (cross entropy over rows + cross entropy over columns) / 2 with the
    diagonal as the target, each averaged over the n pairs.
    """
    raise NotImplementedError
