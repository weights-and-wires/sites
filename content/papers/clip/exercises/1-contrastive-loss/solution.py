import numpy as np


def l2_normalize(x, axis=-1):
    """Scale each vector along axis to unit length."""
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def _log_softmax(x, axis):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def clip_loss(I_f, T_f, t):
    """Return the symmetric contrastive loss of Figure 3 as a float.

    I_f, T_f: (n, d_e) features of n aligned (image, text) pairs, not normalized.
    t: log of the logit scale, so the scale is exp(t).
    """
    I_e = l2_normalize(I_f, axis=1)
    T_e = l2_normalize(T_f, axis=1)
    logits = np.exp(t) * (I_e @ T_e.T)
    loss_i = -np.mean(np.diag(_log_softmax(logits, axis=1)))
    loss_t = -np.mean(np.diag(_log_softmax(logits, axis=0)))
    return float((loss_i + loss_t) / 2)
