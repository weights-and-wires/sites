import numpy as np


def rope(x, pos, base=10000.0):
    """Rotate pair (x[2j], x[2j+1]) by pos * base ** (-2j / d). x: (..., T, d)."""
    x = np.asarray(x, dtype=float)
    d = x.shape[-1]
    theta = base ** (-2.0 * np.arange(d // 2) / d)
    angle = np.asarray(pos, dtype=float)[:, None] * theta[None, :]
    cos, sin = np.cos(angle), np.sin(angle)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = a * sin + b * cos
    return out


def decoupled_scores(qC, kC, qR, kR, d_h, d_hR):
    """Pre-softmax scores, shape (n_h, T, T), from Equations 14-18."""
    T = qC.shape[0]
    pos = np.arange(T)
    qR_rot = rope(np.swapaxes(qR, 0, 1), pos)   # (n_h, T, d_hR)
    kR_rot = rope(kR, pos)                      # (T, d_hR), shared by every head
    content = np.einsum("tid,jid->itj", qC, kC)
    position = np.einsum("itd,jd->itj", qR_rot, kR_rot)
    return (content + position) / np.sqrt(d_h + d_hR)
