import numpy as np


def border_weight(d1, d2, w0=10.0, sigma=5.0, wc=1.0):
    """Equation 2: w(x) = wc + w0 * exp(-(d1 + d2)^2 / (2 sigma^2))."""
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    return wc + w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma**2))


def weighted_pixel_loss(logits, labels, w):
    """Return -sum_x w(x) * log p_{l(x)}(x), with the softmax over axis 0 (channels).

    logits: (K, H, W). labels: (H, W) integers in [0, K). w: (H, W).
    """
    shifted = logits - logits.max(axis=0, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=0, keepdims=True))
    true_log_p = np.take_along_axis(log_p, labels[None], axis=0)[0]
    return float(-(w * true_log_p).sum())
