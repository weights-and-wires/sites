import numpy as np


def border_weight(d1, d2, w0=10.0, sigma=5.0, wc=1.0):
    """Equation 2: w(x) = wc + w0 * exp(-(d1 + d2)^2 / (2 sigma^2))."""
    raise NotImplementedError


def weighted_pixel_loss(logits, labels, w):
    """Return -sum_x w(x) * log p_{l(x)}(x), with the softmax over axis 0 (channels).

    logits: (K, H, W). labels: (H, W) integers in [0, K). w: (H, W).
    """
    raise NotImplementedError
