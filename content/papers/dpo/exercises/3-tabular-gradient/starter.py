import numpy as np


def log_softmax(z):
    z = np.asarray(z, dtype=float)
    m = z.max()
    return z - m - np.log(np.sum(np.exp(z - m)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dpo_loss(theta, ref_logits, pairs, beta):
    """Finished for you: the mean DPO loss over the pairs."""
    rhat = beta * (log_softmax(theta) - log_softmax(ref_logits))
    return float(np.mean([np.logaddexp(0.0, -(rhat[w] - rhat[l])) for w, l in pairs]))


def dpo_grad(theta, ref_logits, pairs, beta):
    """Gradient of the mean DPO loss with respect to theta, shape (K,).

    theta, ref_logits: arrays of shape (K,). pairs: list of (w, l) response indices.
    """
    raise NotImplementedError
