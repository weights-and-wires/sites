import numpy as np


def log_sigmoid(a):
    """Return log(sigmoid(a)) without overflow."""
    return -np.logaddexp(0.0, -np.asarray(a, dtype=float))


def discriminator_loss(real_logits, fake_logits):
    """Return -mean[log D(x) + log(1 - D(G(z)))] from the two logit arrays."""
    return float(-np.mean(log_sigmoid(real_logits)) - np.mean(log_sigmoid(-np.asarray(fake_logits, dtype=float))))


def generator_loss(fake_logits, non_saturating):
    """Return mean log(1 - D(G(z))), or -mean log D(G(z)) when non_saturating is True."""
    a = np.asarray(fake_logits, dtype=float)
    if non_saturating:
        return float(-np.mean(log_sigmoid(a)))
    return float(np.mean(log_sigmoid(-a)))


def generator_grad(fake_logits, non_saturating):
    """Return d generator_loss / d fake_logits, including the 1/m factor."""
    a = np.asarray(fake_logits, dtype=float)
    m = a.size
    if non_saturating:
        return -np.exp(log_sigmoid(-a)) / m
    return -np.exp(log_sigmoid(a)) / m
