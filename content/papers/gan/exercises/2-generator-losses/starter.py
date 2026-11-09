import numpy as np


def log_sigmoid(a):
    """Return log(sigmoid(a)) without overflow."""
    raise NotImplementedError


def discriminator_loss(real_logits, fake_logits):
    """Return -mean[log D(x) + log(1 - D(G(z)))] from the two logit arrays."""
    raise NotImplementedError


def generator_loss(fake_logits, non_saturating):
    """Return mean log(1 - D(G(z))), or -mean log D(G(z)) when non_saturating is True."""
    raise NotImplementedError


def generator_grad(fake_logits, non_saturating):
    """Return d generator_loss / d fake_logits, including the 1/m factor."""
    raise NotImplementedError
