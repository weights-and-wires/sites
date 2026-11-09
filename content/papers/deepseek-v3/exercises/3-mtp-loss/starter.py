import numpy as np


def rmsnorm(x, eps=1e-6):
    """Divide each row by its root mean square (no learned gain)."""
    raise NotImplementedError


def mtp_combine(h_prev, emb_next, M):
    """Equation 21: M [RMSNorm(h_prev); RMSNorm(emb_next)] for each row. Shape (n, d)."""
    raise NotImplementedError


def mtp_loss(logits, tokens, k):
    """Equation 24: cross-entropy of depth k, summed over positions and divided by T."""
    raise NotImplementedError


def mtp_total(losses, lam):
    """Equation 25: lam times the mean of the per-depth losses."""
    raise NotImplementedError
