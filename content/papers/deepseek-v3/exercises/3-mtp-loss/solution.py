import numpy as np


def rmsnorm(x, eps=1e-6):
    """Divide each row by its root mean square (no learned gain)."""
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)


def mtp_combine(h_prev, emb_next, M):
    """Equation 21: M [RMSNorm(h_prev); RMSNorm(emb_next)] for each row. Shape (n, d)."""
    z = np.concatenate([rmsnorm(h_prev), rmsnorm(emb_next)], axis=-1)
    return z @ M.T


def mtp_loss(logits, tokens, k):
    """Equation 24: cross-entropy of depth k, summed over positions and divided by T."""
    logits = np.asarray(logits, dtype=np.float64)
    tokens = np.asarray(tokens)
    T = len(tokens) - 1
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    targets = tokens[k + 1 : k + 1 + logits.shape[0]]
    return -logp[np.arange(logits.shape[0]), targets].sum() / T


def mtp_total(losses, lam):
    """Equation 25: lam times the mean of the per-depth losses."""
    return lam * float(np.mean(losses))
