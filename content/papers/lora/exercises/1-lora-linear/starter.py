import numpy as np


def init_lora(d, k, r, seed=0, sigma=0.02):
    """Return (A, B) with shapes (r, k) and (d, r).

    A is drawn from a Gaussian with standard deviation sigma; B is all zeros
    (Section 4.1), so the update B A is zero before training.
    """
    raise NotImplementedError


def lora_forward(X, W0, A, B, alpha):
    """Return X W0^T + (alpha / r) X A^T B^T, with shape (n, d).

    X is (n, k), W0 is (d, k), A is (r, k), B is (d, r).
    """
    raise NotImplementedError
