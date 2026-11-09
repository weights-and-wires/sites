import numpy as np


def outlier_dims(X, alpha=6.0):
    """Return the sorted column indices of X whose largest magnitude is at least alpha."""
    return np.where(np.max(np.abs(X), axis=0) >= alpha)[0]
