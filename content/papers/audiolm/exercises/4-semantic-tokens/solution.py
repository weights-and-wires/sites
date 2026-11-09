import numpy as np


def semantic_tokens(feats, centroids, mean, std):
    """Normalize feats (T, D) with mean and std, then return the index of the
    nearest centroid (K, D) in squared Euclidean distance, as an int array (T,)."""
    norm = (np.asarray(feats, dtype=float) - mean) / std
    d = ((norm[:, None, :] - np.asarray(centroids)[None, :, :]) ** 2).sum(-1)
    return d.argmin(axis=1)


def collapse_repeats(tokens):
    """Remove consecutive repeats. Return (kept, run_lengths) as int arrays."""
    tokens = np.asarray(tokens)
    if tokens.size == 0:
        return tokens.copy(), np.zeros(0, dtype=int)
    keep = np.concatenate([[True], tokens[1:] != tokens[:-1]])
    starts = np.flatnonzero(keep)
    lengths = np.diff(np.append(starts, len(tokens)))
    return tokens[keep], lengths
