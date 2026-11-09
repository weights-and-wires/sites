"""Semantic tokens: normalize features, then pick the nearest k-means centroid (Section III-B).

Run: python3 scripts/kmeans_tokens.py
"""

import numpy as np

feats = np.array([[10.0, 0.1], [30.0, 0.9], [22.0, 0.2]])
mean = np.array([20.0, 0.5])
std = np.array([10.0, 0.4])
centroids = np.array([[-1.0, -1.0], [1.0, 1.0], [0.2, -1.0]])  # in normalized space

norm = (feats - mean) / std
print("normalized features =", norm.tolist())
d = ((norm[:, None, :] - centroids[None]) ** 2).sum(-1)
print("squared distances   =", np.round(d, 4).tolist())
print("tokens              =", d.argmin(1).tolist())

d_raw = ((feats[:, None, :] - centroids[None]) ** 2).sum(-1)
print("tokens without normalizing =", d_raw.argmin(1).tolist())
print("tokens with argmax (wrong) =", d.argmax(1).tolist())
