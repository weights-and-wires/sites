import numpy as np


def deepseekmoe_layer(u, centroids, shared, routed, k_active):
    """Return (h, idx) for Equations 9-11.

    h has shape (T, d) and includes the residual u; idx has shape (T, k_active).
    """
    logits = u @ centroids.T
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    s = e / e.sum(axis=-1, keepdims=True)
    idx = np.argsort(-s, axis=-1, kind="stable")[:, :k_active]
    h = u.copy()
    for f in shared:
        h += f(u)
    for i, f in enumerate(routed):
        rows, _ = np.where(idx == i)
        if rows.size == 0:
            continue
        h[rows] += s[rows, i][:, None] * f(u[rows])
    return h, idx
