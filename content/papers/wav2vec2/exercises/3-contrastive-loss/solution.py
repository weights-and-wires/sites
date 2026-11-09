import numpy as np


def sample_distractors(masked_idx, K, rng):
    """Return an int array of shape (len(masked_idx), K).

    Row i holds K distinct time steps drawn uniformly from the masked steps other
    than masked_idx[i]. Raise ValueError when fewer than K other masked steps exist.
    """
    masked_idx = np.asarray(masked_idx)
    if len(masked_idx) - 1 < K:
        raise ValueError(f"need at least {K} other masked steps, have {len(masked_idx) - 1}")
    out = np.empty((len(masked_idx), K), dtype=int)
    for i, t in enumerate(masked_idx):
        others = masked_idx[masked_idx != t]
        out[i] = rng.choice(others, size=K, replace=False)
    return out


def contrastive_loss(c, q, masked_idx, distractors, kappa):
    """Return the mean contrastive loss of Equation 3 over the masked steps.

    c: (T, d) context vectors. q: (T, d) quantized targets.
    masked_idx: (n,) masked time steps. distractors: (n, K) time steps.
    kappa: temperature. sim is cosine similarity.
    """
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    total = 0.0
    for i, t in enumerate(masked_idx):
        cand = qn[[t, *distractors[i]]]
        logits = cand @ cn[t] / kappa
        m = logits.max()
        lse = m + np.log(np.exp(logits - m).sum())
        total += lse - logits[0]
    return float(total / len(masked_idx))
