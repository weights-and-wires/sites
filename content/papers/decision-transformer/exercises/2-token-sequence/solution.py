import numpy as np


def causal_self_attention(X):
    """Finished for you: single-head causal attention with identity projections.

    X has shape (n, d). Row i of the output is a softmax-weighted average of
    rows 0..i of X.
    """
    n, d = X.shape
    scores = X @ X.T / np.sqrt(d)
    scores = np.where(np.tril(np.ones((n, n), dtype=bool)), scores, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ X


def build_tokens(R_emb, s_emb, a_emb, t_emb):
    """Return the (3K, d) token sequence R_1, s_1, a_1, ..., R_K, s_K, a_K.

    All four arguments have shape (K, d). t_emb[t] is added to the return, state,
    and action embedding of timestep t.
    """
    K, d = R_emb.shape
    stacked = np.stack([R_emb + t_emb, s_emb + t_emb, a_emb + t_emb], axis=1)
    return stacked.reshape(3 * K, d)


def state_token_indices(K):
    """Positions of the state tokens in the (3K, d) sequence, as an int array."""
    return np.arange(K) * 3 + 1
