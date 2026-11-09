import numpy as np


def nar_input(text_emb, prompt_codes, target_codes, tables, i):
    """Build the NAR transformer input for predicting codebook i (1-based, 2 <= i <= K).

    text_emb:      (n_text, d) phoneme embeddings, already computed.
    prompt_codes:  (T_prompt, K) integer codes of the enrolled recording.
    target_codes:  (T_target, K) integer codes; only columns 0 .. i-2 are valid inputs.
    tables:        (K, V, d) acoustic embedding tables; codebook j is tables[j - 1].
    Returns an array of shape (n_text + T_prompt + T_target, d).
    """
    K = tables.shape[0]
    if not 2 <= i <= K:
        raise ValueError(f"stage i must satisfy 2 <= i <= {K}, got {i}")
    e_prompt = sum(tables[j][prompt_codes[:, j]] for j in range(K))
    e_target = sum(tables[j][target_codes[:, j]] for j in range(i - 1))
    return np.concatenate([text_emb, e_prompt, e_target], axis=0)


def ada_layer_norm(h, a, b):
    """Return a * LayerNorm(h) + b over the last axis, with eps = 1e-5."""
    mu = h.mean(axis=-1, keepdims=True)
    var = h.var(axis=-1, keepdims=True)
    return a * (h - mu) / np.sqrt(var + 1e-5) + b
