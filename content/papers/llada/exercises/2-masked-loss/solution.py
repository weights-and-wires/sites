import numpy as np


def masked_cross_entropy(logits, x0, masked, t):
    """The pre-training loss of Equation 3 for one sequence and one t.

    logits: (L, V) float array, the model's output at each position.
    x0: (L,) int array of the original tokens.
    masked: (L,) bool array, True where the token was replaced by M.
    t: float in (0, 1].
    Returns (1 / t) * the sum over masked positions of -log p(x0[i]).
    """
    z = logits - logits.max(axis=-1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(len(x0)), x0]
    return float(nll[masked].sum() / t)
