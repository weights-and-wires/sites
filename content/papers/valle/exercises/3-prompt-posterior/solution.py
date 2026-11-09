import numpy as np


def speaker_posterior(prompt, trans, prior):
    """Return p(k | prompt) as an array of shape (K,).

    prompt: list of ints. trans: (K, V, V), rows sum to 1. prior: (K,).
    """
    logp = np.log(np.asarray(prior, dtype=float)).copy()
    for a, b in zip(prompt[:-1], prompt[1:]):
        logp += np.log(trans[:, a, b])
    p = np.exp(logp - logp.max())
    return p / p.sum()


def next_token_dist(prefix, trans, prior):
    """Return p(next token | prefix) as an array of shape (V,), mixing over speakers."""
    return speaker_posterior(prefix, trans, prior) @ trans[:, prefix[-1], :]


def continue_greedy(prefix, n, trans, prior):
    """Return prefix + n greedy tokens, as a list of ints."""
    seq = list(prefix)
    for _ in range(n):
        seq.append(int(np.argmax(next_token_dist(seq, trans, prior))))
    return seq
