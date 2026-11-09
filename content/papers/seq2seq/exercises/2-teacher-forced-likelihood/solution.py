import numpy as np


def log_softmax(z):
    """Log-softmax over the last axis. Finished for you; it is stable for large logits."""
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def decoder_inputs(target, eos):
    """Teacher-forced decoder inputs for a target that ends with eos."""
    return [eos] + list(target[:-1])


def sequence_log_prob(logits, target):
    """Sum over steps of log p(y_t | ...). logits has shape (T', V)."""
    logits = np.asarray(logits, dtype=float)
    lp = log_softmax(logits)
    return float(lp[np.arange(len(target)), np.asarray(target)].sum())


def perplexity(total_log_prob, n_tokens):
    """Per-token perplexity: exp(-total_log_prob / n_tokens)."""
    return float(np.exp(-total_log_prob / n_tokens))
