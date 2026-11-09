import numpy as np


def log_softmax(z):
    """Log-softmax over the last axis. Finished for you; it is stable for large logits."""
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def decoder_inputs(target, eos):
    """Teacher-forced decoder inputs for a target that ends with eos."""
    raise NotImplementedError


def sequence_log_prob(logits, target):
    """Sum over steps of log p(y_t | ...). logits has shape (T', V)."""
    raise NotImplementedError


def perplexity(total_log_prob, n_tokens):
    """Per-token perplexity: exp(-total_log_prob / n_tokens)."""
    raise NotImplementedError
