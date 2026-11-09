import numpy as np


def log_softmax(z):
    z = np.asarray(z, dtype=float)
    m = z.max()
    return z - m - np.log(np.sum(np.exp(z - m)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dpo_loss(theta, ref_logits, pairs, beta):
    """Finished for you: the mean DPO loss over the pairs."""
    rhat = beta * (log_softmax(theta) - log_softmax(ref_logits))
    return float(np.mean([np.logaddexp(0.0, -(rhat[w] - rhat[l])) for w, l in pairs]))


def dpo_grad(theta, ref_logits, pairs, beta):
    """Gradient of the mean DPO loss with respect to theta, shape (K,)."""
    theta = np.asarray(theta, dtype=float)
    K = theta.shape[0]
    log_pi = log_softmax(theta)
    pi = np.exp(log_pi)
    rhat = beta * (log_pi - log_softmax(ref_logits))
    grad = np.zeros(K)
    for w, l in pairs:
        weight = sigmoid(rhat[l] - rhat[w])
        grad_log_w = np.eye(K)[w] - pi
        grad_log_l = np.eye(K)[l] - pi
        grad += -beta * weight * (grad_log_w - grad_log_l)
    return grad / len(pairs)
