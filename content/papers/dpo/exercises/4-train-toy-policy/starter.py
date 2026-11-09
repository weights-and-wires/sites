import numpy as np


def log_softmax(z):
    z = np.asarray(z, dtype=float)
    m = z.max()
    return z - m - np.log(np.sum(np.exp(z - m)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def mean_loss(theta, ref_logits, C, beta):
    """Finished for you: the DPO loss averaged over all counted pairs."""
    rhat = beta * (log_softmax(theta) - log_softmax(ref_logits))
    K = len(theta)
    total = 0.0
    for i in range(K):
        for j in range(K):
            if C[i, j]:
                total += C[i, j] * np.logaddexp(0.0, -(rhat[i] - rhat[j]))
    return float(total / C.sum())


def mean_grad(theta, ref_logits, C, beta):
    """Finished for you: the gradient of mean_loss with respect to theta."""
    rhat = beta * (log_softmax(theta) - log_softmax(ref_logits))
    K = len(theta)
    g = np.zeros(K)
    for i in range(K):
        for j in range(K):
            if C[i, j]:
                e = np.zeros(K)
                e[i], e[j] = 1.0, -1.0
                g += C[i, j] * (-beta * sigmoid(rhat[j] - rhat[i]) * e)
    return g / C.sum()


def train_dpo(ref_logits, C, beta, lr, steps):
    """Gradient descent on the DPO loss. Return (theta, losses).

    Start from theta = ref_logits. losses[t] is the mean loss before step t.
    Leave ref_logits unchanged.
    """
    raise NotImplementedError
