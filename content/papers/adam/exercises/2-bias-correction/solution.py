import numpy as np


def effective_steps(grads, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True):
    """Return the Adam step Delta_t for each gradient in `grads` (one scalar parameter).

    With bias_correction=False, use the raw moments m_t and v_t.
    """
    m = 0.0
    v = 0.0
    out = []
    for t, g in enumerate(grads, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        if bias_correction:
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
        else:
            m_hat, v_hat = m, v
        out.append(alpha * m_hat / (np.sqrt(v_hat) + eps))
    return np.array(out)
