import numpy as np


def effective_steps(grads, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True):
    """Return the Adam step Delta_t for each gradient in `grads` (one scalar parameter).

    With bias_correction=False, use the raw moments m_t and v_t.
    """
    raise NotImplementedError
