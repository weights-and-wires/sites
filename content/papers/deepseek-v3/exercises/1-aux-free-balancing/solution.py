import numpy as np


def select_experts(s, b, k):
    """Indices of the top-k experts per token by s + b (Equation 16). Shape (T, k)."""
    return np.argsort(-(s + b), axis=1)[:, :k]


def gate_values(s, idx):
    """Gating values (Equation 13): original s of the chosen experts, normalized over them."""
    chosen = np.take_along_axis(s, idx, axis=1)
    return chosen / chosen.sum(axis=1, keepdims=True)


def expert_load(idx, n):
    """Number of tokens routed to each of the n experts."""
    return np.bincount(np.asarray(idx).ravel(), minlength=n)


def update_bias(b, load, gamma):
    """Subtract gamma for overloaded experts, add gamma for underloaded ones."""
    return b - gamma * np.sign(load - load.mean())
