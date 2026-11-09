import numpy as np


def select_experts(s, b, k):
    """Indices of the top-k experts per token by s + b (Equation 16). Shape (T, k)."""
    raise NotImplementedError


def gate_values(s, idx):
    """Gating values (Equation 13): original s of the chosen experts, normalized over them."""
    raise NotImplementedError


def expert_load(idx, n):
    """Number of tokens routed to each of the n experts."""
    raise NotImplementedError


def update_bias(b, load, gamma):
    """Subtract gamma for overloaded experts, add gamma for underloaded ones."""
    raise NotImplementedError
