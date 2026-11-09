"""Show that batch normalization depends on the other examples and layer normalization does not.

Batch norm here uses Equation 2 with batch statistics (gain 1, no epsilon unless noted).
Layer norm uses Equation 3.

Run: python3 scripts/batch_dependence.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

x = np.array([1.0, 2.0, 3.0, 6.0])


def layer_norm(batch, eps=0.0):
    mu = batch.mean(axis=-1, keepdims=True)
    var = batch.var(axis=-1, keepdims=True)
    return (batch - mu) / np.sqrt(var + eps)


def batch_norm(batch, eps=0.0):
    mu = batch.mean(axis=0, keepdims=True)
    var = batch.var(axis=0, keepdims=True)
    return (batch - mu) / np.sqrt(var + eps)


partner_a = np.array([3.0, 0.0, 1.0, 2.0])
partner_b = np.array([-4.0, 8.0, 5.0, 0.0])

print("normalized x when its batch partner is A, then B")
print("batch norm, partner A:", batch_norm(np.stack([x, partner_a]))[0])
print("batch norm, partner B:", batch_norm(np.stack([x, partner_b]))[0])
print("layer norm, partner A:", layer_norm(np.stack([x, partner_a]))[0])
print("layer norm, partner B:", layer_norm(np.stack([x, partner_b]))[0])
print("layer norm, alone    :", layer_norm(x[None])[0])
print()
print("batch of one: batch norm")
single = batch_norm(x[None], eps=1e-5)
print("  batch norm output:", single[0], "(every feature has zero variance across a batch of 1)")
print("  layer norm output:", layer_norm(x[None], eps=1e-5)[0])
