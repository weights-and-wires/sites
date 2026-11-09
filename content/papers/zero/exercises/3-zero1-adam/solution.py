import numpy as np


def adam_update(p, g, m, v, t, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    """Finished for you. One Adam step on arrays of any shape; returns (p, m, v)."""
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    return p - lr * m_hat / (np.sqrt(v_hat) + eps), m, v


def shard_slices(psi, nd):
    """Finished for you. Slices that split range(psi) into nd nearly equal parts."""
    return [slice(int(a[0]), int(a[-1]) + 1) for a in np.array_split(np.arange(psi), nd)]


def zero1_step(params, worker_grads, m_shards, v_shards, t, lr=0.1):
    """One ZeRO-1 step. Update m_shards and v_shards in place; return the new full params."""
    nd, psi = worker_grads.shape
    g = worker_grads.mean(axis=0)
    new = []
    for i, s in enumerate(shard_slices(psi, nd)):
        p_i, m_shards[i], v_shards[i] = adam_update(params[s], g[s], m_shards[i], v_shards[i], t, lr)
        new.append(p_i)
    return np.concatenate(new)
