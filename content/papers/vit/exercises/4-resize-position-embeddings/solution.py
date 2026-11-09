import numpy as np


def _axis(g, g_new):
    """Left index, right index, and weight of the right neighbor for each new coordinate."""
    if g == 1:
        z = np.zeros(g_new, dtype=int)
        return z, z, np.zeros(g_new)
    pos = np.linspace(0, g - 1, g_new) if g_new > 1 else np.zeros(1)
    i0 = np.minimum(np.floor(pos).astype(int), g - 2)
    return i0, i0 + 1, pos - i0


def resize_pos_embed(E_pos, g_new):
    """Keep the class row; resize the g x g grid bilinearly with aligned corners."""
    D = E_pos.shape[1]
    g = int(round(np.sqrt(E_pos.shape[0] - 1)))
    grid = E_pos[1:].reshape(g, g, D)
    r0, r1, tr = _axis(g, g_new)
    c0, c1, tc = _axis(g, g_new)
    tr = tr[:, None, None]
    tc = tc[None, :, None]
    top = grid[r0][:, c0] * (1 - tc) + grid[r0][:, c1] * tc
    bottom = grid[r1][:, c0] * (1 - tc) + grid[r1][:, c1] * tc
    out = top * (1 - tr) + bottom * tr
    return np.vstack([E_pos[:1], out.reshape(g_new * g_new, D)])
