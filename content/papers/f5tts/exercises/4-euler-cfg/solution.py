import numpy as np


def cfg_velocity(v_cond, v_uncond, w):
    """Equation 4: v_cond + w * (v_cond - v_uncond)."""
    return v_cond + w * (v_cond - v_uncond)


def euler_cfg(v_fn, x0, ts, w):
    """Euler integration with classifier-free guidance.

    v_fn(x, t, cond) returns the velocity; cond is True or False.
    ts: increasing flow-step boundaries from 0 to 1 (not necessarily even).
    For each interval, x = x + (ts[i+1] - ts[i]) * cfg_velocity(...), with the
    velocities evaluated at x and ts[i]. Skip the unconditional call if w == 0.
    Do not modify x0.
    """
    x = np.array(x0, dtype=float)
    for a, b in zip(ts[:-1], ts[1:]):
        vc = v_fn(x, a, True)
        if w == 0:
            v = vc
        else:
            v = cfg_velocity(vc, v_fn(x, a, False), w)
        x = x + (b - a) * v
    return x
