import numpy as np


def cfg_velocity(v_cond, v_uncond, w):
    """Equation 4: v_cond + w * (v_cond - v_uncond)."""
    raise NotImplementedError


def euler_cfg(v_fn, x0, ts, w):
    """Euler integration with classifier-free guidance.

    v_fn(x, t, cond) returns the velocity; cond is True or False.
    ts: increasing flow-step boundaries from 0 to 1 (not necessarily even).
    For each interval, x = x + (ts[i+1] - ts[i]) * cfg_velocity(...), with the
    velocities evaluated at x and ts[i]. Skip the unconditional call if w == 0.
    Do not modify x0.
    """
    raise NotImplementedError
