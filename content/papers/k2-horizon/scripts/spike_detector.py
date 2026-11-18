"""Robust z-score spike detector (Section 3.3.2, Equation 12) on a synthetic loss curve.

The curve is made up. The window is 1% of the run, as in the paper. The
trailing window excludes the current step; that choice is this page's reading.
"""
import numpy as np


def robust_z(y, w):
    z = np.zeros_like(y)
    for t in range(w, len(y)):
        win = y[t - w:t]
        m = np.median(win)
        mad = np.median(np.abs(win - m))
        z[t] = (y[t] - m) / (1.4826 * mad)
    return z


def episodes(z, on=5.0, off=2.0, calm=3):
    out, start, run = [], None, 0
    for t, v in enumerate(z):
        if start is None:
            if v > on:
                start, run = t, 0
        else:
            run = run + 1 if v < off else 0
            if run == calm:
                out.append((start, t))
                start = None
    if start is not None:
        out.append((start, len(z) - 1))
    return out


rng = np.random.default_rng(0)
T = 20_000
steps = np.arange(T)
loss = 2.0 + 1.5 * np.exp(-steps / 3000) + rng.normal(0, 0.01, T)
loss[8000:8004] += [0.30, 0.20, 0.10, 0.02]
loss[15000] += 0.08
loss[8150] += 0.08
w = T // 100
z = robust_z(loss, w)
print("window", w)
print("episodes", episodes(z))
print("z at 8000..8003", np.round(z[8000:8004], 1).tolist())
print("z at 15000", round(float(z[15000]), 1))

print("z at 8150", round(float(z[8150]), 1))

# A mean and standard deviation z-score at 8150 sees the step-8000 spike in its window.
win = loss[8150 - w:8150]
print("plain z at 8150", round(float((loss[8150] - win.mean()) / win.std()), 1))
