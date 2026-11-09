"""Derive-and-check the layer norm backward pass.

The paper gives no backward formulas, so this script checks the ones on the page
against central finite differences.

With xhat = (x - mu) / sigma, y = g * xhat + b, and dxhat = g * dy:
    dx = (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)) / sigma
    dg = sum over the batch of dy * xhat
    db = sum over the batch of dy

Run: python3 scripts/gradient_check.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def forward(x, g, b, eps=0.0):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    sigma = np.sqrt(var + eps)
    xhat = (x - mu) / sigma
    return g * xhat + b, (xhat, sigma, g)


def backward(dy, cache):
    xhat, sigma, g = cache
    dxhat = dy * g
    dx = (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / sigma
    return dx, (dy * xhat).sum(0), dy.sum(0)


def numeric(f, p, h=1e-6):
    grad = np.zeros_like(p)
    it = np.nditer(p, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = p[i]
        p[i] = old + h
        hi = f()
        p[i] = old - h
        lo = f()
        p[i] = old
        grad[i] = (hi - lo) / (2 * h)
    return grad


rng = np.random.default_rng(0)
x = rng.normal(size=(3, 4))
g = rng.normal(size=4)
b = rng.normal(size=4)
w = rng.normal(size=(3, 4))  # loss = sum(w * y), so dy = w

y, cache = forward(x, g, b)
dx, dg, db = backward(w, cache)
loss = lambda: float((w * forward(x, g, b)[0]).sum())

for name, analytic, p in (("dx", dx, x), ("dg", dg, g), ("db", db, b)):
    num = numeric(loss, p)
    err = np.abs(analytic - num).max()
    print(f"{name}: max |analytic - numeric| = {err:.2e}")

print()
print("rows of dx sum to      ", dx.sum(-1))
xhat = cache[0]
print("rows of dx . xhat equal", (dx * xhat).sum(-1))

# The same gradient without the two correction terms, for contrast.
naive = w * g / cache[1]
print()
print("max |dx - naive dx| with naive = g * dy / sigma:", f"{np.abs(dx - naive).max():.4f}")
