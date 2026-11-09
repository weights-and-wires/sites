"""Compute the numbers for the gradient-flow examples.

Run: python3 scripts/gradient_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Part 1: a scalar chain with 20 weight layers, all weights 0.3, ignoring ReLU
# (every value is positive, so the ReLU is the identity).
w = 0.3
plain = w**20
per_block = 1 + w * w  # y = x + w * (w * x), so dy/dx = 1 + w^2
residual = per_block**10
print("Scalar chain, 20 weights of 0.3")
print(f"plain:    dy/dx = 0.3^20 = {plain:.3e}")
print(f"residual: dy/dx per block = 1 + 0.3^2 = {per_block:.2f}; ten blocks = {residual:.4f}")


# Part 2: gradient norms per layer in a 20-matrix MLP (the setup of exercise 3).
def relu(x):
    return np.maximum(x, 0.0)


def make_net(depth, d, seed, residual, gain=0.5):
    rng = np.random.default_rng(seed)
    std = gain * np.sqrt(2.0 / d)
    mats = [rng.normal(0, std, size=(d, d)) for _ in range(depth)]
    if residual:
        return [(mats[2 * i], mats[2 * i + 1]) for i in range(depth // 2)]
    return mats


def norms(net, x, residual):
    h, caches = x, []
    for layer in net:
        if residual:
            W1, W2 = layer
            a = W1 @ h
            z = W2 @ relu(a) + h
            caches.append((a, z))
        else:
            z = layer @ h
            caches.append((None, z))
        h = relu(z)
    g = h.copy()
    out = [np.linalg.norm(g)]
    for layer, (a, z) in zip(reversed(net), reversed(caches)):
        g_z = g * (z > 0)
        if residual:
            W1, W2 = layer
            g = W1.T @ ((W2.T @ g_z) * (a > 0)) + g_z
        else:
            g = layer.T @ g_z
        out.append(np.linalg.norm(g))
    return np.array(out[::-1])


print()
print("20-matrix MLP, width 32, weight std 0.5 * sqrt(2 / 32), seed 0, loss 0.5 ||h_n||^2")
x = np.random.default_rng(100).normal(size=32)
for residual_flag, name in ((False, "plain (20 layers)"), (True, "residual (10 blocks)")):
    n = norms(make_net(20, 32, 0, residual_flag), x, residual_flag)
    print(f"{name}: ||g_0|| = {n[0]:.3e}, ||g_n|| = {n[-1]:.3e}, ratio = {n[0] / n[-1]:.3e}")
    print("   norms at layers 0, n/4, n/2, 3n/4, n:", [f"{n[int(i)]:.2e}" for i in np.linspace(0, len(n) - 1, 5)])

print()
print("Ratio ||g_0|| / ||g_n|| for seeds 0 to 3")
for residual_flag, name in ((False, "plain"), (True, "residual")):
    r = []
    for seed in range(4):
        xs = np.random.default_rng(100 + seed).normal(size=32)
        n = norms(make_net(20, 32, seed, residual_flag), xs, residual_flag)
        r.append(n[0] / n[-1])
    print(f"{name}:", [f"{v:.2e}" for v in r])
