"""Compute the numbers for the weight-map (Equation 2) and weighted-loss (Equation 1) examples.

Run: python3 content/papers/unet/scripts/weight_map.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

w0, sigma, wc = 10.0, 5.0, 1.0


def weight(d1, d2):
    return wc + w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma**2))


print("Equation 2 with w_c = 1, w_0 = 10, sigma = 5")
for d1, d2 in [(1, 1), (2, 3), (5, 5), (10, 10), (20, 20)]:
    print(f"  d1={d1:>2}, d2={d2:>2}: w = {weight(d1, d2):.4f}")
print("  squared-sum bug, (d1^2 + d2^2) for (2, 3):", round(wc + w0 * np.exp(-(2**2 + 3**2) / (2 * sigma**2)), 4))

# Equation 1 on a 1 x 2 image with K = 2 classes.
logits = np.array([[[2.0, 0.0]], [[0.0, 1.0]]])  # shape (K, H, W) = (2, 1, 2)
labels = np.array([[0, 0]])
w = np.array([[1.0, 7.0]])
shifted = logits - logits.max(axis=0, keepdims=True)
p = np.exp(shifted) / np.exp(shifted).sum(axis=0, keepdims=True)
p_true = np.take_along_axis(p, labels[None], axis=0)[0]
print("p_k(x) pixel 0:", p[:, 0, 0].tolist(), " pixel 1:", p[:, 0, 1].tolist())
print("p_l(x)(x):", p_true.tolist())
print("-log p_l(x)(x):", (-np.log(p_true)).tolist())
print("weighted terms:", (-w * np.log(p_true)).tolist())
print("loss (weighted sum):", round(float((-w * np.log(p_true)).sum()), 4))
print("loss (unweighted sum):", round(float((-np.log(p_true)).sum()), 4))

# Initialization: standard deviation sqrt(2/N) for a 3x3 convolution on 64 channels.
N = 9 * 64
print("N =", N, " std = sqrt(2/N) =", round(float(np.sqrt(2 / N)), 4))
rng = np.random.default_rng(0)
x = rng.normal(size=(64, 40, 40))
W = rng.normal(scale=np.sqrt(2 / N), size=(64, 64, 3, 3))
out = np.zeros((64, 38, 38))
for i in range(3):
    for j in range(3):
        out += np.einsum("oc,chw->ohw", W[:, :, i, j], x[:, i : i + 38, j : j + 38])
print("input variance:", round(float(x.var()), 3))
print("pre-ReLU output variance with std sqrt(2/N):", round(float(out.var()), 3))
print("post-ReLU mean square:", round(float((np.maximum(out, 0) ** 2).mean()), 3))
W2 = rng.normal(scale=np.sqrt(1 / N), size=(64, 64, 3, 3))
out2 = np.zeros((64, 38, 38))
for i in range(3):
    for j in range(3):
        out2 += np.einsum("oc,chw->ohw", W2[:, :, i, j], x[:, i : i + 38, j : j + 38])
print("pre-ReLU output variance with std sqrt(1/N):", round(float(out2.var()), 3))
print("post-ReLU mean square with std sqrt(1/N):", round(float((np.maximum(out2, 0) ** 2).mean()), 3))
