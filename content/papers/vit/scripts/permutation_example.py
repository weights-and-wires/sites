#!/usr/bin/env python3
"""Without position embeddings, a ViT cannot tell the order of patches apart."""
import numpy as np

rng = np.random.default_rng(0)
D, heads, mlp, N = 8, 2, 16, 6


def layernorm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def block(z, p):
    x = layernorm(z, p["g1"], p["b1"])
    n = len(x)
    split = lambda M: (x @ M).reshape(n, heads, D // heads).transpose(1, 0, 2)
    q, k, v = split(p["Wq"]), split(p["Wk"]), split(p["Wv"])
    a = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(D // heads)) @ v
    z = z + a.transpose(1, 0, 2).reshape(n, D) @ p["Wo"]
    return z + gelu(layernorm(z, p["g2"], p["b2"]) @ p["W1"]) @ p["W2"]


p = dict(g1=np.ones(D), b1=np.zeros(D), g2=np.ones(D), b2=np.zeros(D),
         Wq=rng.normal(size=(D, D)), Wk=rng.normal(size=(D, D)), Wv=rng.normal(size=(D, D)),
         Wo=rng.normal(size=(D, D)), W1=rng.normal(size=(D, mlp)), W2=rng.normal(size=(mlp, D)))
cls = rng.normal(size=(1, D))
patches = rng.normal(size=(N, D))
pos = rng.normal(size=(N + 1, D))
perm = rng.permutation(N)
print("permutation of the patches:", perm.tolist())


def readout(patch_rows, with_pos):
    z = np.vstack([cls, patch_rows])
    if with_pos:
        z = z + pos
    return block(block(z, p), p)[0]


for with_pos in (False, True):
    a = readout(patches, with_pos)
    b = readout(patches[perm], with_pos)
    print(f"position embeddings {'on ' if with_pos else 'off'}: max |class output difference| = {np.abs(a - b).max():.2e}")
