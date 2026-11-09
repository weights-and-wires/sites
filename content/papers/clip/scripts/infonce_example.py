#!/usr/bin/env python3
"""The symmetric contrastive loss on a 2-pair batch, plus two reference values.

Follows the pseudocode in Figure 3 of the CLIP paper. Run from this folder:

    python3 scripts/infonce_example.py
"""

import numpy as np


def softmax(x, axis):
    z = np.exp(x - x.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


def clip_loss(I_e, T_e, scale):
    """I_e, T_e: (n, d_e), already L2-normalized. scale = exp(t). Returns the pieces."""
    logits = scale * I_e @ T_e.T
    n = len(logits)
    idx = np.arange(n)
    p_img = softmax(logits, axis=1)   # for each image, a distribution over the n texts
    p_txt = softmax(logits, axis=0)   # for each text, a distribution over the n images
    loss_img = -np.log(p_img[idx, idx]).mean()
    loss_txt = -np.log(p_txt[idx, idx]).mean()
    return logits, p_img, p_txt, loss_img, loss_txt, (loss_img + loss_txt) / 2


np.set_printoptions(precision=5, suppress=True)

# A batch of two pairs. Image 1 and text 1 point the same way; text 2 leans toward image 1.
I_e = np.array([[1.0, 0.0], [0.0, 1.0]])
T_e = np.array([[1.0, 0.0], [0.6, 0.8]])
S = I_e @ T_e.T
print("cosine similarities I_e @ T_e.T:")
print(S)

scale = 10.0
logits, p_img, p_txt, l_img, l_txt, loss = clip_loss(I_e, T_e, scale)
print(f"\nscale = {scale}")
print("logits = scale * S:")
print(logits)
print("softmax over the texts (each row), diagonal entries:", np.diag(p_img))
print("softmax over the images (each column), diagonal entries:", np.diag(p_txt))
print(f"image-to-text loss = {l_img:.5f}")
print(f"text-to-image loss = {l_txt:.5f}")
print(f"symmetric loss     = {loss:.5f}")

print("\nIdentity similarity: orthonormal embeddings, so S is the identity matrix.")
for n, s in [(2, 10.0), (4, 1.0), (4, 14.2857), (8, 1.0)]:
    I = np.eye(n)
    got = clip_loss(I, I, s)[-1]
    want = np.log(1 + (n - 1) * np.exp(-s))
    print(f"  n = {n}, scale = {s:>7}: loss = {got:.5f}, ln(1 + (n-1)e^-scale) = {want:.5f}")

print("\nWith scale 0 every logit is 0, so the loss is ln n:")
for n in (2, 4, 32768):
    print(f"  n = {n}: ln n = {np.log(n):.5f}")
