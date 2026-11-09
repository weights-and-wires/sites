#!/usr/bin/env python3
"""Zero-shot classification and prompt ensembling in embedding space.

Section 3.1.2 builds a linear classifier from class-name text embeddings.
Section 3.1.4 averages the text embeddings of several prompts per class.

    python3 scripts/zero_shot_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def softmax(x):
    z = np.exp(x - x.max())
    return z / z.sum()


def at(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


scale = 100.0  # the clipped maximum of Section 2.5

# Two classes in a 2-D embedding space, so each embedding is an angle. The image sits at 30 degrees.
# The true class is class 0. Prompt A places class 1 closer to the image than class 0.
image = at(30)
prompts = np.array([
    [at(60), at(5)],     # prompt A: [class 0, class 1]
    [at(20), at(100)],   # prompt B
])
print("image at 30 degrees; class 0 is the true class")

for name, W in zip("AB", prompts):
    cos = W @ image
    print(f"\nprompt {name} alone: cosines {cos}, probabilities {softmax(scale * cos)}")

W = unit(unit(prompts).mean(axis=0))  # embedding-space ensemble
print("\nensemble classifier (normalize each prompt, average, normalize again):")
print(W)
print("angles of the ensemble rows (degrees):", np.rad2deg(np.arctan2(W[:, 1], W[:, 0])))
cos = W @ image
print("cosines", cos, " probabilities", softmax(scale * cos))

p_space = np.mean([softmax(scale * unit(p) @ image) for p in prompts], axis=0)
print("\naveraging the two probability vectors instead:", p_space)

print("\nnorm of each averaged embedding before the second normalization:",
      np.linalg.norm(unit(prompts).mean(axis=0), axis=1))

print("\nCost: 1000 classes x 80 prompts = %d text-encoder calls, once." % (1000 * 80))
print("Per image afterward: one image-encoder pass and 1000 dot products.")
