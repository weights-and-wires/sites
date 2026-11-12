"""Worked numbers for "Softmax needs somewhere to put its attention".

Run from the paper folder: python3 scripts/sink_softmax.py

A toy query has nothing relevant to attend to: every non-sink logit is close
to 0. Token 0 is the sink, with a large logit and a value vector near zero.
The logits and values are this page's own, not the paper's.
"""
import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def softmax_one(x):
    """Equation 2: SoftMax_1(x)_i = exp(x_i) / (1 + sum_j exp(x_j)), computed stably."""
    m = max(np.max(x), 0.0)
    e = np.exp(x - m)
    return e / (np.exp(-m) + e.sum())


rng = np.random.default_rng(0)
x = np.array([6.0, 0.2, -0.1, 0.3, 0.0, -0.2, 0.1, 0.2])
V = rng.normal(size=(8, 4))
V[0] = 0.01 * V[0]  # the sink carries almost no information

w = softmax(x)
print("weights with the sink:", np.round(w, 4))
print("mass on the sink:", round(float(w[0]), 4))
print("largest non-sink weight:", round(float(w[1:].max()), 4))
out_full = w @ V
print("output norm with the sink:", round(float(np.linalg.norm(out_full)), 4))

w_ev = softmax(x[1:])
print("\nweights after evicting the sink:", np.round(w_ev, 4))
print("each non-sink weight grows by a factor of", np.round(w_ev / w[1:], 1))
out_ev = w_ev @ V[1:]
print("output norm without the sink:", round(float(np.linalg.norm(out_ev)), 4))
print("distance between the two outputs:", round(float(np.linalg.norm(out_full - out_ev)), 4))

# Equation 2: an extra 1 in the denominator lets the weights sum to less than 1.
x0 = np.array([0.2, -0.1, 0.3, 0.0, -0.2, 0.1, 0.2])
print("\nsoftmax over the 7 non-sink tokens sums to", round(float(softmax(x0).sum()), 4))
print("SoftMax_1 over the same logits sums to", round(float(softmax_one(x0).sum()), 4))
big = np.array([6.0, 5.0, 4.0])
print("with large logits, SoftMax_1 sums to", round(float(softmax_one(big).sum()), 4))
