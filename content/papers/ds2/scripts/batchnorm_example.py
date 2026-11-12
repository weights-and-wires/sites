"""Compare sequence-wise and frame-wise batch normalization (Section 3.2, Equations 6 to 8).

Run: python3 scripts/batchnorm_example.py
"""

import numpy as np

eps = 0.0

# Part 1: two utterances, one unit. Utterance 0 has 3 frames and utterance 1 has 2.
a = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, np.nan]])
valid = a[~np.isnan(a)]
mean, var = valid.mean(), valid.var()
print("Sequence-wise: statistics over every valid (utterance, frame) pair")
print(f"valid values = {valid.tolist()}, mean = {mean:.4f}, variance = {var:.4f}, std = {np.sqrt(var):.4f}")
print(f"normalized utterance 0 = {np.round((a[0] - mean) / np.sqrt(var + eps), 4).tolist()}")
print(f"normalized utterance 1 = {np.round((a[1, :2] - mean) / np.sqrt(var + eps), 4).tolist()}")

# Part 2: frame-wise on the same two utterances, full frames only (T = 2).
b = np.array([[1.0, 2.0], [4.0, 6.0]])
fm, fv = b.mean(axis=0), b.var(axis=0)
print()
print("Frame-wise: statistics over the batch at each time step")
print(f"per-step means = {fm.tolist()}, variances = {fv.tolist()}")
print(f"normalized = {np.round((b - fm) / np.sqrt(fv + eps), 4).tolist()}")

# Part 3: a batch of one utterance.
c = np.array([[1.0, 2.0, 3.0]])
cm, cv = c.mean(axis=0), c.var(axis=0)
print()
print("Batch of one utterance, 3 frames, frame-wise")
print(f"variances per step = {cv.tolist()}, normalized = {((c - cm) / np.sqrt(cv + 1e-5)).round(4).tolist()}")
print("Sequence-wise on the same utterance")
print(f"mean = {c.mean():.4f}, variance = {c.var():.4f}, normalized = {((c - c.mean()) / np.sqrt(c.var() + eps)).round(4).tolist()}")

# Part 4: the input-to-hidden term for all time steps is one matrix product, so it can be normalized before the recurrence.
rng = np.random.default_rng(0)
B, T, din, dh = 4, 6, 5, 3
h_prev = rng.normal(size=(B, T, din))
W = rng.normal(size=(din, dh))
Wh = h_prev @ W  # Equation 8 needs this for every t at once
loop = np.stack([h_prev[:, t] @ W for t in range(T)], axis=1)
print()
print(f"W h for all {T} steps in one product equals the step-by-step loop: {np.allclose(Wh, loop)}")
pooled = Wh.reshape(-1, dh)
print(f"sequence-wise mean over {B * T} pairs, per unit = {np.round(pooled.mean(axis=0), 4).tolist()}")
