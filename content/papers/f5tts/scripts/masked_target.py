"""Build the infilling inputs and the masked flow-matching loss
(Section 3.1; Equations 3 and 5 are on the other scripts).

Run: python3 scripts/masked_target.py
"""

import numpy as np

# One mel "channel" over six frames, so every number fits in a table.
x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # data (the mel spectrogram)
x0 = np.array([0.5, -1.0, 0.0, 1.0, -0.5, 2.0])  # noise sample
m = np.array([0, 0, 1, 1, 1, 0])  # 1 = masked frame the model must generate
t = 0.5

x_t = (1 - t) * x0 + t * x1  # noisy speech
cond = (1 - m) * x1  # masked speech (1 - m) * x1
target = x1 - x0  # OT target with sigma_min = 0 (Equation 3)

print("frame      ", np.arange(6))
print("x1         ", x1)
print("x0         ", x0)
print("mask m     ", m)
print("x_t        ", x_t)
print("cond       ", cond)
print("target     ", target)


def masked_loss(pred, target, m):
    err = (pred - target) ** 2 * m
    return err.sum() / m.sum()


def full_loss(pred, target):
    return ((pred - target) ** 2).mean()


print()
perfect = target.copy()
print("loss of the exact target, masked / full:", masked_loss(perfect, target, m), full_loss(perfect, target))

# A model that is right on the masked frames and wrong on the context frames.
wrong_context = target.copy()
wrong_context[m == 0] += 3.0
print("wrong only on context frames, masked loss:", masked_loss(wrong_context, target, m))
print("wrong only on context frames, full loss  :", full_loss(wrong_context, target))

zero = np.zeros(6)
print("zero prediction, masked loss:", masked_loss(zero, target, m))
print("masked frames:", int(m.sum()), "of", len(m), "; masked fraction:", m.mean())
