"""Compute the small worked numbers that the qwn page states.

These are our own toy examples, not figures from the post, except the memory
napkin, which re-derives the post's arithmetic.
"""
import numpy as np

# 1. Mask on a four-token distribution.
logits = np.array([2.0, 1.0, 0.5, 0.0])  # the, walk, a, dog
names = ["the", "walk", "a", "dog"]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


p = softmax(logits)
masked = logits.copy()
masked[0] = -np.inf
q = softmax(masked)
print("mask example")
for nme, a, b in zip(names, p, q):
    print(f"  {nme:5s} before {a:.3f}  after {b:.3f}")
print(f"  mass the mask removed from the 'the' option: {p[0]:.3f}")

# 2. Group advantage with the post's reward formula.
cr = np.array([(5, 5), (4, 4), (5, 3), (2, 2)])
rew = (cr[:, 0] - 1 + cr[:, 1] - 1) / 8
adv = (rew - rew.mean()) / rew.std()
print("group of four")
print("  rewards   ", rew.tolist(), "mean", rew.mean(), "std", round(float(rew.std()), 4))
print("  advantages", np.round(adv, 2).tolist())
mn = (cr.min(1) - 1) / 4
print("  min rewards", mn.tolist())
print("  min advantages", np.round((mn - mn.mean()) / mn.std(), 2).tolist())

# 3. LoRA parameters for one square matrix.
d, r = 2048, 16
print("lora", d * d, 2 * d * r, (d * d) // (2 * d * r))

# 4. Memory napkin, in decimal GB.
GB = 1e9
params, b, L, d, V, layers = 2e9, 4, 2048, 2048, 150_000, 28
weights = params * 2 / GB
grads = params * 2 / GB
adam = params * 4 * 2 / GB
acts = b * L * d * 2 * 12 * layers / GB
logit = b * L * V * 2 / GB
print(f"memory: weights {weights:.1f} grads {grads:.1f} adam {adam:.1f} "
      f"acts {acts:.1f} logits {logit:.2f} total {weights + grads + adam + acts + logit:.1f}")
ckpt = layers * b * L * d * 2 / GB
print(f"checkpoint inputs {ckpt:.2f} GB, one layer {b * L * d * 2 * 12 / GB:.2f} GB")
