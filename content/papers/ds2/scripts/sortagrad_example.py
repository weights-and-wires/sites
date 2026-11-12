"""Show why CTC cost grows with utterance length, and the SortaGrad batch order (Section 3.3, Equation 9).

Run: python3 scripts/sortagrad_example.py
"""

import numpy as np

# Part 1: one alignment with per-frame probability 0.9 costs -T * log(0.9).
print("Single path, per-frame probability 0.9")
for T in (10, 100, 1000):
    print(f"T = {T:<5} -log prod p = {-T * np.log(0.9):.3f}")

# Part 2: the batch order.
lengths = np.array([7, 3, 9, 1, 5, 8, 2, 6, 4, 10, 12, 11])
batch_size = 3
order = np.argsort(lengths, kind="stable")
batches = [order[i : i + batch_size].tolist() for i in range(0, len(order), batch_size)]


def longest(batch):
    return int(lengths[batch].max())


print()
print(f"utterance lengths = {lengths.tolist()}")
print("Minibatches (indices), formed by sorting on length:")
for b in batches:
    print(f"  {b}  lengths = {lengths[b].tolist()}  longest = {longest(b)}")

epoch0 = sorted(batches, key=longest)
print(f"epoch 0 order, by longest utterance: {[longest(b) for b in epoch0]}")
rng = np.random.default_rng(5)
for epoch in (1, 2):
    perm = rng.permutation(len(batches))
    print(f"epoch {epoch} order, shuffled:             {[longest(batches[i]) for i in perm]}")
