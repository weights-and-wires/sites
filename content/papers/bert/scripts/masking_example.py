"""The 15 percent masking rule with 80/10/10 replacement (Section 3.1,
Appendix A.1), checked on a large seeded sequence and on the paper's
"my dog is hairy" example.

Run: python3 scripts/masking_example.py
"""

import numpy as np

PAD, CLS, SEP, MASK = 0, 1, 2, 3
N_SPECIAL = 4
VOCAB = 30000


def mask_tokens(ids, vocab_size, mask_id, rng, mlm_prob=0.15, n_special=N_SPECIAL):
    ids = np.asarray(ids)
    selected = (rng.random(ids.shape) < mlm_prob) & (ids >= n_special)
    u = rng.random(ids.shape)
    random_ids = rng.integers(n_special, vocab_size, ids.shape)
    inputs = ids.copy()
    inputs[selected & (u < 0.8)] = mask_id
    swap = selected & (u >= 0.8) & (u < 0.9)
    inputs[swap] = random_ids[swap]
    labels = np.where(selected, ids, -100)
    return inputs, labels, selected, u


rng = np.random.default_rng(0)
n = 1_000_000
ids = rng.integers(N_SPECIAL, VOCAB, n)
inputs, labels, selected, u = mask_tokens(ids, VOCAB, MASK, rng)
k = int(selected.sum())
is_mask = selected & (inputs == MASK)
is_kept = selected & (u >= 0.9)
is_rand = selected & (u >= 0.8) & (u < 0.9)
print(f"tokens                : {n}")
print(f"selected              : {k} ({k / n:.4f} of all tokens)")
print(f"  replaced by [MASK]  : {is_mask.sum() / k:.4f} of selected")
print(f"  replaced at random  : {is_rand.sum() / k:.4f} of selected ({is_rand.sum() / n:.4f} of all tokens)")
print(f"  left unchanged      : {is_kept.sum() / k:.4f} of selected ({is_kept.sum() / n:.4f} of all tokens)")
print(f"label positions == selected positions: {bool(np.array_equal(labels != -100, selected))}")
print(f"tokens that differ from the original: {(inputs != ids).sum() / n:.4f} of all tokens")

# The paper's example: the 4th token of "my dog is hairy" is chosen.
words = ["my", "dog", "is", "hairy"]
print("\nThe paper's example, with the 4th token chosen:")
print(f"  80%: {' '.join(words[:3] + ['[MASK]'])}")
print(f"  10%: {' '.join(words[:3] + ['apple'])}")
print(f"  10%: {' '.join(words)}")
print("  the target at that position is always 'hairy'")

# Short sequence, as the exercise uses.
rng = np.random.default_rng(1)
short = np.array([CLS, 11, 12, 13, 14, 15, 16, 17, 18, 19, SEP])
inp, lab, sel, _ = mask_tokens(short, VOCAB, MASK, rng, mlm_prob=0.3)
print("\nshort sequence, mlm_prob=0.3, seed 1")
print(f"  ids   : {short.tolist()}")
print(f"  inputs: {inp.tolist()}")
print(f"  labels: {lab.tolist()}")
