"""Numbers for temperature and top-k sampling (Section 3.6 uses top-k with k = 2).

Run: python3 scripts/sampling_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


logits = np.array([2.0, 1.0, 0.5, -1.0])
print("logits               =", logits)
print("softmax (T = 1)      =", softmax(logits))
print("softmax (T = 0.5)    =", softmax(logits / 0.5))
print("softmax (T = 2)      =", softmax(logits / 2))


def top_k_probs(logits, k, temperature=1.0):
    z = logits / temperature
    keep = np.argsort(z)[-k:]
    masked = np.full_like(z, -np.inf)
    masked[keep] = z[keep]
    return softmax(masked)


print("top-k k=2 (T = 1)    =", top_k_probs(logits, 2))
print("top-k k=1            =", top_k_probs(logits, 1))
print("top-k k=2 (T = 0.5)  =", top_k_probs(logits, 2, 0.5))
# Renormalizing the original probabilities over the kept tokens gives the same answer.
p = softmax(logits)
print("p[:2] / p[:2].sum()  =", p[:2] / p[:2].sum())

# Entropy falls as temperature falls.
for T in (0.5, 1, 2):
    q = softmax(logits / T)
    print(f"entropy at T = {T}: {-(q * np.log(q)).sum():.4f} nats")
