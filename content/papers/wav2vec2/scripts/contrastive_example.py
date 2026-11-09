"""Compute the contrastive-loss, temperature, diversity-loss, and codebook numbers.

Run: python3 scripts/contrastive_example.py
Equation 3 (contrastive loss), Equation 4 (diversity loss), Section 4.2 (G, V, kappa).
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


c = np.array([2.0, 1.0])
q_true = np.array([1.0, 1.0])
q_d = [np.array([1.0, -1.0]), np.array([-1.0, 0.5])]
cands = [q_true] + q_d
sims = np.array([cos(c, q) for q in cands])
print("cosine sims          =", sims)
for kappa in (1.0, 0.1):
    logits = sims / kappa
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    loss = -np.log(probs[0])
    print(f"kappa={kappa}: logits={logits} probs={probs} loss={loss:.4f}")
# Scale invariance of cosine versus a raw dot product.
big = 10 * q_true
print("cos with 10 * q_true =", round(float(cos(c, big)), 4), "dot =", c @ big, "vs dot", c @ q_true)

# Chance level for K = 100 distractors.
print("log(K+1), K=100      =", round(float(np.log(101)), 4))

# Diversity loss, Equation 4, with G = 1 and V = 4.
V = 4
for name, p in [("uniform", np.full(V, 0.25)), ("skewed", np.array([0.7, 0.1, 0.1, 0.1])), ("collapsed", np.array([0.97, 0.01, 0.01, 0.01]))]:
    ent = -(p * np.log(p)).sum()
    Ld = (p * np.log(p)).sum() / (1 * V)
    print(f"{name:9s} p={p} H={ent:.4f} Ld={Ld:.4f}")

# Paper-sized codebook.
G, V = 2, 320
print("codewords G^V        =", V ** G)
print("Ld uniform, V=320    =", round(float(-np.log(V) / V), 5))
# Entropy of the average is not the average of entropies.
batch = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
avg = batch.mean(axis=0)
H = lambda p: -sum(x * np.log(x) for x in p if x > 0)
print("entropy of average   =", round(float(H(avg)), 4), "average of entropies =", round(float(np.mean([H(r) for r in batch])), 4))
