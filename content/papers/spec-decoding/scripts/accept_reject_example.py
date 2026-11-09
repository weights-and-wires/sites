"""Compute the numbers for speculative sampling (Section 2.3, Appendix A.1).

Run: python3 scripts/accept_reject_example.py
"""

import numpy as np

p = np.array([0.5, 0.4, 0.1])  # target distribution
q = np.array([0.2, 0.3, 0.5])  # draft distribution

accept_prob = np.minimum(1.0, p / q)  # acceptance probability for each drafted token
beta = float(np.sum(q * accept_prob))  # Theorem 3.5: beta = sum_x min(p, q)
residual = np.maximum(0.0, p - q)
print("p                       :", p)
print("q                       :", q)
print("accept prob min(1, p/q) :", np.round(accept_prob, 4))
print("sum_x min(p, q)         :", round(float(np.minimum(p, q).sum()), 4))
print("beta = E_q[min(1, p/q)] :", round(beta, 4))
print("max(0, p - q)           :", np.round(residual, 4), "sums to", round(float(residual.sum()), 4), "= 1 - beta")
p_adj = residual / residual.sum()
print("p' = norm(max(0, p - q)):", np.round(p_adj, 4))

accepted = q * accept_prob  # P(guess accepted, x = x')
rejected = (1 - beta) * p_adj  # P(guess rejected, x = x')
print("P(accepted, x)          :", np.round(accepted, 4))
print("P(rejected, x)          :", np.round(rejected, 4))
print("P(x)                    :", np.round(accepted + rejected, 4))
assert np.allclose(accepted + rejected, p)

rng = np.random.default_rng(0)
n = 100_000
counts = np.zeros(3)
n_accepted = 0
for _ in range(n):
    x = rng.choice(3, p=q)
    if rng.random() <= min(1.0, p[x] / q[x]):
        n_accepted += 1
    else:
        x = rng.choice(3, p=p_adj)
    counts[x] += 1
expected = p * n
chi2 = float(np.sum((counts - expected) ** 2 / expected))
print()
print(f"{n} seeded samples")
print("empirical frequencies   :", np.round(counts / n, 4))
print("empirical accept rate   :", round(n_accepted / n, 4))
print("chi-square, 2 df        :", round(chi2, 2), "(the 0.001 critical value is 13.82)")

# Without the residual: reject, then resample from p itself.
no_resid = accepted + (1 - beta) * p
print()
print("accept, else resample p :", np.round(no_resid, 4), "(wrong: token 0 is under-sampled)")
