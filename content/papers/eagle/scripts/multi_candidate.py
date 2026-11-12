"""Verifying k candidates at one tree node (Section 3.3, Appendix A.2).

The k candidates are drawn i.i.d. from q. They are tested in order against
the current target distribution. After each rejection the target becomes
norm(max(0, p - q)). If every candidate is rejected, the token comes from the
last adjusted distribution.

Run: python3 scripts/multi_candidate.py
"""

import numpy as np

p = np.array([0.5, 0.4, 0.1])
q = np.array([0.2, 0.3, 0.5])


def residual(p, q):
    r = np.maximum(0.0, p - q)
    return r / r.sum()


def exact_output(p, q, k):
    """Exact output distribution and P(some candidate accepted)."""
    if k == 0:
        return p.copy(), 0.0
    accepted = np.minimum(p, q)  # q(x) * min(1, p/q)
    beta = accepted.sum()
    rest, rest_acc = exact_output(residual(p, q), q, k - 1)
    return accepted + (1 - beta) * rest, beta + (1 - beta) * rest_acc


for k in (1, 2, 3):
    out, acc = exact_output(p, q, k)
    print(f"k={k}: output {np.round(out, 4).tolist()}  P(accept some candidate) {acc:.4f}")

p2 = residual(p, q)
print("p after one rejection:", np.round(p2, 4).tolist(), " beta_2 =", round(float(np.minimum(p2, q).sum()), 4))


def verify_node(p, q, cands, u, rng):
    for x, ui in zip(cands, u):
        if ui <= p[x] / q[x]:
            return x, True
        p = residual(p, q)
    return int(rng.choice(len(p), p=p)), False


rng = np.random.default_rng(0)
trials = 200_000
counts = np.zeros(3)
hits = 0
for _ in range(trials):
    cands = rng.choice(3, size=2, p=q)
    t, ok = verify_node(p, q, cands, rng.random(2), rng)
    counts[t] += 1
    hits += ok
print("sampled k=2, frequencies:", np.round(counts / trials, 4).tolist(), " accept rate:", round(hits / trials, 4))
stat = float(np.sum((counts - p * trials) ** 2 / (p * trials)))
print(f"chi-square vs p: {stat:.2f} (2 d.o.f., 0.001 critical value 13.82)")

# The common bug: never update p between candidates.
def verify_no_update(p, q, cands, u, rng):
    p0 = p
    for x, ui in zip(cands, u):
        if ui <= p0[x] / q[x]:
            return x
    return int(rng.choice(len(p0), p=residual(p0, q)))

rng = np.random.default_rng(0)
bad = np.zeros(3)
for _ in range(trials):
    cands = rng.choice(3, size=2, p=q)
    bad[verify_no_update(p, q, cands, rng.random(2), rng)] += 1
print("no-update bug, frequencies:", np.round(bad / trials, 4).tolist())
