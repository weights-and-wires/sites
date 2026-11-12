"""Expected accepted length for a chain and for a draft tree, and the
paper's Table 5 gains.

Model (this page's assumption, not the paper's): node i is tested only if its
parent was accepted and its earlier siblings were rejected, and then it is
accepted with probability accept[i]. The target adds one more token per round.

Run: python3 scripts/expected_length.py
"""

import numpy as np


def expected_length(parents, accept):
    n = len(parents)
    reach = np.zeros(n)  # P(node accepted)
    remaining = {}  # parent -> P(parent accepted and every earlier child rejected)
    total = 0.0
    for i in range(n):
        p = parents[i]
        if p not in remaining:
            remaining[p] = 1.0 if p < 0 else reach[p]
        reach[i] = remaining[p] * accept[i]
        remaining[p] *= 1.0 - accept[i]
        total += reach[i]
    return 1.0 + total


alpha, d = 0.7, 5
chain = list(range(-1, d - 1))
print(f"chain, depth {d}, alpha {alpha}: {expected_length(chain, [alpha] * d):.4f}"
      f"  (closed form {(1 - alpha ** (d + 1)) / (1 - alpha):.4f})")

# Tree: every node on the main chain gets a second child that is accepted with
# probability 0.5 when the first child was rejected. Second children are leaves.
parents, accept = [], []
spine = -1
for level in range(d):
    parents.append(spine); accept.append(alpha)          # first child
    parents.append(spine); accept.append(0.5)            # sibling
    spine = len(parents) - 2
print(f"tree, depth {d}, {len(parents)} nodes: {expected_length(parents, accept):.4f}")

# One sibling at the root only (a 6-node tree).
parents2 = [-1, -1, 0, 2, 3, 4]
accept2 = [alpha, 0.5, alpha, alpha, alpha, alpha]
print(f"tree, root has two children, {len(parents2)} nodes: {expected_length(parents2, accept2):.4f}")

# Monte Carlo check of the 10-node tree.
rng = np.random.default_rng(0)
trials = 200_000
tot = 0
for _ in range(trials):
    node = -1
    length = 1
    while True:
        kids = [i for i, p in enumerate(parents) if p == node]
        nxt = None
        for k in kids:
            if rng.random() < accept[k]:
                nxt = k
                break
        if nxt is None:
            break
        length += 1
        node = nxt
    tot += length
print(f"simulated, 10-node tree: {tot / trials:.4f}")

# Cost: d draft passes, relative cost c each, ignoring the larger verify pass.
c = 0.05
for name, tau in (("chain", expected_length(chain, [alpha] * d)), ("tree", expected_length(parents, accept))):
    print(f"{name}: tau {tau:.4f}, speedup tau / (1 + {d} c) = {tau / (1 + d * c):.4f} at c = {c}")

# Table 5 of the paper (MT-bench, temperature 0): tau for a chain and a tree.
table5 = {
    "Vicuna 7B": (3.20, 3.94),
    "Vicuna 13B": (3.23, 3.98),
    "Vicuna 33B": (2.97, 3.68),
    "LLaMA2-Chat 7B": (3.00, 3.62),
    "LLaMA2-Chat 13B": (3.18, 3.90),
    "LLaMA2-Chat 70B": (3.12, 3.81),
}
gains = []
for name, (chain_tau, tree_tau) in table5.items():
    gains.append(tree_tau - chain_tau)
    print(f"Table 5, {name}: {chain_tau:.2f} -> {tree_tau:.2f}, gain {tree_tau - chain_tau:.2f}")
print(f"gain range {min(gains):.2f} to {max(gains):.2f}")
