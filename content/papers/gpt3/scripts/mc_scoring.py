"""Multiple-choice scoring rules from Section 2.4.

For each completion the paper compares the model's likelihood. Most tasks use
the per-token likelihood. ARC, OpenBookQA, and RACE use the ratio
P(completion | context) / P(completion | answer context). The log-probabilities
below are invented for illustration; they are not from the paper.

Run: python3 scripts/mc_scoring.py
"""

import numpy as np

# Question: "Q: What is the capital of France?\nA:". Per-token log-probabilities.
candidates = {
    " Paris": {"ctx": [-1.5], "unc": [-6.0]},
    " a large city": {"ctx": [-0.5, -0.6, -0.5, -0.4], "unc": [-0.4, -0.5, -0.5, -0.4]},
    " Lyon": {"ctx": [-2.5], "unc": [-6.5]},
}

print(f"{'completion':<16}{'tokens':>7}{'sum ctx':>9}{'mean ctx':>10}{'sum unc':>9}{'log ratio':>11}")
rows = {}
for name, c in candidates.items():
    ctx, unc = np.array(c["ctx"]), np.array(c["unc"])
    rows[name] = (ctx.sum(), ctx.mean(), ctx.sum() - unc.sum())
    print(f"{name!r:<16}{len(ctx):>7}{ctx.sum():>9.2f}{ctx.mean():>10.3f}{unc.sum():>9.2f}{ctx.sum() - unc.sum():>11.2f}")

for label, idx in (("total log-likelihood", 0), ("per-token log-likelihood", 1), ("log ratio to answer context", 2)):
    best = max(rows, key=lambda n: rows[n][idx])
    print(f"{label:<28} picks {best!r}")
