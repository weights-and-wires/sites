"""Compute the numbers that compare classic rejection sampling with speculative sampling (Appendix A.2).

Run: python3 scripts/rejection_vs_speculative.py
"""

import numpy as np

p = np.array([0.5, 0.4, 0.1])
q = np.array([0.2, 0.3, 0.5])

M = float(np.max(p / q))
classic = float(np.sum(q * p / (M * q)))  # E_{x~q}[p / (M q)] = 1 / M
speculative = float(np.minimum(p, q).sum())
print("p / q                    :", np.round(p / q, 4))
print("M = max p / q            :", round(M, 4))
print("classic accept rate 1/M  :", round(classic, 4))
print("speculative accept rate  :", round(speculative, 4))
