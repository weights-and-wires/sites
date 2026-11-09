"""Compute the routing-combination counts and expert budgets for DeepSeekMoE.

Run: python3 scripts/combinatorics.py
"""

from fractions import Fraction
from math import comb

print("Section 3.1: combinations of activated experts, N = 16")
print("  top-2 of 16 (m = 1)          :", comb(16, 2))
print("  top-8 of 64 (m = 4, Eq. 7)   :", f"{comb(64, 8):,}")
print("  ratio                        :", f"{comb(64, 8) / comb(16, 2):,.0f}")
print("  with 1 shared expert (Eq. 10): top-7 of 63 =", f"{comb(63, 7):,}")
print()

# Table 7 configurations: (name, shared, routed, activated routed, expert size in FFNs)
configs = [
    ("2.0B validation", 1, 63, 7, Fraction(1, 4)),
    ("16.4B", 2, 64, 6, Fraction(1, 4)),
    ("144.6B", 4, 128, 12, Fraction(1, 8)),
]
print("Table 7 configurations, in units of one standard FFN")
for name, ks, routed, active, size in configs:
    total = (ks + routed) * size
    act = (ks + active) * size
    print(f"  {name:16s} experts {ks}+{routed}, active {ks}+{active}: "
          f"total {float(total):5.2f} FFNs, active {float(act):4.2f} FFNs, "
          f"routing choices C({routed},{active}) = {comb(routed, active):,}")
