"""Compute the decay quantity from Section 3.4.3 (Equation 37 and Figure 2).

The paper bounds |sum_i q_i k_i* e^{i(m-n)theta_i}| by (max_i |h_{i+1} - h_i|) * sum_i |S_{i+1}|
and says the mean of |S_j| shrinks as m - n grows. This script computes that mean,
with S_j = sum_{i<j} e^{i(m-n)theta_i}, for d = 256 and theta_i = 10000^(-2i/d).

Run: python3 scripts/long_term_decay.py
"""

import numpy as np

d = 256
theta = 10000.0 ** (-2.0 * np.arange(d // 2) / d)

print("distance   mean |S_j|, j=1..d/2")
for dist in [0, 1, 2, 5, 10, 25, 50, 100, 150, 200, 250, 500, 1000]:
    S = np.cumsum(np.exp(1j * dist * theta))
    print(f"{dist:8d}   {np.abs(S).mean():8.3f}")

print()
print("The mean is not monotone at large distances (it oscillates):")
vals = [np.abs(np.cumsum(np.exp(1j * dist * theta))).mean() for dist in range(1, 2001)]
print("min over 1..2000 at distance", 1 + int(np.argmin(vals)), "=", round(min(vals), 3))
print("max over 1000..2000:", round(max(vals[999:]), 3))
