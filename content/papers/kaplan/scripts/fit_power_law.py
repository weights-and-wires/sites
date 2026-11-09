"""Fit a power law to synthetic loss curves in log space and extrapolate 10x.

The data follow Equation 1.1 of Kaplan et al. with the paper's constants
(alpha_N = 0.076, N_c = 8.8e13) and 1% multiplicative noise.

Run: python3 scripts/fit_power_law.py
"""

import numpy as np

ALPHA_N, N_C = 0.076, 8.8e13

rng = np.random.default_rng(0)
N = 10.0 ** np.arange(4, 9.5, 0.5)  # 1e4 ... 1e9 non-embedding parameters
L = (N_C / N) ** ALPHA_N * np.exp(rng.normal(0, 0.01, N.size))

# Fit on the eight smallest models (1e4 ... 10**7.5). Hold out the rest.
train = N <= 10**7.5
lx, ly = np.log(N[train]), np.log(L[train])
slope, intercept = np.polyfit(lx, ly, 1)
alpha = -slope
n_c = np.exp(intercept / alpha)
print(f"models used for the fit = {train.sum()}")
print(f"fitted alpha_N          = {alpha:.4f}   (true {ALPHA_N})")
print(f"fitted N_c              = {n_c:.3e}   (true {N_C:.1e})")

# One-sigma uncertainty of a NEW observation at a point x: the fitted line's
# own uncertainty plus the scatter of single measurements (the "1 +" term).
n = lx.size
resid = ly - (slope * lx + intercept)
s2 = resid @ resid / (n - 2)
sxx = ((lx - lx.mean()) ** 2).sum()
print(f"residual sigma (log)    = {np.sqrt(s2):.4f}")

print()
print(f"{'N':>10} {'observed':>9} {'predicted':>9} {'1-sigma':>8} {'observed/pred':>14}")
for i in np.where(~train)[0]:
    x = np.log(N[i])
    pred = np.exp(slope * x + intercept)
    sd = np.sqrt(s2 * (1 + 1 / n + (x - lx.mean()) ** 2 / sxx))
    print(f"{N[i]:10.2e} {L[i]:9.4f} {pred:9.4f} {sd:8.4f} {L[i] / pred:14.4f}")

# Fitting a line to log L against N (not log N) is a different model.
semi_slope, semi_int = np.polyfit(N[train], np.log(L[train]), 1)
print()
print(f"semilog fit at N=1e9    = {np.exp(semi_slope * 1e9 + semi_int):.4f}  (observed {L[-1]:.4f})")
