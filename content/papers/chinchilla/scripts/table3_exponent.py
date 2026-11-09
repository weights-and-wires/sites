"""Fit N_opt proportional to C^a to the rows of Table 3 (Approach 1) and
Table A3 (Approach 2). Also fits the tokens exponent b.

Run: python3 scripts/table3_exponent.py
"""
import numpy as np

N = np.array([400e6, 1e9, 10e9, 67e9, 175e9, 280e9, 520e9, 1e12, 10e12])
C1 = np.array([1.92e19, 1.21e20, 1.23e22, 5.76e23, 3.85e24, 9.90e24, 3.43e25, 1.27e26, 1.30e28])
D1 = np.array([8.0e9, 20.2e9, 205.1e9, 1.5e12, 3.7e12, 5.9e12, 11.0e12, 21.2e12, 216.2e12])
C2 = np.array([1.84e19, 1.20e20, 1.32e22, 6.88e23, 4.54e24, 1.18e25, 4.19e25, 1.59e26, 1.75e28])
D2 = np.array([7.7e9, 20.0e9, 219.5e9, 1.7e12, 4.3e12, 7.1e12, 13.4e12, 26.5e12, 292.0e12])

for name, c, d in [("Table 3 (Approach 1)", C1, D1), ("Table A3 (Approach 2)", C2, D2)]:
    a = np.polyfit(np.log(c), np.log(N), 1)[0]
    b = np.polyfit(np.log(c), np.log(d), 1)[0]
    print(f"{name}: slope of log N_opt vs log C = {a:.4f}; slope of log D_opt vs log C = {b:.4f}; sum = {a+b:.4f}")
    print(f"   tokens per parameter: min {np.min(d/N):.1f}, max {np.max(d/N):.1f}, at 1B {d[1]/N[1]:.1f}")

# Kaplan's exponents applied to the same tenfold step.
print(f"Kaplan 0.73/0.27: tokens per parameter after 1000x compute x{1000**(0.27-0.73):.4f}")
