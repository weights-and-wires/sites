#!/usr/bin/env python3
"""Arithmetic on the training mix and on Table 6 (dataset scaling, Section 4.2)."""

import numpy as np

TOTAL, MULTI, X_EN = 680_000, 117_000, 125_000  # Section 1
print(f"English transcription hours (derived): {TOTAL - MULTI - X_EN}")
for name, h in [("English", TOTAL - MULTI - X_EN), ("multilingual", MULTI), ("X to en translation", X_EN)]:
    print(f"  {name:20s} {h:8d} h  {100 * h / TOTAL:5.1f}%")
print(f"hours per 30 s segment: {30 / 3600:.5f}; segments in 680k hours: {TOTAL * 3600 / 30:,.0f}")

# Table 6: hours, English WER, multilingual WER, X->en BLEU.
rows = np.array(
    [
        [3405, 30.5, 92.4, 0.2],
        [6811, 19.6, 72.7, 1.7],
        [13621, 14.4, 56.6, 7.9],
        [27243, 12.3, 45.0, 13.9],
        [54486, 10.9, 36.4, 19.2],
        [681070, 9.9, 29.2, 24.8],
    ]
)
hours = rows[:, 0]
print("\nfraction of the full set:", ", ".join(f"{100 * h / hours[-1]:.1f}%" for h in hours[:-1]))
print(f"last step multiplies the data by {hours[-1] / hours[-2]:.1f}x")
for label, col in [("English WER", 1), ("multilingual WER", 2)]:
    y = rows[:, col]
    drop = y[-2] - y[-1]
    print(f"{label}: 54,486 h -> 681,070 h changes {y[-2]} to {y[-1]} ({drop:.1f} points, {100 * drop / y[-2]:.1f}% relative)")

# Power-law fit WER ~ a * hours^(-b) for multilingual WER up to 54,486 hours (the paper's claim).
x, y = np.log(hours[:5]), np.log(rows[:5, 2])
slope, intercept = np.polyfit(x, y, 1)
print(f"\nmultilingual WER, power-law fit on the first five rows: WER ~ {np.exp(intercept):.0f} * hours^({slope:.3f})")
pred = np.exp(intercept + slope * np.log(hours[-1]))
print(f"extrapolated WER at 681,070 h: {pred:.1f}; reported: {rows[-1, 2]}")
print(f"each doubling of data multiplies WER by {2 ** slope:.3f}")
