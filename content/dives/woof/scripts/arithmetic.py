#!/usr/bin/env python3
"""Our own arithmetic on the numbers the post reports (post: underdog.ai/woof)."""
import math

# Word counts the post reports: (errors, words)
scores = {
    "hard, one voice": (90, 4377),
    "hard, two voices, Mira": (62, 4379),
    "hard, two voices, Juno": (67, 4371),
    "everyday (LibriTTS-R, 200 sentences)": (66, 3217),
    "expressive (30 lines)": (3, 329),
}
for name, (e, n) in scores.items():
    print(f"{name}: {e}/{n} = {100 * e / n:.2f}%")

print(f"\n25% claim: 1 - 1.42/1.89 = {100 * (1 - 1.42 / 1.89):.1f}%")
print(f"25% claim from raw counts: 1 - (62/4379)/1.89% = {100 * (1 - (62 / 4379) / 0.0189):.1f}%")

print("\nSize ratios (decimal units, 41.5 GB = 41,500 MB)")
print(f"41.5 GB / 40.6 MB = {41500 / 40.6:.0f}x   (post: 1,022x)")
print(f"41.5 GB / 31.7 MB = {41500 / 31.7:.0f}x   (post: 1,308x)")
print(f"35.9B / 30.7M params = {35.9e9 / 30.7e6:.0f}x   (post: 1,169x)")
print(f"35.9B / 38.2M params = {35.9e9 / 38.2e6:.0f}x   (post: 939x)")

print("\nParams x bytes")
for name, p, reported in (("one voice", 30.7e6, 31.7), ("two voices", 38.2e6, 40.6)):
    print(f"{name}: int8 {p / 1e6:.1f} MB, fp16 {2 * p / 1e6:.1f} MB, fp32 {4 * p / 1e6:.1f} MB; "
          f"reported {reported} MB, gap to int8 {reported - p / 1e6:.1f} MB")
print(f"Drama 3 at 8 bits: {35.9e9 / 1e9:.1f} GB; reported 41.5 GB, gap {41.5 - 35.9:.1f} GB")

print("\nSpeed")
print(f"13.2 s / 0.73 s = {13.2 / 0.73:.1f}x real time (post: 18x)")
print(f"2.3 s / 0.12 s = {2.3 / 0.12:.1f}x real time")
for steps, rtf in ((16, 18), (8, 24), (4, 29)):
    print(f"--steps {steps}: {rtf}x real time -> {13.2 / rtf:.2f} s for 13.2 s of speech")

print("\nWER interval (binomial approximation, treats words as independent)")
for e, n in ((62, 4379), (90, 4377)):
    p = e / n
    se = math.sqrt(p * (1 - p) / n)
    print(f"{e}/{n}: {100 * p:.2f}% +/- {196 * se:.2f} points (95%)")
