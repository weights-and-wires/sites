#!/usr/bin/env python3
"""Arithmetic on the numbers that the paper's text and tables give."""

import numpy as np

print("== Training volume (Section 4)")
print(f"board games: 1,000,000 x 2048 = {1_000_000 * 2048:,} sampled sequences")
print(f"Atari:       1,000,000 x 1024 = {1_000_000 * 1024:,} sampled sequences")

print("\n== Atari replay buffer (Appendix D)")
print(f"125,000 sequences x 200 moves = {125_000 * 200:,} moves")

print("\n== Visit-count temperature (Equation 6, Appendix D)")
N = np.array([30.0, 15.0, 5.0])
for T in [1.0, 0.5, 0.25]:
    w = N ** (1.0 / T)
    print(f"T = {T}: {np.round(w / w.sum(), 4).tolist()}")
steps = [(500_000, 1.0), (250_000, 0.5), (250_000, 0.25)]
print("schedule covers", sum(s for s, _ in steps), "steps")

print("\n== Table 1")
print(f"MuZero median / R2D2 median: {2041.1 / 1920.6:.3f}")
print(f"MuZero Reanalyze median / LASER median: {731.1 / 431:.3f}")
print(f"MuZero Reanalyze median / Rainbow median: {731.1 / 231.1:.3f}")
print(f"frames: MuZero 20.0B vs R2D2 37.5B, ratio {20.0 / 37.5:.3f}")
print(f"MuZero training steps 1M vs R2D2 2.16M, ratio {1 / 2.16:.3f}")

print("\n== Support size (Appendix F)")
print("integers from -300 to 300:", len(range(-300, 301)))

print("\n== Gradient scaling (Appendix G), K = 5")
K = 5
print(f"each head loss is multiplied by 1/K = {1 / K}")

print("\n== Discount and bootstrap (Appendix C, G)")
print(f"gamma^10 = {0.997 ** 10:.4f} for the Atari discount 0.997")
print(f"effective horizon 1/(1 - gamma) = {1 / (1 - 0.997):.1f} steps")

print("\n== Compute per simulation (Appendix B)")
print("search makes at most 1 dynamics call and 1 prediction call per simulation")
print(f"800 simulations -> at most 800 calls of each per move; 50 -> at most 50")
