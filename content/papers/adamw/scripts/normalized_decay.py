#!/usr/bin/env python3
"""Appendix B.1: normalized weight decay, lambda = lambda_norm * sqrt(b / (B * T)).

Run from anywhere:  python3 content/papers/adamw/scripts/normalized_decay.py
"""
import math


def lam(lam_norm, b, B, T):
    return lam_norm * math.sqrt(b / (B * T))


b = 128
cifar_B = 50_000          # CIFAR-10 training images
imnet_B = 1_281_167       # ImageNet32x32 training images (the paper says "1.2 million")
lam_norm = 0.025          # a value the paper reports as near-optimal (Appendix D)

print("lambda_norm = 0.025, batch size b = 128")
for T in (100, 400, 1800):
    n = cifar_B // b * T
    print(f"  CIFAR-10, T={T:<4} epochs: {n:>7} batch passes, lambda = {lam(lam_norm, b, cifar_B, T):.6f}")
print(f"  ImageNet32x32, T=100 epochs: lambda = {lam(lam_norm, b, imnet_B, 100):.6f}")
print(f"  ratio of raw lambda needed, CIFAR-10 over ImageNet32x32 at equal T: {math.sqrt(imnet_B / cifar_B):.2f}")
print(f"  ratio of lambda, T=100 over T=1800: {math.sqrt(1800 / 100):.2f}")
