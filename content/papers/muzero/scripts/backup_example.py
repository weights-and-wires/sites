#!/usr/bin/env python3
"""Backup with rewards and a discount (Appendix B, Equations 3 and 4)."""

import numpy as np

gamma = 0.9
# A simulation path s0 -a1-> s1 -a2-> s2 -a3-> s3 with l = 3 edges.
r = {1: 1.0, 2: 0.0, 3: 2.0}  # r_k = R(s_{k-1}, a_k), predicted by the dynamics function
v_l = 5.0  # value predicted at the leaf s3
l = 3

# Equation 3: G_k = sum_{tau=0}^{l-1-k} gamma^tau r_{k+1+tau} + gamma^(l-k) v_l, for k = l..1.
G = {}
for k in range(l, 0, -1):
    G[k] = sum(gamma ** tau * r[k + 1 + tau] for tau in range(l - k)) + gamma ** (l - k) * v_l
for k in range(1, l + 1):
    print(f"G_{k} = {G[k]:.4f}")

# Equation 4: Q := (N * Q + G) / (N + 1) on each edge, N := N + 1.
N = {1: 4.0, 2: 1.0, 3: 0.0}
Q = {1: 3.0, 2: 6.0, 3: 0.0}
print("before:", {k: (N[k], Q[k]) for k in N})
for k in range(1, l + 1):
    Q[k] = (N[k] * Q[k] + G[k]) / (N[k] + 1)
    N[k] += 1
print("after: ", {k: (N[k], round(Q[k], 4)) for k in N})

# The same G_k as a recursion: G_k = r_{k+1} + gamma * G_{k+1}, with G_l = v_l.
check = {l: v_l}
for k in range(l - 1, 0, -1):
    check[k] = r[k + 1] + gamma * check[k + 1]
print("recursion matches:", all(abs(check[k] - G[k]) < 1e-12 for k in range(1, l)))
