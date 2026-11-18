"""Grouped RMSNorm (Section 2.1, Equation 3) on a vector with one large feature."""
import numpy as np


def rmsnorm(x, eps=0.0):
    return x / np.sqrt(np.mean(x**2) + eps)


def grouped_rmsnorm(x, k, eps=0.0):
    g = x.reshape(k, -1)
    return (g / np.sqrt(np.mean(g**2, axis=1, keepdims=True) + eps)).reshape(-1)


x = np.array([8.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
np.set_printoptions(precision=4, suppress=True)
print("x            =", x)
print("RMSNorm      =", rmsnorm(x), " max", round(float(rmsnorm(x).max()), 4))
print("grouped k=2  =", grouped_rmsnorm(x, 2), " max", round(float(grouped_rmsnorm(x, 2).max()), 4))
print("bound sqrt(d)=", round(np.sqrt(8), 4), " bound sqrt(d/k)=", round(np.sqrt(4), 4))
# A one-hot vector reaches the bound.
e = np.zeros(8); e[0] = 1.0
print("one-hot: RMSNorm max", round(float(rmsnorm(e, 1e-6).max()), 4), " grouped k=2 max", round(float(grouped_rmsnorm(e, 2, 1e-6).max()), 4))
