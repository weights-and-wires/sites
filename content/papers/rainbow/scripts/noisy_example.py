"""A noisy linear layer (Equation 4) with fixed noise, and the factorised noise trick.

Run: python3 scripts/noisy_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

x = np.array([1.0, 2.0])
W = np.array([[0.5, -0.5]])
b = np.array([0.1])
W_noisy = np.full((1, 2), 0.5)  # sigma_0 = 0.5 initialises the noisy stream
b_noisy = np.array([0.5])

# Factorised Gaussian noise: one noise value per input and per output,
# combined as eps_w = f(eps_out) f(eps_in)^T with f(x) = sign(x) sqrt(|x|).
f = lambda e: np.sign(e) * np.sqrt(np.abs(e))
eps_in = np.array([1.0, -4.0])
eps_out = np.array([4.0])
eps_w = np.outer(f(eps_out), f(eps_in))
eps_b = f(eps_out)
print("eps_w =", eps_w, " eps_b =", eps_b)

clean = b + W @ x
noise = b_noisy * eps_b + (W_noisy * eps_w) @ x
print("deterministic stream b + Wx          =", clean)
print("noisy stream                         =", noise)
print("output y                             =", clean + noise)
print("noise values used (factorised)       =", eps_in.size + eps_out.size)
print("noise values used (independent)      =", W.size + b.size)
n_in, n_out = 512, 512
print(f"for a {n_in} x {n_out} layer: factorised {n_in + n_out}, independent {n_in * n_out + n_out}")
