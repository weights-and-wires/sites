"""Spectral reconstruction loss (Equation 1), commitment loss (Equation 3),
and relative feature matching (Equation 2) on small examples.

Run: python3 scripts/losses_example.py
"""

import numpy as np

fs = 24000
t = np.arange(fs // 4) / fs
x = np.sin(2 * np.pi * 440 * t)


def stft_mag(x, n_fft):
    hop = n_fft // 4
    win = np.hanning(n_fft)
    n = 1 + (len(x) - n_fft) // hop
    frames = np.stack([x[i * hop : i * hop + n_fft] * win for i in range(n)])
    return np.abs(np.fft.rfft(frames, axis=-1)) / np.sqrt((win**2).sum())


SCALES = [2**i for i in range(5, 12)]  # e = 5..11 in Equation 1


def spectral_loss(x, xh):
    total = 0.0
    for s in SCALES:
        d = stft_mag(x, s) - stft_mag(xh, s)
        total += np.abs(d).sum() + np.linalg.norm(d)  # alpha_i = 1
    return total / len(SCALES)


print(f"{len(SCALES)} scales, windows {SCALES}")
print(f"identical signals: {spectral_loss(x, x):.6f}")
for gain in (0.9, 0.5):
    ref = spectral_loss(x, np.zeros_like(x))
    print(f"x_hat = {gain} x: loss = {spectral_loss(x, gain * x):.4f}, equals {1 - gain:.1f} * loss(x, 0) = {(1 - gain) * ref:.4f}")
print(f"x_hat = -x: spectral loss = {spectral_loss(x, -x):.6f}, time L1 = {np.abs(x - (-x)).sum():.1f}")

# Commitment loss, Equation 3, on the toy RVQ of the SoundStream page.
C1 = np.array([[0, 0], [4, 0], [0, 4], [4, 4]], float)
C2 = np.array([[0, 0], [-1, 1], [1, -1], [1, 1]], float)
C3 = 0.25 * C2
z = np.array([3.2, 0.9])
lw = 0.0
for c in (C1, C2, C3):
    q = c[np.argmin(((c - z) ** 2).sum(1))]
    lw += ((z - q) ** 2).sum()
    z = z - q
print(f"\ncommitment loss with 3 stages = {lw:.4f}")

# Relative feature matching, Equation 2, for one discriminator layer.
a = np.array([1.0, -2.0, 3.0, 0.0])  # D^l(x)
b = np.array([1.5, -1.0, 2.0, 0.5])  # D^l(x_hat)
num = np.abs(a - b).mean()
den = np.abs(a).mean()
print(f"\nfeature matching: mean|a-b| = {num:.4f}, mean|a| = {den:.4f}, relative = {num / den:.4f}")
print(f"same layer with both features scaled by 10: {np.abs(10*a - 10*b).mean() / np.abs(10*a).mean():.4f}")
