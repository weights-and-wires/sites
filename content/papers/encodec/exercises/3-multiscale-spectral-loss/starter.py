import numpy as np


def stft_mag(x, n_fft):
    """Magnitude STFT, Hann window, hop n_fft // 4. Shape (frames, n_fft // 2 + 1)."""
    hop = n_fft // 4
    win = np.hanning(n_fft)
    n = 1 + (len(x) - n_fft) // hop
    frames = np.stack([x[i * hop : i * hop + n_fft] * win for i in range(n)])
    return np.abs(np.fft.rfft(frames, axis=-1)) / np.sqrt((win**2).sum())


def spectral_loss(x, x_hat, exponents=range(5, 12)):
    """Equation 1 with alpha_i = 1 and magnitude STFTs instead of mel-spectrograms."""
    raise NotImplementedError


def time_l1(x, x_hat):
    """The time-domain term: the L1 norm of x - x_hat."""
    raise NotImplementedError
