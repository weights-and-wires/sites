import numpy as np


def log_power_spectrogram(x, sr, win_ms=20, hop_ms=10, eps=1e-10):
    """Return the log power spectrogram of the 1-D signal `x`, shape (n_frames, win // 2 + 1)."""
    x = np.asarray(x, dtype=float)
    win = int(round(sr * win_ms / 1000))
    hop = int(round(sr * hop_ms / 1000))
    n_frames = 1 + (len(x) - win) // hop
    window = np.hanning(win)
    frames = np.stack([x[i * hop : i * hop + win] * window for i in range(n_frames)])
    spec = np.fft.rfft(frames, axis=1)
    power = np.abs(spec) ** 2
    return np.log(power + eps)
