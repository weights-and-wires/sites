"""Build the log power spectrogram of a sinusoid (Section 3.1: the network input is a spectrogram).

The paper's text does not give the window or hop, so this script uses 20 ms and 10 ms,
a common choice for speech.

Run: python3 scripts/spectrogram_example.py
"""

import numpy as np

sr, win_ms, hop_ms, eps = 16000, 20, 10, 1e-10
win, hop = sr * win_ms // 1000, sr * hop_ms // 1000
n = np.arange(sr)  # one second
x = np.sin(2 * np.pi * 1000 * n / sr)

n_frames = 1 + (len(x) - win) // hop
window = np.hanning(win)
frames = np.stack([x[i * hop : i * hop + win] * window for i in range(n_frames)])
spec = np.log(np.abs(np.fft.rfft(frames, axis=1)) ** 2 + eps)

print(f"window = {win} samples, hop = {hop} samples")
print(f"frames = {n_frames}, bins = {spec.shape[1]}, shape = {spec.shape}")
print(f"bin spacing = {sr / win:.1f} Hz, so 1000 Hz is bin {1000 / (sr / win):.0f}")
print(f"peak bin in every frame: {sorted(set(spec.argmax(axis=1).tolist()))}")
mid = spec[50]
print(f"log power at bins 19, 20, 21: {mid[19]:.3f}, {mid[20]:.3f}, {mid[21]:.3f}")
print(f"log(P[19] / P[20]) = {mid[19] - mid[20]:.3f}   (Hann: amplitude ratio 0.5, so log(0.25) = {np.log(0.25):.3f})")
print(f"log(P[30] / P[20]) = {mid[30] - mid[20]:.3f}")

rect = np.log(np.abs(np.fft.rfft(np.stack([x[i * hop : i * hop + win] for i in range(n_frames)]), axis=1)) ** 2 + eps)
print(f"rectangular window: log(P[19] / P[20]) = {rect[50, 19] - rect[50, 20]:.3f}")
