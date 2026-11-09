#!/usr/bin/env python3
"""Shapes and sizes of the log-Mel front end and the encoder stem (Section 2.2)."""

import numpy as np

from whisper_ref import HOP, N_FFT, N_MELS, N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, mel_filterbank

print(f"samples in 30 s at {SAMPLE_RATE} Hz: {N_SAMPLES}")
print(f"window {N_FFT} samples = {1000 * N_FFT / SAMPLE_RATE:.0f} ms, hop {HOP} samples = {1000 * HOP / SAMPLE_RATE:.0f} ms")
print(f"FFT bins: {N_FFT // 2 + 1}")
frames_centered = 1 + N_SAMPLES // HOP
print(f"centered STFT frames: {frames_centered}; after dropping the last: {frames_centered - 1}")
frames_valid = 1 + (N_SAMPLES - N_FFT) // HOP
print(f"uncentered frames: {frames_valid}")
conv2_len = (3000 + 2 * 1 - 3) // 2 + 1
print(f"conv1 (k=3, s=1, pad=1): 3000 -> {(3000 + 2 - 3) // 1 + 1}; conv2 (k=3, s=2, pad=1): 3000 -> {conv2_len}")
print(f"encoder positions: {conv2_len}, each {1000 * 2 * HOP / SAMPLE_RATE:.0f} ms")

filters = mel_filterbank()
print(f"filterbank shape: {filters.shape}")

t = np.arange(N_SAMPLES) / SAMPLE_RATE
tone = np.sin(2 * np.pi * 1000 * t)
spec = log_mel_spectrogram(tone, filters)
print(f"log-mel of a 1 kHz tone: shape {spec.shape}, max {spec.max():.3f}, min {spec.min():.3f}")
peak_band = int(np.argmax(spec[:, 1500]))
centers = np.argmax(filters, axis=1) * (SAMPLE_RATE / 2) / (N_FFT // 2)
print(f"loudest band at frame 1500: {peak_band} (filter peak near {centers[peak_band]:.0f} Hz)")
silence = log_mel_spectrogram(np.zeros(N_SAMPLES), filters)
print(f"silence: every value {silence.min():.3f}")
print(f"input values for the encoder per 30 s window: {N_MELS * 3000}")
