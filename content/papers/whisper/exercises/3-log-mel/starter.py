import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400  # 25 ms at 16 kHz
HOP = 160  # 10 ms
N_MELS = 80


def _hz_to_mel(f):
    """Slaney mel scale: linear below 1 kHz, logarithmic above (librosa default)."""
    f = np.asarray(f, dtype=float)
    f_sp = 200.0 / 3
    mel = f / f_sp
    log_step = np.log(6.4) / 27.0
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / log_step, mel)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=float)
    f_sp = 200.0 / 3
    log_step = np.log(6.4) / 27.0
    return np.where(m >= 15.0, 1000.0 * np.exp(log_step * (m - 15.0)), f_sp * m)


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SAMPLE_RATE):
    """Triangular filters on the Slaney mel scale with area normalization; shape (n_mels, n_fft // 2 + 1)."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0.0, np.minimum(lower, upper))
    w *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w


def log_mel_spectrogram(audio, filters):
    """Return the (80, frames) log-Mel spectrogram of a 1-D float waveform sampled at 16 kHz.

    filters is the (80, 201) matrix from mel_filterbank(). Steps, as in the released code:
    reflect-pad N_FFT // 2 samples on both sides, take frames of N_FFT samples every HOP samples,
    multiply by a periodic Hann window, take the rFFT, use the squared magnitude, drop the last frame,
    apply the filters, take log10 (floor the mel energies at 1e-10), clamp to at most 8 decades below
    the maximum, and finish with (x + 4) / 4.
    """
    raise NotImplementedError
