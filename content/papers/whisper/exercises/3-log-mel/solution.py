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
    """Whisper's front end, following whisper/audio.py: centered STFT, drop the last frame,
    power, mel, log10, clamp to 8 decades below the maximum, then (x + 4) / 4."""
    audio = np.asarray(audio, dtype=np.float64)
    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    window = np.hanning(N_FFT + 1)[:-1]  # periodic Hann
    spec = np.fft.rfft(padded[idx] * window, axis=-1)  # (frames, 201)
    power = (np.abs(spec) ** 2)[:-1]  # drop the last frame
    mel = filters @ power.T  # (80, frames)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0
