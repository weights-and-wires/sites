import numpy as np
from submission import HOP, N_FFT, SAMPLE_RATE, log_mel_spectrogram, mel_filterbank

F = mel_filterbank()


def _tone(freq, seconds):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return np.sin(2 * np.pi * freq * t)


def test_thirty_seconds_gives_80_by_3000():
    got = log_mel_spectrogram(_tone(440, 30), F)
    assert got.shape == (80, 3000), f"expected (80, 3000) for 30 s of audio, got {got.shape}"


def test_frame_count_is_samples_over_hop():
    for seconds, frames in [(1, 100), (0.5, 50), (10, 1000)]:
        got = log_mel_spectrogram(_tone(300, seconds), F)
        assert got.shape == (80, frames), f"{seconds} s expected (80, {frames}), got {got.shape}"


def test_silence_is_a_constant():
    got = log_mel_spectrogram(np.zeros(SAMPLE_RATE), F)
    assert np.allclose(got, -1.5), f"silence floors at log10(1e-10) = -10, which scales to (-10 + 4) / 4 = -1.5; got min {got.min()}, max {got.max()}"


def test_dynamic_range_is_two_after_scaling():
    got = log_mel_spectrogram(_tone(1000, 2), F)
    assert np.isclose(got.max() - got.min(), 2.0), f"8 decades divided by 4 is 2.0; got range {got.max() - got.min()}"


def test_clamp_is_global_not_per_frame():
    loud = _tone(1000, 1)
    quiet = 1e-6 * _tone(1000, 1)  # 12 decades of power below the loud tone
    got = log_mel_spectrogram(np.concatenate([loud, quiet]), F)
    last = got[:, -5]
    assert np.allclose(last, got.min()), f"quiet frames sit below the global floor, so every band equals the global minimum {got.min():.3f}; got range {last.min():.3f} to {last.max():.3f}"


def test_tone_lands_in_the_right_band():
    got = log_mel_spectrogram(_tone(1000, 2), F)
    want = int(np.argmax(F[:, 1000 * N_FFT // SAMPLE_RATE]))
    peak = int(np.argmax(got[:, got.shape[1] // 2]))
    assert peak == want, f"a 1 kHz tone should peak in band {want}, got band {peak}"


def test_log10_of_power():
    # Scaling the audio by 10 multiplies the power by 100, which adds 2 to log10, then 2 / 4 = 0.5 after scaling.
    a = log_mel_spectrogram(_tone(1000, 1), F)
    b = log_mel_spectrogram(10 * _tone(1000, 1), F)
    assert np.isclose(b.max() - a.max(), 0.5), f"expected +0.5 for 10x amplitude (log10 of power), got {b.max() - a.max()}"


def test_known_values():
    rng = np.random.default_rng(0)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    x = np.sin(2 * np.pi * 440 * t) + 0.1 * rng.normal(size=SAMPLE_RATE)
    got = log_mel_spectrogram(x, F)
    for (i, j), want in {(3, 0): 1.122565, (20, 50): 0.661799, (3, 99): 0.775732}.items():
        assert abs(got[i, j] - want) < 5e-5, f"value at band {i}, frame {j} expected {want}, got {got[i, j]:.6f}"
    assert abs(got.mean() - 0.657571) < 5e-5, f"mean expected 0.657571, got {got.mean():.6f}"
