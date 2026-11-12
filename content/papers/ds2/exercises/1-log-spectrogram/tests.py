import numpy as np
from submission import log_power_spectrogram


def sine(freq, sr=16000, seconds=1.0):
    n = np.arange(int(sr * seconds))
    return np.sin(2 * np.pi * freq * n / sr)


def test_shape_for_one_second_at_16_khz():
    got = log_power_spectrogram(sine(1000), 16000).shape
    assert got == (99, 161), f"expected (99, 161) = (1 + (16000 - 320) // 160, 320 // 2 + 1), got {got}"


def test_shape_with_other_window_and_hop():
    x = np.zeros(8000)
    got = log_power_spectrogram(x, 8000, win_ms=25, hop_ms=10).shape
    # 25 ms at 8 kHz is 200 samples and 10 ms is 80 samples.
    assert got == (98, 101), f"expected (98, 101), got {got}"


def test_sinusoid_peaks_in_the_right_bin_in_every_frame():
    spec = log_power_spectrogram(sine(1000), 16000)
    peaks = spec.argmax(axis=1)
    # Bin spacing is 16000 / 320 = 50 Hz, so 1000 Hz is bin 20.
    assert np.all(peaks == 20), f"expected bin 20 in every frame, got bins {np.unique(peaks)}"


def test_other_frequency_moves_the_peak():
    spec = log_power_spectrogram(sine(3000), 16000)
    peaks = np.unique(spec.argmax(axis=1))
    assert list(peaks) == [60], f"expected bin 60 for 3000 Hz, got {peaks}"


def test_hann_window_leaks_half_amplitude_into_the_neighbors():
    spec = log_power_spectrogram(sine(1000), 16000)
    mid = spec[50]
    # A Hann window spreads an on-bin sinusoid over three bins with amplitudes 0.5, 1, 0.5.
    # Power ratio 0.25, so the log ratio is about -1.386. A rectangular window leaks nothing.
    ratio = mid[19] - mid[20]
    assert -1.7 < ratio < -1.1, f"expected log(P[19]/P[20]) near -1.39 with a Hann window, got {ratio:.3f}"
    far = mid[30] - mid[20]
    assert far < -10, f"expected bin 30 at least 10 nats below the peak, got {far:.3f}"


def test_matches_a_direct_dft_of_one_frame():
    rng = np.random.default_rng(3)
    x = rng.normal(size=1000)
    sr, win, hop = 16000, 320, 160
    spec = log_power_spectrogram(x, sr)
    k = 3  # frame index
    frame = x[k * hop : k * hop + win] * np.hanning(win)
    n = np.arange(win)
    want = []
    for b in range(win // 2 + 1):
        c = np.sum(frame * np.exp(-2j * np.pi * b * n / win))
        want.append(np.log(abs(c) ** 2 + 1e-10))
    want = np.array(want)
    assert np.allclose(spec[k], want, atol=1e-6), f"expected {np.round(want[:5], 4)}..., got {np.round(spec[k][:5], 4)}..."


def test_power_not_amplitude():
    # A frame of constant 1 with a Hann window: the DC bin holds sum(window).
    x = np.ones(320)
    spec = log_power_spectrogram(x, 16000)
    want = np.log(np.sum(np.hanning(320)) ** 2 + 1e-10)
    assert abs(spec[0, 0] - want) < 1e-6, f"expected DC bin {want:.4f} (log of squared magnitude), got {spec[0, 0]:.4f}"


def test_silence_is_floored_by_eps():
    spec = log_power_spectrogram(np.zeros(1000), 16000, eps=1e-6)
    want = np.log(1e-6)
    assert np.allclose(spec, want), f"expected log(eps) = {want:.4f} everywhere, got min {spec.min():.4f} and max {spec.max():.4f}"


def test_hop_moves_the_frame_start():
    x = np.zeros(960)
    x[480:] = 1.0  # a step at sample 480
    spec = log_power_spectrogram(x, 16000)
    energy = np.exp(spec).sum(axis=1)
    quiet = np.where(energy < 1e-6)[0]
    # Frames start at 0, 160, 320, ...; frames 0..1 end at or before sample 480 and see only zeros.
    assert list(quiet) == [0, 1], f"expected only frames 0 and 1 to be silent, got {list(quiet)}"
