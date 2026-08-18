import numpy as np

from sigteach.signals import Signal
from sigteach.processing.dft import amplitude_spectrum, compute_dft, direct_dft


def test_direct_dft_matches_fft():
    x = np.array([1.0, 2.0, -1.0, 0.5])
    assert np.allclose(direct_dft(x, n_fft=8), np.fft.fft(x, n=8), atol=1e-12)


def test_impulse_dft_is_flat():
    s = Signal(np.array([1.0, 0.0, 0.0, 0.0]), fs=4.0)
    result = compute_dft(s, frame_length=4, n_fft=4, window="Rectangular")
    assert np.allclose(result.spectrum, np.ones(4))


def test_on_bin_sine_amplitude():
    fs = 128.0
    n = np.arange(128)
    x = np.sin(2 * np.pi * 16 * n / fs)
    result = compute_dft(Signal(x, fs=fs), frame_length=128, n_fft=128, window="Rectangular")
    f, mag, _ = amplitude_spectrum(result, view="one-sided")
    peak = int(np.argmax(mag))
    assert f[peak] == 16.0
    assert np.isclose(mag[peak], 1.0, atol=1e-12)


def test_zero_padding_bin_spacing_changes():
    s = Signal(np.ones(64), fs=640.0)
    a = compute_dft(s, frame_length=64, n_fft=64, window="Rectangular")
    b = compute_dft(s, frame_length=64, n_fft=256, window="Rectangular")
    assert a.bin_spacing_hz == 10.0
    assert b.bin_spacing_hz == 2.5
    assert a.frame_duration_s == b.frame_duration_s
