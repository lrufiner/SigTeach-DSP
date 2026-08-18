import io
import numpy as np

from sigteach.signals import Signal, generate_bank_signal, load_csv_signal, parse_manual_samples


def test_signal_model_timebase():
    s = Signal(np.arange(4), fs=2.0)
    assert s.duration == 2.0
    assert np.allclose(s.t, [0.0, 0.5, 1.0, 1.5])


def test_bank_sine_length():
    s = generate_bank_signal("Seno", fs=1000, duration=0.25, frequency=50)
    assert s.n == 250
    assert np.max(np.abs(s.samples)) <= 1.0000001


def test_synthetic_vowel_is_finite():
    s = generate_bank_signal("Vocal /a/ sintética", fs=8000, duration=0.2, f0=120)
    assert s.n == 1600
    assert np.all(np.isfinite(s.samples))
    assert np.max(np.abs(s.samples)) <= 1.000001


def test_csv_with_header():
    raw = io.BytesIO(b"sample\n0\n1\n-2.5\n")
    s = load_csv_signal(raw, fs=100)
    assert np.allclose(s.samples, [0, 1, -2.5])
    assert s.fs == 100


def test_manual_parser():
    x = parse_manual_samples("1, 2; -3 4.5")
    assert np.allclose(x, [1, 2, -3, 4.5])
