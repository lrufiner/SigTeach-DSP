from __future__ import annotations

import numpy as np
from scipy.signal import chirp as scipy_chirp, square as scipy_square

from .model import Signal


BANK_SIGNAL_NAMES = (
    "Seno",
    "Coseno",
    "Dos tonos",
    "Impulso",
    "Tren de impulsos",
    "Cuadrada",
    "Chirp",
    "Vocal /a/ sintética",
    "Ruido blanco",
)


def _timebase(fs: float, duration: float) -> np.ndarray:
    n = max(1, int(round(float(fs) * float(duration))))
    return np.arange(n, dtype=float) / float(fs)


def _normalize(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x if peak == 0 else x / peak


def _synthetic_vowel_a(t: np.ndarray, fs: float, f0: float = 120.0) -> np.ndarray:
    """Simple harmonic source shaped by broad /a/ formant envelopes."""
    formants = np.array([730.0, 1090.0, 2440.0])
    bandwidths = np.array([90.0, 120.0, 180.0])
    max_h = max(1, int((fs / 2.0) // f0))
    x = np.zeros_like(t)
    for h in range(1, max_h + 1):
        fh = h * f0
        envelope = 0.08
        envelope += float(np.sum(np.exp(-0.5 * ((fh - formants) / bandwidths) ** 2)))
        x += (envelope / h) * np.sin(2.0 * np.pi * fh * t)
    if t.size > 8:
        ramp_n = min(t.size // 10, max(1, int(0.02 * fs)))
        ramp = np.linspace(0.0, 1.0, ramp_n, endpoint=False)
        env = np.ones_like(t)
        env[:ramp_n] = ramp
        env[-ramp_n:] = ramp[::-1]
        x *= env
    return _normalize(x)


def generate_bank_signal(
    kind: str,
    *,
    fs: float = 1000.0,
    duration: float = 1.0,
    amplitude: float = 1.0,
    frequency: float = 50.0,
    frequency2: float = 120.0,
    phase: float = 0.0,
    impulse_index: int = 0,
    impulse_period: int = 100,
    chirp_end: float = 300.0,
    f0: float = 120.0,
    noise_std: float = 0.25,
    seed: int = 7,
) -> Signal:
    """Generate one of the built-in didactic signals."""
    fs = float(fs)
    duration = float(duration)
    if fs <= 0 or duration <= 0:
        raise ValueError("fs and duration must be positive.")

    t = _timebase(fs, duration)
    w0 = 2.0 * np.pi * float(frequency)

    if kind == "Seno":
        x = amplitude * np.sin(w0 * t + phase)
    elif kind == "Coseno":
        x = amplitude * np.cos(w0 * t + phase)
    elif kind == "Dos tonos":
        x = amplitude * (
            np.sin(w0 * t + phase)
            + 0.55 * np.sin(2.0 * np.pi * float(frequency2) * t)
        )
    elif kind == "Impulso":
        x = np.zeros_like(t)
        x[int(np.clip(impulse_index, 0, t.size - 1))] = amplitude
    elif kind == "Tren de impulsos":
        x = np.zeros_like(t)
        period = max(1, int(impulse_period))
        x[::period] = amplitude
    elif kind == "Cuadrada":
        x = amplitude * scipy_square(w0 * t + phase)
    elif kind == "Chirp":
        x = amplitude * scipy_chirp(
            t,
            f0=float(frequency),
            f1=float(chirp_end),
            t1=max(t[-1], 1.0 / fs),
            method="linear",
        )
    elif kind == "Vocal /a/ sintética":
        x = amplitude * _synthetic_vowel_a(t, fs, f0=float(f0))
    elif kind == "Ruido blanco":
        rng = np.random.default_rng(int(seed))
        x = amplitude * rng.normal(0.0, float(noise_std), size=t.size)
    else:
        raise ValueError(f"Unknown bank signal: {kind!r}")

    return Signal(x, fs=fs, name=kind)
