from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.fft import fft, fftfreq, rfftfreq
from scipy.signal import get_window

from sigteach.signals.model import Signal


WINDOW_NAMES = ("Rectangular", "Hann", "Hamming", "Blackman", "Bartlett", "Tukey")
_WINDOW_MAP = {
    "Rectangular": "boxcar",
    "Hann": "hann",
    "Hamming": "hamming",
    "Blackman": "blackman",
    "Bartlett": "bartlett",
    "Tukey": ("tukey", 0.25),
}


@dataclass(frozen=True)
class DFTResult:
    signal_name: str
    fs: float
    start: int
    frame: np.ndarray
    window_name: str
    window: np.ndarray
    windowed: np.ndarray
    n_fft: int
    spectrum: np.ndarray
    freqs: np.ndarray

    @property
    def frame_length(self) -> int:
        return int(self.frame.size)

    @property
    def coherent_gain_sum(self) -> float:
        return float(np.sum(self.window))

    @property
    def bin_spacing_hz(self) -> float:
        return self.fs / self.n_fft

    @property
    def frame_duration_s(self) -> float:
        return self.frame_length / self.fs


def make_window(name: str, length: int) -> np.ndarray:
    if name not in _WINDOW_MAP:
        raise ValueError(f"Unknown window {name!r}.")
    if length <= 0:
        raise ValueError("Window length must be positive.")
    return np.asarray(get_window(_WINDOW_MAP[name], length, fftbins=True), dtype=float)


def compute_dft(
    signal: Signal,
    *,
    start: int = 0,
    frame_length: int | None = None,
    n_fft: int | None = None,
    window: str = "Rectangular",
) -> DFTResult:
    """Compute a windowed DFT/FFT of an analysis frame."""
    start = int(start)
    if start < 0 or start >= signal.n:
        raise ValueError("start is outside the signal.")

    if frame_length is None:
        frame_length = signal.n - start
    frame_length = int(frame_length)
    if frame_length <= 0:
        raise ValueError("frame_length must be positive.")
    frame_length = min(frame_length, signal.n - start)

    frame = signal.samples[start : start + frame_length].astype(float, copy=True)
    win = make_window(window, frame_length)
    windowed = frame * win

    if n_fft is None:
        n_fft = frame_length
    n_fft = int(n_fft)
    if n_fft < frame_length:
        raise ValueError("n_fft must be >= frame_length.")

    X = fft(windowed, n=n_fft)
    f = fftfreq(n_fft, d=1.0 / signal.fs)
    return DFTResult(
        signal_name=signal.name,
        fs=signal.fs,
        start=start,
        frame=frame,
        window_name=window,
        window=win,
        windowed=windowed,
        n_fft=n_fft,
        spectrum=X,
        freqs=f,
    )


def direct_dft(x: np.ndarray, n_fft: int | None = None) -> np.ndarray:
    """Direct O(N²) DFT, retained deliberately for teaching and verification."""
    x = np.asarray(x, dtype=complex).reshape(-1)
    if n_fft is None:
        n_fft = x.size
    n_fft = int(n_fft)
    if n_fft < x.size:
        raise ValueError("n_fft must be >= len(x).")
    n = np.arange(x.size)
    k = np.arange(n_fft)[:, None]
    W = np.exp(-2j * np.pi * k * n / n_fft)
    return W @ x


def amplitude_spectrum(result: DFTResult, view: str = "one-sided") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequency, amplitude-corrected magnitude and phase."""
    gain = max(abs(result.coherent_gain_sum), np.finfo(float).eps)
    X = result.spectrum

    if view == "one-sided":
        stop = result.n_fft // 2 + 1
        Xv = X[:stop]
        f = rfftfreq(result.n_fft, d=1.0 / result.fs)
        mag = np.abs(Xv) / gain
        if result.n_fft % 2 == 0:
            if mag.size > 2:
                mag[1:-1] *= 2.0
        elif mag.size > 1:
            mag[1:] *= 2.0
    elif view == "shifted":
        Xv = np.fft.fftshift(X)
        f = np.fft.fftshift(result.freqs)
        mag = np.abs(Xv) / gain
    elif view == "full":
        Xv = X
        f = np.arange(result.n_fft, dtype=float) * result.fs / result.n_fft
        mag = np.abs(Xv) / gain
    else:
        raise ValueError("view must be 'one-sided', 'shifted' or 'full'.")

    phase = np.angle(Xv).astype(float)
    threshold = max(float(np.max(mag)) * 1e-10, 1e-14)
    phase[mag < threshold] = np.nan
    return f, mag, phase


def dominant_bins(result: DFTResult, n: int = 8, one_sided: bool = True) -> list[dict]:
    view = "one-sided" if one_sided else "shifted"
    f, mag, phase = amplitude_spectrum(result, view=view)
    if mag.size == 0:
        return []
    idx = np.argsort(mag)[::-1][: max(1, int(n))]
    return [
        {
            "frequency_hz": float(f[i]),
            "amplitude": float(mag[i]),
            "phase_rad": None if np.isnan(phase[i]) else float(phase[i]),
        }
        for i in idx
    ]
