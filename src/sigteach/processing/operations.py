from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import correlate, correlation_lags


@dataclass(frozen=True)
class ConvolutionResult:
    values: np.ndarray
    n: np.ndarray


@dataclass(frozen=True)
class CorrelationResult:
    values: np.ndarray
    lags: np.ndarray


def convolve_sequences(x: np.ndarray, h: np.ndarray, mode: str = "full") -> ConvolutionResult:
    x = np.asarray(x, dtype=float).reshape(-1)
    h = np.asarray(h, dtype=float).reshape(-1)
    if x.size == 0 or h.size == 0:
        raise ValueError("Both sequences must be non-empty.")
    if mode not in {"full", "same", "valid"}:
        raise ValueError("mode must be 'full', 'same' or 'valid'.")
    y = np.convolve(x, h, mode=mode)
    return ConvolutionResult(values=y, n=np.arange(y.size))


def correlate_sequences(
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    normalization: str = "coeff",
) -> CorrelationResult:
    """Cross-correlation r_xy[l] = sum_n x[n] y[n-l] for real sequences."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = x if y is None else np.asarray(y, dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("Signals must be non-empty.")

    r = correlate(x, y, mode="full", method="auto").astype(float)
    lags = correlation_lags(x.size, y.size, mode="full")

    if normalization == "none":
        pass
    elif normalization == "coeff":
        denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
        if denom > 0:
            r = r / denom
    elif normalization == "biased":
        r = r / max(x.size, y.size)
    elif normalization == "unbiased":
        overlap = correlate(np.ones_like(x), np.ones_like(y), mode="full")
        r = np.divide(r, overlap, out=np.zeros_like(r), where=overlap > 0)
    else:
        raise ValueError("Unknown correlation normalization.")

    return CorrelationResult(values=r, lags=lags)


def convolution_step(x: np.ndarray, h: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return k, h[m-k], x[k]h[m-k] and y[m]."""
    x = np.asarray(x, dtype=float).reshape(-1)
    h = np.asarray(h, dtype=float).reshape(-1)
    if x.size == 0 or h.size == 0:
        raise ValueError("Sequences must be non-empty.")
    m = int(m)
    if m < 0 or m > x.size + h.size - 2:
        raise ValueError("m is outside the full-convolution output range.")

    k = np.arange(x.size)
    h_shift = np.zeros_like(x)
    h_index = m - k
    valid = (h_index >= 0) & (h_index < h.size)
    h_shift[valid] = h[h_index[valid]]
    product = x * h_shift
    return k, h_shift, product, float(np.sum(product))
