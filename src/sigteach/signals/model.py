from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Signal:
    """Container for a uniformly sampled real-valued signal."""

    samples: np.ndarray
    fs: float
    name: str = "Signal"
    units: str = "a.u."

    def __post_init__(self) -> None:
        x = np.asarray(self.samples, dtype=float).reshape(-1)
        if x.size == 0:
            raise ValueError("A signal must contain at least one sample.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Signal samples must be finite.")
        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("Sampling frequency fs must be positive.")
        object.__setattr__(self, "samples", x)

    @property
    def n(self) -> int:
        return int(self.samples.size)

    @property
    def duration(self) -> float:
        return self.n / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n, dtype=float) / self.fs

    def segment(self, start: int = 0, length: int | None = None, name: str | None = None) -> "Signal":
        start = int(start)
        if start < 0 or start >= self.n:
            raise ValueError("start is outside the signal.")
        stop = self.n if length is None else min(self.n, start + int(length))
        if stop <= start:
            raise ValueError("Segment length must be positive.")
        return Signal(
            self.samples[start:stop],
            fs=self.fs,
            name=name or f"{self.name} [{start}:{stop}]",
            units=self.units,
        )
