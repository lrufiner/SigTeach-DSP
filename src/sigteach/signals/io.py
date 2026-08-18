from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .model import Signal


def _read_bytes(source) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if isinstance(source, bytes):
        return source
    if hasattr(source, "getvalue"):
        data = source.getvalue()
        return data.encode("utf-8") if isinstance(data, str) else bytes(data)
    if hasattr(source, "read"):
        data = source.read()
        return data.encode("utf-8") if isinstance(data, str) else bytes(data)
    raise TypeError("Unsupported file source.")


def parse_manual_samples(text: str) -> np.ndarray:
    """Parse samples separated by commas, semicolons or whitespace."""
    cleaned = text.replace(";", " ").replace(",", " ")
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        raise ValueError("No numeric samples were found.")
    try:
        values = np.asarray([float(token) for token in tokens], dtype=float)
    except ValueError as exc:
        raise ValueError("Manual samples must be numeric.") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("Manual samples must be finite.")
    return values


def load_csv_signal(source, *, fs: float, column: int = 0, name: str = "CSV") -> Signal:
    """Load a signal from a CSV file, ignoring non-numeric rows."""
    raw = _read_bytes(source)
    text = raw.decode("utf-8-sig", errors="replace")
    values: list[float] = []
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if not row or column >= len(row):
            continue
        try:
            value = float(row[column].strip())
        except (ValueError, TypeError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("The CSV does not contain numeric samples in the selected column.")
    return Signal(np.asarray(values), fs=float(fs), name=name)


def load_wav_signal(source, *, name: str = "WAV") -> Signal:
    """Load a mono or stereo WAV file and convert it to mono float64."""
    raw = _read_bytes(source)
    fs, data = wavfile.read(io.BytesIO(raw))
    x = np.asarray(data)

    if x.ndim == 2:
        x = x.astype(np.float64).mean(axis=1)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        scale = max(abs(info.min), abs(info.max))
        x = x.astype(np.float64) / float(scale)
    else:
        x = x.astype(np.float64)

    return Signal(x, fs=float(fs), name=name)


def signal_to_csv_bytes(signal: Signal) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["sample"])
    for value in signal.samples:
        writer.writerow([f"{float(value):.12g}"])
    return out.getvalue().encode("utf-8")
