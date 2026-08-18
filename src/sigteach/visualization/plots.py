from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sigteach.signals.model import Signal
from sigteach.processing.dft import DFTResult, amplitude_spectrum, compute_dft, make_window


def time_figure(signal: Signal, *, title: str | None = None, max_markers: int = 160) -> go.Figure:
    mode = "lines+markers" if signal.n <= max_markers else "lines"
    fig = go.Figure(
        go.Scatter(
            x=signal.t,
            y=signal.samples,
            mode=mode,
            name=signal.name,
            hovertemplate="t=%{x:.6g} s<br>x=%{y:.6g}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title or f"Dominio temporal — {signal.name}",
        xaxis_title="Tiempo [s]",
        yaxis_title=f"Amplitud [{signal.units}]",
        margin=dict(l=30, r=20, t=55, b=35),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, zeroline=True)
    return fig


def _view_complex(result: DFTResult, view: str) -> tuple[np.ndarray, np.ndarray]:
    if view == "one-sided":
        stop = result.n_fft // 2 + 1
        return np.fft.rfftfreq(result.n_fft, d=1.0 / result.fs), result.spectrum[:stop]
    if view == "shifted":
        return np.fft.fftshift(result.freqs), np.fft.fftshift(result.spectrum)
    if view == "full":
        f = np.arange(result.n_fft, dtype=float) * result.fs / result.n_fft
        return f, result.spectrum
    raise ValueError("Unknown spectrum view.")


def spectrum_figure(
    result: DFTResult,
    *,
    view: str = "one-sided",
    db: bool = False,
    phase_degrees: bool = False,
) -> go.Figure:
    f, mag, phase = amplitude_spectrum(result, view=view)
    if db:
        floor = max(float(np.max(mag)) * 1e-8, 1e-12)
        mag_plot = 20.0 * np.log10(np.maximum(mag, floor))
        mag_label = "Magnitud [dB re 1]"
    else:
        mag_plot = mag
        mag_label = "Amplitud"

    if phase_degrees:
        phase_plot = np.rad2deg(phase)
        phase_label = "Fase [°]"
    else:
        phase_plot = phase
        phase_label = "Fase [rad]"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=f, y=mag_plot, mode="lines+markers", name="Magnitud"), row=1, col=1)
    fig.add_trace(go.Scatter(x=f, y=phase_plot, mode="markers", name="Fase"), row=2, col=1)
    fig.update_yaxes(title_text=mag_label, row=1, col=1)
    fig.update_yaxes(title_text=phase_label, row=2, col=1)
    fig.update_xaxes(title_text="Frecuencia [Hz]", row=2, col=1)
    fig.update_layout(
        title=f"DFT — magnitud y fase ({result.window_name}, NFFT={result.n_fft})",
        height=620,
        margin=dict(l=40, r=20, t=60, b=35),
        showlegend=False,
    )
    return fig


def rectangular_figure(result: DFTResult, *, view: str = "one-sided") -> go.Figure:
    f, X = _view_complex(result, view)
    gain = max(abs(result.coherent_gain_sum), np.finfo(float).eps)
    Xn = X / gain
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=f, y=np.real(Xn), mode="lines+markers", name="Re"), row=1, col=1)
    fig.add_trace(go.Scatter(x=f, y=np.imag(Xn), mode="lines+markers", name="Im"), row=2, col=1)
    fig.update_yaxes(title_text="Re{X}", row=1, col=1)
    fig.update_yaxes(title_text="Im{X}", row=2, col=1)
    fig.update_xaxes(title_text="Frecuencia [Hz]", row=2, col=1)
    fig.update_layout(
        title="DFT — representación rectangular",
        height=580,
        margin=dict(l=40, r=20, t=60, b=35),
        showlegend=False,
    )
    return fig


def complex_plane_figure(result: DFTResult, *, max_vectors: int = 18) -> go.Figure:
    gain = max(abs(result.coherent_gain_sum), np.finfo(float).eps)
    X = result.spectrum / gain
    mag = np.abs(X)
    top = np.argsort(mag)[::-1][: min(max_vectors, X.size)]
    xs, ys = [], []
    for i in top:
        xs += [0.0, float(np.real(X[i])), None]
        ys += [0.0, float(np.imag(X[i])), None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Vectores dominantes", hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=np.real(X),
            y=np.imag(X),
            mode="markers",
            marker=dict(size=7),
            customdata=np.column_stack((np.arange(X.size), result.freqs)),
            hovertemplate=(
                "k=%{customdata[0]:.0f}<br>f=%{customdata[1]:.6g} Hz"
                "<br>Re=%{x:.6g}<br>Im=%{y:.6g}<extra></extra>"
            ),
            name="X[k]",
        )
    )
    fig.update_layout(
        title="DFT en el plano complejo",
        xaxis_title="Re{X[k]}",
        yaxis_title="Im{X[k]}",
        yaxis_scaleanchor="x",
        margin=dict(l=35, r=20, t=55, b=35),
    )
    return fig


def polar_figure(result: DFTResult, *, max_bins: int = 24) -> go.Figure:
    f, mag, phase = amplitude_spectrum(result, view="one-sided")
    idx = np.argsort(mag)[::-1][: min(max_bins, mag.size)]
    fig = go.Figure(
        go.Scatterpolar(
            r=mag[idx],
            theta=np.rad2deg(phase[idx]),
            mode="markers",
            text=[f"{f[i]:.4g} Hz" for i in idx],
            hovertemplate="%{text}<br>|X|=%{r:.6g}<br>fase=%{theta:.3f}°<extra></extra>",
        )
    )
    fig.update_layout(
        title="Representación polar de componentes dominantes",
        polar=dict(radialaxis=dict(title="Amplitud"), angularaxis=dict(direction="counterclockwise")),
        margin=dict(l=35, r=35, t=55, b=35),
    )
    return fig


def window_shapes_figure(length: int, windows: list[str]) -> go.Figure:
    n = np.arange(length)
    fig = go.Figure()
    for name in windows:
        fig.add_trace(go.Scatter(x=n, y=make_window(name, length), mode="lines", name=name))
    fig.update_layout(
        title=f"Ventanas de análisis (L={length})",
        xaxis_title="n",
        yaxis_title="w[n]",
        margin=dict(l=35, r=20, t=55, b=35),
    )
    return fig


def window_spectra_figure(
    signal: Signal,
    *,
    start: int,
    frame_length: int,
    n_fft: int,
    windows: list[str],
) -> go.Figure:
    fig = go.Figure()
    for name in windows:
        result = compute_dft(signal, start=start, frame_length=frame_length, n_fft=n_fft, window=name)
        f, mag, _ = amplitude_spectrum(result, view="one-sided")
        floor = max(float(np.max(mag)) * 1e-8, 1e-12)
        db = 20.0 * np.log10(np.maximum(mag, floor))
        fig.add_trace(go.Scatter(x=f, y=db, mode="lines", name=name))
    fig.update_layout(
        title="Comparación espectral de ventanas",
        xaxis_title="Frecuencia [Hz]",
        yaxis_title="Amplitud [dB re 1]",
        margin=dict(l=35, r=20, t=55, b=35),
    )
    return fig


def convolution_figure(x: np.ndarray, h: np.ndarray, y: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.09)
    fig.add_trace(go.Scatter(x=np.arange(len(x)), y=x, mode="lines+markers", name="x[n]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(h)), y=h, mode="lines+markers", name="h[n]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode="lines+markers", name="y[n]"), row=3, col=1)
    fig.update_yaxes(title_text="x[n]", row=1, col=1)
    fig.update_yaxes(title_text="h[n]", row=2, col=1)
    fig.update_yaxes(title_text="y[n]", row=3, col=1)
    fig.update_xaxes(title_text="Índice", row=3, col=1)
    fig.update_layout(title="Convolución: y[n] = x[n] * h[n]", height=700, showlegend=False)
    return fig


def convolution_step_figure(k: np.ndarray, x: np.ndarray, h_shift: np.ndarray, product: np.ndarray, m: int) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.09)
    fig.add_trace(go.Scatter(x=k, y=x, mode="lines+markers", name="x[k]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=k, y=h_shift, mode="lines+markers", name="h[m-k]"), row=2, col=1)
    fig.add_trace(go.Bar(x=k, y=product, name="producto"), row=3, col=1)
    fig.update_yaxes(title_text="x[k]", row=1, col=1)
    fig.update_yaxes(title_text="h[m-k]", row=2, col=1)
    fig.update_yaxes(title_text="producto", row=3, col=1)
    fig.update_xaxes(title_text="k", row=3, col=1)
    fig.update_layout(title=f"Paso de convolución para m={m}", height=690, showlegend=False)
    return fig


def correlation_figure(lags: np.ndarray, values: np.ndarray, fs: float, *, title: str) -> go.Figure:
    lag_s = lags / fs
    fig = go.Figure(
        go.Scatter(
            x=lag_s,
            y=values,
            mode="lines+markers",
            customdata=lags,
            hovertemplate="lag=%{customdata} muestras<br>τ=%{x:.6g} s<br>r=%{y:.6g}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Retardo τ [s]",
        yaxis_title="Correlación",
        margin=dict(l=35, r=20, t=55, b=35),
    )
    return fig
