from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go

from sigteach.processing.dft import DFTResult


@dataclass(frozen=True)
class BinMechanics:
    k: int
    frequency_hz: float
    kernel: np.ndarray
    contributions: np.ndarray
    cumulative: np.ndarray
    Xk: complex


def bin_mechanics(result: DFTResult, k: int) -> BinMechanics:
    k = int(k)
    if k < 0 or k >= result.n_fft:
        raise ValueError("k is outside [0, NFFT-1].")
    n = np.arange(result.frame_length)
    kernel = np.exp(-2j * np.pi * k * n / result.n_fft)
    contributions = result.windowed.astype(complex) * kernel
    cumulative = np.cumsum(contributions)
    return BinMechanics(
        k=k,
        frequency_hz=float(result.freqs[k]),
        kernel=kernel,
        contributions=contributions,
        cumulative=cumulative,
        Xk=complex(result.spectrum[k]),
    )


def _sample_indices(n: int, max_points: int = 600) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


def wrapped_dft_3d_figure(mech: BinMechanics) -> go.Figure:
    idx = _sample_indices(mech.contributions.size)
    c = mech.contributions[idx]
    s = mech.cumulative[idx]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=idx,
            y=np.real(c),
            z=np.imag(c),
            mode="lines+markers",
            name="xw[n]·e^{-j2πkn/N}",
            marker=dict(size=3),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=idx,
            y=np.real(s),
            z=np.imag(s),
            mode="lines",
            name="Suma acumulada",
        )
    )
    fig.update_layout(
        title=f"DFT 3D para k={mech.k}: giro complejo y acumulación",
        scene=dict(xaxis_title="n", yaxis_title="Re", zaxis_title="Im"),
        height=650,
        margin=dict(l=0, r=0, t=55, b=0),
    )
    return fig


def cumulative_complex_figure(mech: BinMechanics) -> go.Figure:
    idx = _sample_indices(mech.cumulative.size)
    s = mech.cumulative[idx]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.real(s),
            y=np.imag(s),
            mode="lines+markers",
            customdata=idx,
            hovertemplate="n=%{customdata}<br>Re=%{x:.6g}<br>Im=%{y:.6g}<extra></extra>",
            name="Σ hasta n",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, np.real(mech.Xk)],
            y=[0.0, np.imag(mech.Xk)],
            mode="lines+markers",
            name="X[k]",
        )
    )
    fig.update_layout(
        title=f"Integración discreta en el plano complejo — X[{mech.k}]",
        xaxis_title="Parte real",
        yaxis_title="Parte imaginaria",
        yaxis_scaleanchor="x",
        margin=dict(l=35, r=20, t=55, b=35),
    )
    return fig


def dft_accumulation_animation(mech: BinMechanics, max_frames: int = 80) -> go.Figure:
    n_total = mech.cumulative.size
    frame_points = np.unique(np.linspace(0, n_total - 1, min(max_frames, n_total)).astype(int))
    c = mech.cumulative
    bound = max(
        float(np.max(np.abs(np.real(c)))),
        float(np.max(np.abs(np.imag(c)))),
        1e-6,
    ) * 1.15

    first = int(frame_points[0])
    fig = go.Figure(
        data=[
            go.Scatter(
                x=np.real(c[: first + 1]),
                y=np.imag(c[: first + 1]),
                mode="lines+markers",
                name="Suma acumulada",
            ),
            go.Scatter(
                x=[0.0, np.real(c[first])],
                y=[0.0, np.imag(c[first])],
                mode="lines+markers",
                name="Vector parcial",
            ),
        ],
        layout=go.Layout(
            title=f"Animación de la suma DFT — k={mech.k}",
            xaxis=dict(title="Re", range=[-bound, bound], zeroline=True),
            yaxis=dict(title="Im", range=[-bound, bound], zeroline=True, scaleanchor="x"),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="▶ Reproducir",
                            method="animate",
                            args=[None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}],
                        ),
                        dict(
                            label="❚❚ Pausa",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        ),
                    ],
                )
            ],
            margin=dict(l=35, r=20, t=55, b=35),
        ),
        frames=[
            go.Frame(
                name=str(int(m)),
                data=[
                    go.Scatter(
                        x=np.real(c[: m + 1]),
                        y=np.imag(c[: m + 1]),
                        mode="lines+markers",
                    ),
                    go.Scatter(
                        x=[0.0, np.real(c[m])],
                        y=[0.0, np.imag(c[m])],
                        mode="lines+markers",
                    ),
                ],
            )
            for m in frame_points
        ],
    )
    fig.update_layout(
        sliders=[
            dict(
                currentvalue={"prefix": "n = "},
                steps=[
                    dict(
                        label=str(int(m)),
                        method="animate",
                        args=[[str(int(m))], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    )
                    for m in frame_points
                ],
            )
        ]
    )
    return fig
