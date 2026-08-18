from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sigteach.processing.laplace import (
    CanonicalLaplaceInfo,
    LaplaceGridResult,
    LaplacePointResult,
)


def laplace_decomposition_figure(point: LaplacePointResult) -> go.Figure:
    """Show x(t), the exponential envelope and the exponentially weighted signal."""
    weighted = point.samples * point.envelope
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=point.t, y=point.samples, mode="lines", name="x(t)"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=point.t, y=point.envelope, mode="lines", name="e^{-σt}"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=point.t, y=weighted, mode="lines", name="x(t)e^{-σt}"),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="x(t)", row=1, col=1)
    fig.update_yaxes(title_text="e^{-σt}", row=2, col=1)
    fig.update_yaxes(title_text="ponderada", row=3, col=1)
    fig.update_xaxes(title_text="Tiempo [s]", row=3, col=1)
    fig.update_layout(
        title=rf"Paso 1 — ponderación exponencial, σ={point.sigma:.4g} s⁻¹",
        height=720,
        showlegend=False,
        margin=dict(l=45, r=20, t=60, b=35),
    )
    return fig


def laplace_integrand_3d_figure(point: LaplacePointResult, max_points: int = 700) -> go.Figure:
    """3D curve of the complex integrand versus time."""
    n = point.t.size
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    z = point.integrand[idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=point.t[idx],
            y=np.real(z),
            z=np.imag(z),
            mode="lines+markers",
            marker=dict(size=2.5),
            name="x(t)e^{-st}",
            hovertemplate="t=%{x:.6g}s<br>Re=%{y:.6g}<br>Im=%{z:.6g}<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            f"Paso 2 — integrando complejo para s={point.sigma:.4g} "
            f"+ j{point.omega:.4g} rad/s"
        ),
        scene=dict(xaxis_title="t [s]", yaxis_title="Re", zaxis_title="Im"),
        height=650,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def laplace_cumulative_figure(point: LaplacePointResult, max_points: int = 700) -> go.Figure:
    n = point.cumulative.size
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    c = point.cumulative[idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.real(c),
            y=np.imag(c),
            mode="lines+markers",
            customdata=point.t[idx],
            hovertemplate="t=%{customdata:.6g}s<br>Re=%{x:.6g}<br>Im=%{y:.6g}<extra></extra>",
            name="Integral acumulada",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, point.value.real],
            y=[0.0, point.value.imag],
            mode="lines+markers",
            name="X(s)",
        )
    )
    fig.update_layout(
        title="Paso 3 — integral acumulada en el plano complejo",
        xaxis_title="Re",
        yaxis_title="Im",
        yaxis_scaleanchor="x",
        height=590,
        margin=dict(l=45, r=20, t=60, b=40),
    )
    return fig


def laplace_accumulation_animation(point: LaplacePointResult, max_frames: int = 80) -> go.Figure:
    n_total = point.cumulative.size
    frame_points = np.unique(np.linspace(0, n_total - 1, min(max_frames, n_total)).astype(int))
    c = point.cumulative
    bound = max(
        float(np.max(np.abs(np.real(c)))),
        float(np.max(np.abs(np.imag(c)))),
        1e-9,
    ) * 1.15
    first = int(frame_points[0])

    fig = go.Figure(
        data=[
            go.Scatter(
                x=np.real(c[: first + 1]),
                y=np.imag(c[: first + 1]),
                mode="lines+markers",
                name="Integral acumulada",
            ),
            go.Scatter(
                x=[0.0, np.real(c[first])],
                y=[0.0, np.imag(c[first])],
                mode="lines+markers",
                name="Resultado parcial",
            ),
        ],
        frames=[
            go.Frame(
                name=str(int(m)),
                data=[
                    go.Scatter(x=np.real(c[: m + 1]), y=np.imag(c[: m + 1]), mode="lines+markers"),
                    go.Scatter(x=[0.0, np.real(c[m])], y=[0.0, np.imag(c[m])], mode="lines+markers"),
                ],
            )
            for m in frame_points
        ],
    )
    fig.update_layout(
        title="Paso 4 — animación de la integración de Laplace",
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
        sliders=[
            dict(
                currentvalue={"prefix": "muestra = "},
                steps=[
                    dict(
                        label=str(int(m)),
                        method="animate",
                        args=[[str(int(m))], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    )
                    for m in frame_points
                ],
            )
        ],
        height=620,
        margin=dict(l=45, r=20, t=60, b=35),
    )
    return fig


def laplace_magnitude_heatmap(grid: LaplaceGridResult, *, db: bool = True) -> go.Figure:
    mag = np.abs(grid.values).T  # f x sigma
    if db:
        ref = max(float(np.nanmax(mag)), 1e-15)
        z = 20.0 * np.log10(np.maximum(mag / ref, 1e-10))
        label = "Magnitud [dB, normalizada al máximo]"
    else:
        z = mag
        label = "|X(s)|"

    fig = go.Figure(
        go.Heatmap(
            x=grid.sigmas,
            y=grid.frequencies_hz,
            z=z,
            colorbar=dict(title=label),
            hovertemplate="σ=%{x:.5g} s⁻¹<br>f=%{y:.5g} Hz<br>z=%{z:.5g}<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line_dash="dash", annotation_text="eje jω (σ=0)")
    fig.update_layout(
        title="Plano s — magnitud de X(σ+jω)",
        xaxis_title="σ = Re{s} [s⁻¹]",
        yaxis_title="f = Im{s}/2π [Hz]",
        height=650,
        margin=dict(l=50, r=30, t=60, b=45),
    )
    return fig


def laplace_phase_heatmap(grid: LaplaceGridResult) -> go.Figure:
    phase = np.angle(grid.values).T
    fig = go.Figure(
        go.Heatmap(
            x=grid.sigmas,
            y=grid.frequencies_hz,
            z=phase,
            zmin=-np.pi,
            zmax=np.pi,
            colorbar=dict(title="Fase [rad]"),
            hovertemplate="σ=%{x:.5g} s⁻¹<br>f=%{y:.5g} Hz<br>fase=%{z:.5g} rad<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line_dash="dash")
    fig.update_layout(
        title="Plano s — fase de X(σ+jω)",
        xaxis_title="σ = Re{s} [s⁻¹]",
        yaxis_title="f = Im{s}/2π [Hz]",
        height=650,
        margin=dict(l=50, r=30, t=60, b=45),
    )
    return fig


def laplace_magnitude_surface(grid: LaplaceGridResult) -> go.Figure:
    mag = np.abs(grid.values).T
    ref = max(float(np.nanmax(mag)), 1e-15)
    z = 20.0 * np.log10(np.maximum(mag / ref, 1e-10))
    fig = go.Figure(
        go.Surface(
            x=grid.sigmas,
            y=grid.frequencies_hz,
            z=z,
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(
        title="Superficie 3D de |X(s)|",
        scene=dict(
            xaxis_title="σ [s⁻¹]",
            yaxis_title="f [Hz]",
            zaxis_title="Magnitud [dB rel.]",
        ),
        height=700,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def laplace_sigma_slices_figure(grid: LaplaceGridResult, sigmas_to_show: list[float]) -> go.Figure:
    fig = go.Figure()
    for sigma0 in sigmas_to_show:
        i = int(np.argmin(np.abs(grid.sigmas - sigma0)))
        mag = np.abs(grid.values[i])
        fig.add_trace(
            go.Scatter(
                x=grid.frequencies_hz,
                y=mag,
                mode="lines",
                name=f"σ={grid.sigmas[i]:.4g}",
            )
        )
    fig.update_layout(
        title="Cortes verticales del plano s: |X(σ+jω)|",
        xaxis_title="f [Hz]",
        yaxis_title="Magnitud",
        height=520,
        margin=dict(l=45, r=20, t=60, b=40),
    )
    return fig


def canonical_roc_figure(info: CanonicalLaplaceInfo, *, sigma_span: tuple[float, float], f_span_hz: tuple[float, float]) -> go.Figure:
    smin, smax = sigma_span
    fmin, fmax = f_span_hz
    fig = go.Figure()

    if info.roc_side == "right":
        x0, x1 = info.roc_boundary, smax
    else:
        x0, x1 = smin, info.roc_boundary
    fig.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=fmin,
        y1=fmax,
        fillcolor="rgba(0,180,140,0.16)",
        line_width=0,
        layer="below",
    )
    fig.add_vline(x=info.roc_boundary, line_dash="dash", annotation_text="frontera ROC")
    fig.add_vline(x=0.0, line_dash="dot", annotation_text="eje jω")

    if info.poles.size:
        fig.add_trace(
            go.Scatter(
                x=np.real(info.poles),
                y=np.imag(info.poles) / (2.0 * np.pi),
                mode="markers",
                marker=dict(symbol="x", size=14, line=dict(width=2)),
                name="Polos",
            )
        )
    if info.zeros.size:
        fig.add_trace(
            go.Scatter(
                x=np.real(info.zeros),
                y=np.imag(info.zeros) / (2.0 * np.pi),
                mode="markers",
                marker=dict(symbol="circle-open", size=14, line=dict(width=2)),
                name="Ceros",
            )
        )

    fig.update_layout(
        title=f"Polos, ceros y región de convergencia — {info.name}",
        xaxis=dict(title="σ = Re{s} [s⁻¹]", range=[smin, smax]),
        yaxis=dict(title="f = Im{s}/2π [Hz]", range=[fmin, fmax]),
        height=580,
        margin=dict(l=50, r=25, t=60, b=45),
    )
    return fig
