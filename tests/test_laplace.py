import numpy as np

from sigteach.signals import Signal
from sigteach.processing.laplace import (
    canonical_laplace_info,
    canonical_laplace_values,
    laplace_grid,
    laplace_point,
)


def test_laplace_point_matches_finite_constant_integral():
    fs = 1000.0
    n = 1000
    signal = Signal(np.ones(n), fs=fs, name="constant")
    sigma = 1.0
    omega = 2.0
    out = laplace_point(signal, sigma=sigma, omega=omega)
    T = (n - 1) / fs
    s = sigma + 1j * omega
    expected = (1.0 - np.exp(-s * T)) / s
    assert np.allclose(out.value, expected, rtol=2e-6, atol=2e-6)
    assert np.allclose(out.cumulative[-1], out.value)


def test_laplace_grid_agrees_with_point_evaluation():
    signal = Signal(np.array([1.0, 0.5, -0.25, 0.125]), fs=8.0)
    sigmas = np.array([-0.5, 0.0, 1.0])
    omegas = np.array([-2.0, 0.0, 3.0])
    grid = laplace_grid(signal, sigmas=sigmas, omegas=omegas)
    point = laplace_point(signal, sigma=sigmas[2], omega=omegas[0])
    assert np.allclose(grid.values[2, 0], point.value, atol=1e-12)


def test_sigma_zero_dc_is_record_integral():
    fs = 100.0
    signal = Signal(np.ones(101), fs=fs)
    out = laplace_point(signal, sigma=0.0, omega=0.0)
    assert np.isclose(out.value.real, 1.0, atol=1e-12)
    assert np.isclose(out.value.imag, 0.0, atol=1e-12)


def test_canonical_decaying_exponential_pair_and_roc():
    a = 2.5
    info = canonical_laplace_info("Exponencial decreciente", a=a)
    assert np.allclose(info.poles, [-a + 0j])
    assert info.roc_boundary == -a
    assert info.roc_side == "right"
    value = canonical_laplace_values("Exponencial decreciente", np.array([1.0 + 0.0j]), a=a)[0]
    assert np.isclose(value, 1.0 / (1.0 + a))


def test_laplace_visualizations_return_plotly_figures():
    import plotly.graph_objects as go
    from sigteach.visualization.laplace import (
        laplace_cumulative_figure,
        laplace_decomposition_figure,
        laplace_integrand_3d_figure,
        laplace_magnitude_heatmap,
        laplace_magnitude_surface,
        laplace_phase_heatmap,
    )

    signal = Signal(np.sin(2 * np.pi * np.arange(64) / 16), fs=64.0)
    point = laplace_point(signal, sigma=0.5, omega=2 * np.pi * 4)
    grid = laplace_grid(
        signal,
        sigmas=np.linspace(-2, 2, 11),
        omegas=2 * np.pi * np.linspace(-8, 8, 21),
    )
    figs = [
        laplace_decomposition_figure(point),
        laplace_integrand_3d_figure(point),
        laplace_cumulative_figure(point),
        laplace_magnitude_heatmap(grid),
        laplace_phase_heatmap(grid),
        laplace_magnitude_surface(grid),
    ]
    assert all(isinstance(fig, go.Figure) for fig in figs)
