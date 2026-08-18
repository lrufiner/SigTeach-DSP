import numpy as np
import plotly.graph_objects as go

from sigteach.signals import Signal
from sigteach.processing.dft import compute_dft
from sigteach.visualization import (
    bin_mechanics,
    complex_plane_figure,
    cumulative_complex_figure,
    polar_figure,
    rectangular_figure,
    spectrum_figure,
    time_figure,
    wrapped_dft_3d_figure,
)


def test_plot_functions_return_plotly_figures():
    s = Signal(np.sin(2 * np.pi * np.arange(32) / 8), fs=32)
    r = compute_dft(s, frame_length=32, n_fft=32, window="Hann")
    m = bin_mechanics(r, 4)

    figs = [
        time_figure(s),
        spectrum_figure(r),
        rectangular_figure(r),
        complex_plane_figure(r),
        polar_figure(r),
        wrapped_dft_3d_figure(m),
        cumulative_complex_figure(m),
    ]
    assert all(isinstance(fig, go.Figure) for fig in figs)
