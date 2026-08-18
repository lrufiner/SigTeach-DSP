import numpy as np

from sigteach.signals import Signal
from sigteach.processing.dft import compute_dft
from sigteach.visualization.mechanics import bin_mechanics


def test_cumulative_sum_equals_selected_dft_bin():
    s = Signal(np.array([1.0, 2.0, 0.0, -1.0]), fs=8.0)
    result = compute_dft(s, frame_length=4, n_fft=8, window="Rectangular")
    for k in [0, 1, 3, 7]:
        mech = bin_mechanics(result, k)
        assert np.allclose(mech.cumulative[-1], result.spectrum[k], atol=1e-12)
