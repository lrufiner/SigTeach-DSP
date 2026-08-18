import numpy as np

from sigteach.processing.operations import convolve_sequences, correlate_sequences, convolution_step


def test_convolution_matches_numpy():
    x = np.array([1.0, 2.0, 3.0])
    h = np.array([1.0, -1.0])
    out = convolve_sequences(x, h)
    assert np.allclose(out.values, np.convolve(x, h))


def test_convolution_step_matches_output():
    x = np.array([1.0, 2.0, 3.0])
    h = np.array([2.0, 1.0])
    y = np.convolve(x, h)
    for m in range(y.size):
        _, _, _, ym = convolution_step(x, h, m)
        assert np.isclose(ym, y[m])


def test_autocorrelation_coeff_peaks_at_zero():
    x = np.array([1.0, -2.0, 1.0, 0.5])
    r = correlate_sequences(x, None, normalization="coeff")
    peak = int(np.argmax(r.values))
    assert r.lags[peak] == 0
    assert np.isclose(r.values[peak], 1.0)
