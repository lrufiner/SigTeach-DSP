from __future__ import annotations

import numpy as np
import streamlit as st

from sigteach.signals import Signal
from sigteach.signals.io import signal_to_csv_bytes
from sigteach.processing import (
    WINDOW_NAMES,
    compute_dft,
    convolve_sequences,
    correlate_sequences,
    convolution_step,
    dominant_bins,
    LaplaceGridResult,
    canonical_laplace_info,
    canonical_laplace_values,
    laplace_grid,
    laplace_point,
)
from sigteach.visualization import (
    bin_mechanics,
    complex_plane_figure,
    convolution_figure,
    convolution_step_figure,
    correlation_figure,
    cumulative_complex_figure,
    dft_accumulation_animation,
    polar_figure,
    rectangular_figure,
    spectrum_figure,
    time_figure,
    window_shapes_figure,
    window_spectra_figure,
    wrapped_dft_3d_figure,
    canonical_roc_figure,
    laplace_accumulation_animation,
    laplace_cumulative_figure,
    laplace_decomposition_figure,
    laplace_integrand_3d_figure,
    laplace_magnitude_heatmap,
    laplace_magnitude_surface,
    laplace_phase_heatmap,
    laplace_sigma_slices_figure,
)
from .controls import dft_controls, operation_signal_controls, signal_source_controls


def _style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.25);
            border-radius: .7rem;
            padding: .55rem .8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header() -> None:
    st.title("SigTeach DSP Explorer")
    st.caption(
        "DFT, transformada de Laplace, ventanas, suma de exponenciales complejas, "
        "convolución y correlación — una herramienta didáctica interactiva."
    )


def _signal_tab(signal: Signal) -> None:
    st.plotly_chart(time_figure(signal), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Muestras", f"{signal.n}")
    c2.metric("fs", f"{signal.fs:g} Hz")
    c3.metric("Duración", f"{signal.duration:.6g} s")

    st.download_button(
        "Descargar señal como CSV",
        data=signal_to_csv_bytes(signal),
        file_name="sigteach_signal.csv",
        mime="text/csv",
    )

    with st.expander("Interpretación"):
        st.markdown(
            r"""
            Una señal discreta se representa como \(x[n]=x(nT_s)\), con
            \(T_s=1/f_s\). La aplicación conserva explícitamente \(f_s\), porque
            permite interpretar los índices de la DFT en Hz y no solamente como
            frecuencia normalizada.
            """
        )


def _dft_tab(result) -> None:
    c1, c2, c3 = st.columns(3)
    view_label = c1.selectbox("Eje espectral", ["Unilateral", "Centrado", "0 … fs"], index=0)
    db = c2.checkbox("Magnitud en dB", value=False)
    phase_deg = c3.checkbox("Fase en grados", value=False)

    view_map = {"Unilateral": "one-sided", "Centrado": "shifted", "0 … fs": "full"}
    view = view_map[view_label]

    st.plotly_chart(
        spectrum_figure(result, view=view, db=db, phase_degrees=phase_deg),
        use_container_width=True,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(rectangular_figure(result, view=view), use_container_width=True)
    with right:
        st.plotly_chart(complex_plane_figure(result), use_container_width=True)

    st.plotly_chart(polar_figure(result), use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Δf de la grilla FFT", f"{result.bin_spacing_hz:.6g} Hz")
    m2.metric("Duración del marco", f"{result.frame_duration_s:.6g} s")
    m3.metric("NFFT", str(result.n_fft))

    st.subheader("Componentes dominantes")
    st.dataframe(dominant_bins(result, n=10, one_sided=True), use_container_width=True)

    st.info(
        "La separación entre bins es fs/NFFT. Si NFFT se incrementa sólo mediante zero-padding, "
        "la curva espectral queda muestreada más densamente, pero la resolución intrínseca sigue "
        "determinada principalmente por la duración del marco y la ventana."
    )


def _mechanics_tab(result) -> None:
    st.markdown(
        r"""
        Para un bin \(k\), la DFT calcula
        \[
        X[k]=\sum_{n=0}^{L-1} x[n]\,w[n]\,e^{-j2\pi kn/N_{\mathrm{FFT}}}.
        \]
        Cada muestra aporta un vector complejo. La transformada es la suma vectorial
        de todos esos aportes.
        """
    )

    k = int(st.slider("Bin k", 0, result.n_fft - 1, min(1, result.n_fft - 1), 1))
    mech = bin_mechanics(result, k)

    c1, c2, c3 = st.columns(3)
    c1.metric("k", str(k))
    c2.metric("Frecuencia del bin", f"{mech.frequency_hz:.6g} Hz")
    c3.metric("|X[k]|", f"{abs(mech.Xk):.6g}")

    st.plotly_chart(wrapped_dft_3d_figure(mech), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(cumulative_complex_figure(mech), use_container_width=True)
    with right:
        st.markdown("#### Resultado complejo")
        sign = "+" if mech.Xk.imag >= 0 else "-"
        st.latex(rf"X[{k}] = {mech.Xk.real:.6g} {sign} j\,{abs(mech.Xk.imag):.6g}")
        st.latex(rf"|X[{k}]|={abs(mech.Xk):.6g}")
        st.latex(rf"\angle X[{k}]={np.angle(mech.Xk):.6g}\ \mathrm{{rad}}")
        st.markdown(
            "La curva acumulada permite ver la **integración discreta en el tiempo**: "
            "si los vectores se alinean, la suma crece; si se cancelan, el bin queda pequeño."
        )

    if st.checkbox("Mostrar animación de la suma acumulada", value=True):
        st.plotly_chart(dft_accumulation_animation(mech), use_container_width=True)


def _window_tab(signal: Signal, cfg: dict) -> None:
    windows = st.multiselect(
        "Ventanas a comparar",
        list(WINDOW_NAMES),
        default=["Rectangular", "Hann", "Hamming", "Blackman"],
    )
    if not windows:
        st.warning("Selecciona al menos una ventana.")
        return

    st.plotly_chart(window_shapes_figure(cfg["frame_length"], windows), use_container_width=True)
    st.plotly_chart(
        window_spectra_figure(
            signal,
            start=cfg["start"],
            frame_length=cfg["frame_length"],
            n_fft=cfg["n_fft"],
            windows=windows,
        ),
        use_container_width=True,
    )

    st.markdown(
        """
        **Qué conviene observar:** ancho del lóbulo principal, nivel de lóbulos laterales,
        fuga espectral y separación de componentes próximas. No existe una ventana
        universalmente mejor: cada una modifica el compromiso entre resolución y leakage.
        """
    )


def _laplace_tab(signal: Signal, cfg: dict) -> None:
    st.markdown(
        r"""
        La transformada bilateral de Laplace puede interpretarse como una **familia de
        transformadas de Fourier ponderadas exponencialmente**:

        \[
        X(s)=\int_{-\infty}^{\infty}x(t)e^{-st}\,dt,
        \qquad s=\sigma+j\omega,
        \]

        \[
        X(\sigma+j\omega)=\int x(t)\,\underbrace{e^{-\sigma t}}_{\text{ponderación}}
        \underbrace{e^{-j\omega t}}_{\text{rotación}}\,dt.
        \]

        La aplicación interpreta el segmento seleccionado como un registro causal finito que
        comienza en \(t=0\) y aproxima la integral mediante la regla trapezoidal.
        """
    )

    max_available = signal.n - cfg["start"]
    max_laplace_len = min(max_available, 1024)
    default_len = min(cfg["frame_length"], max_laplace_len)
    laplace_len = int(
        st.slider(
            "Longitud del registro para Laplace",
            2 if max_laplace_len >= 2 else 1,
            max_laplace_len,
            max(1, default_len),
            key="laplace_len",
        )
    )
    segment = signal.segment(cfg["start"], laplace_len, name=f"{signal.name} — Laplace")
    T = max(segment.t[-1] if segment.n > 1 else 1.0 / segment.fs, 1.0 / segment.fs)

    subtabs = st.tabs(["Plano s", "Punto s: paso a paso", "Casos analíticos y ROC"])

    with subtabs[0]:
        c1, c2, c3 = st.columns(3)
        sigma_default = max(2.0, min(50.0, 5.0 / T))
        sigma_lim = float(
            c1.number_input(
                "Rango |σ| [s⁻¹]",
                min_value=0.1,
                max_value=500.0,
                value=float(sigma_default),
                step=0.5,
                key="lap_sigma_lim",
            )
        )
        f_default = min(segment.fs / 2.0, max(10.0, segment.fs / 4.0))
        f_max = float(
            c2.number_input(
                "Rango |f| [Hz]",
                min_value=0.1,
                max_value=float(segment.fs / 2.0),
                value=float(f_default),
                step=1.0,
                key="lap_fmax",
            )
        )
        resolution = int(
            c3.select_slider(
                "Resolución del plano",
                options=[31, 41, 51, 61, 81],
                value=51,
                key="lap_grid_res",
            )
        )

        sigmas = np.linspace(-sigma_lim, sigma_lim, resolution)
        freqs = np.linspace(-f_max, f_max, 2 * resolution - 1)
        grid = laplace_grid(segment, sigmas=sigmas, omegas=2.0 * np.pi * freqs)

        st.plotly_chart(laplace_magnitude_heatmap(grid, db=True), use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(laplace_phase_heatmap(grid), use_container_width=True)
        with right:
            st.plotly_chart(
                laplace_sigma_slices_figure(grid, [-0.5 * sigma_lim, 0.0, 0.5 * sigma_lim]),
                use_container_width=True,
            )

        if st.checkbox("Mostrar superficie 3D de |X(s)|", value=True, key="lap_surface_show"):
            st.plotly_chart(laplace_magnitude_surface(grid), use_container_width=True)

        st.info(
            "El corte σ=0 corresponde al eje jω. Si la región de convergencia de una señal "
            "infinita contiene ese eje, la transformada de Fourier existe y se obtiene como "
            "X(jω). Para este registro finito, la integral numérica es finita para todo s finito."
        )

    with subtabs[1]:
        c1, c2 = st.columns(2)
        point_sigma_lim = max(2.0, min(100.0, 5.0 / T))
        sigma0 = float(
            c1.slider(
                "σ del punto s [s⁻¹]",
                min_value=-point_sigma_lim,
                max_value=point_sigma_lim,
                value=0.0,
                step=point_sigma_lim / 100.0,
                key="lap_point_sigma",
            )
        )
        point_fmax = min(segment.fs / 2.0, max(10.0, segment.fs / 4.0))
        f0 = float(
            c2.slider(
                "f = ω/2π [Hz]",
                min_value=-point_fmax,
                max_value=point_fmax,
                value=min(5.0, point_fmax),
                step=max(point_fmax / 200.0, 0.01),
                key="lap_point_freq",
            )
        )
        point = laplace_point(segment, sigma=sigma0, omega=2.0 * np.pi * f0)

        r1, r2, r3 = st.columns(3)
        r1.metric("s", f"{sigma0:.4g} + j{2*np.pi*f0:.4g}")
        r2.metric("|X(s)|", f"{abs(point.value):.6g}")
        r3.metric("∠X(s)", f"{np.angle(point.value):.6g} rad")

        st.plotly_chart(laplace_decomposition_figure(point), use_container_width=True)
        st.markdown(
            r"""
            **Lectura:** primero \(e^{-\sigma t}\) modifica el crecimiento/decaimiento del
            registro. Luego \(e^{-j\omega t}\) hace girar cada contribución en el plano
            complejo. La integral suma esos vectores ponderados.
            """
        )
        st.plotly_chart(laplace_integrand_3d_figure(point), use_container_width=True)
        st.plotly_chart(laplace_cumulative_figure(point), use_container_width=True)
        if st.checkbox("Animar la integración muestra a muestra", value=True, key="lap_anim"):
            st.plotly_chart(laplace_accumulation_animation(point), use_container_width=True)

        st.latex(
            rf"X({sigma0:.4g}+j\,{2*np.pi*f0:.4g}) "
            rf"\approx {point.value.real:.6g} "
            rf"{'+' if point.value.imag >= 0 else '-'} j\,{abs(point.value.imag):.6g}"
        )

    with subtabs[2]:
        st.markdown(
            "Un registro finito no permite experimentar directamente una verdadera región de "
            "convergencia de duración infinita. Por eso esta sección usa pares analíticos clásicos."
        )
        kind = st.selectbox(
            "Ejemplo",
            [
                "Escalón u(t)",
                "Exponencial decreciente",
                "Exponencial creciente",
                "Coseno amortiguado",
                "Seno amortiguado",
            ],
            key="lap_canonical_kind",
        )
        ca, cf = st.columns(2)
        a = float(ca.number_input("Parámetro a [s⁻¹]", 0.05, 50.0, 1.0, 0.1, key="lap_a"))
        fcan = float(cf.number_input("Frecuencia f₀ [Hz]", 0.05, 100.0, 1.0, 0.1, key="lap_fcan"))
        info = canonical_laplace_info(kind, a=a, frequency_hz=fcan)

        st.latex(info.signal_latex)
        st.latex(info.formula_latex)
        st.write(info.explanation)

        span = max(4.0, abs(info.roc_boundary) + 3.0 * max(a, 1.0))
        fspan = max(3.0, 2.5 * fcan)
        st.plotly_chart(
            canonical_roc_figure(info, sigma_span=(-span, span), f_span_hz=(-fspan, fspan)),
            use_container_width=True,
        )

        sigmas_a = np.linspace(-span, span, 81)
        freqs_a = np.linspace(-fspan, fspan, 161)
        S = sigmas_a[:, None] + 1j * 2.0 * np.pi * freqs_a[None, :]
        vals = canonical_laplace_values(kind, S, a=a, frequency_hz=fcan)
        if info.roc_side == "right":
            vals = np.where(sigmas_a[:, None] > info.roc_boundary, vals, np.nan + 1j * np.nan)
        else:
            vals = np.where(sigmas_a[:, None] < info.roc_boundary, vals, np.nan + 1j * np.nan)
        analytic_grid = LaplaceGridResult(
            signal_name=kind,
            fs=1.0,
            sigmas=sigmas_a,
            omegas=2.0 * np.pi * freqs_a,
            values=vals,
        )
        st.plotly_chart(laplace_magnitude_heatmap(analytic_grid, db=True), use_container_width=True)

        fourier_exists = (
            (info.roc_side == "right" and 0.0 > info.roc_boundary)
            or (info.roc_side == "left" and 0.0 < info.roc_boundary)
        )
        if fourier_exists:
            st.success("La ROC contiene el eje jω: la transformada de Fourier existe como X(jω).")
        else:
            st.warning("La ROC no contiene el eje jω: la transformada de Fourier no existe en el sentido ordinario.")

def _convolution_tab(signal: Signal) -> None:
    max_x = min(signal.n, 1024)
    x_len = int(st.slider("Muestras de x usadas", 1, max_x, min(max_x, 128)))
    x = signal.samples[:x_len]
    h = operation_signal_controls(signal.fs, x, key_prefix="conv")

    result = convolve_sequences(x, h, mode="full")
    st.plotly_chart(convolution_figure(x, h, result.values), use_container_width=True)

    st.subheader("Convolución paso a paso")
    m = int(st.slider("Índice de salida m", 0, len(result.values) - 1, 0, 1))
    k, h_shift, product, value = convolution_step(x, h, m)
    st.plotly_chart(convolution_step_figure(k, x, h_shift, product, m), use_container_width=True)
    st.latex(rf"y[{m}] = \sum_k x[k]h[{m}-k] = {value:.8g}")

    rows = [{"n": int(i), "y[n]": float(v)} for i, v in enumerate(result.values[:500])]
    st.dataframe(rows, use_container_width=True)
    if result.values.size > 500:
        st.caption("La tabla se limita a las primeras 500 muestras.")


def _correlation_tab(signal: Signal) -> None:
    max_x = min(signal.n, 2048)
    x_len = int(st.slider("Muestras de x usadas ", 1, max_x, min(max_x, 256)))
    x = signal.samples[:x_len]
    y = operation_signal_controls(signal.fs, x, key_prefix="corr")

    normalization = st.selectbox("Normalización", ["coeff", "none", "biased", "unbiased"], index=0)
    cross = correlate_sequences(x, y, normalization=normalization)
    auto = correlate_sequences(x, None, normalization=normalization)

    st.plotly_chart(
        correlation_figure(cross.lags, cross.values, signal.fs, title="Correlación cruzada rxy"),
        use_container_width=True,
    )
    st.plotly_chart(
        correlation_figure(auto.lags, auto.values, signal.fs, title="Autocorrelación rxx"),
        use_container_width=True,
    )

    peak = int(np.argmax(np.abs(cross.values)))
    st.metric(
        "Retardo del máximo |rxy|",
        f"{int(cross.lags[peak])} muestras = {cross.lags[peak]/signal.fs:.6g} s",
    )


def main() -> None:
    st.set_page_config(
        page_title="SigTeach DSP Explorer",
        page_icon="〰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _style()
    _header()

    signal = signal_source_controls()
    if signal is None:
        st.stop()

    cfg = dft_controls(signal)
    result = compute_dft(signal, **cfg)

    tabs = st.tabs(
        ["Señal", "DFT", "Cómo funciona la DFT", "Laplace", "Ventanas y NFFT", "Convolución", "Correlación"]
    )

    with tabs[0]:
        _signal_tab(signal)
    with tabs[1]:
        _dft_tab(result)
    with tabs[2]:
        _mechanics_tab(result)
    with tabs[3]:
        _laplace_tab(signal, cfg)
    with tabs[4]:
        _window_tab(signal, cfg)
    with tabs[5]:
        _convolution_tab(signal)
    with tabs[6]:
        _correlation_tab(signal)


if __name__ == "__main__":
    main()
