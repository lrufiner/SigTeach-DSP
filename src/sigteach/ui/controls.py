from __future__ import annotations

import math
import numpy as np
import streamlit as st

from sigteach.signals import (
    BANK_SIGNAL_NAMES,
    Signal,
    generate_bank_signal,
    load_csv_signal,
    load_wav_signal,
    parse_manual_samples,
)


def _bank_signal_controls() -> Signal:
    kind = st.sidebar.selectbox("Señal", BANK_SIGNAL_NAMES, index=0)
    fs = float(st.sidebar.number_input("Frecuencia de muestreo fs [Hz]", 1.0, 192000.0, 1000.0, 10.0))
    duration = float(st.sidebar.number_input("Duración [s]", 0.001, 30.0, 1.0, 0.1, format="%.3f"))
    amplitude = float(st.sidebar.number_input("Amplitud", value=1.0, step=0.1))
    kwargs = dict(fs=fs, duration=duration, amplitude=amplitude)

    if kind in {"Seno", "Coseno", "Dos tonos", "Cuadrada", "Chirp"}:
        kwargs["frequency"] = float(
            st.sidebar.number_input("f₁ [Hz]", 0.0, fs / 2.0, min(50.0, fs / 4.0), 1.0)
        )
    if kind == "Dos tonos":
        kwargs["frequency2"] = float(
            st.sidebar.number_input("f₂ [Hz]", 0.0, fs / 2.0, min(120.0, fs / 3.0), 1.0)
        )
    if kind == "Impulso":
        max_idx = max(0, int(round(fs * duration)) - 1)
        kwargs["impulse_index"] = int(
            st.sidebar.number_input("Índice del impulso", 0, max_idx, min(20, max_idx), 1)
        )
    if kind == "Tren de impulsos":
        kwargs["impulse_period"] = int(
            st.sidebar.number_input(
                "Período [muestras]",
                1,
                max(2, int(fs)),
                max(1, int(fs // 10)),
                1,
            )
        )
    if kind == "Chirp":
        kwargs["chirp_end"] = float(
            st.sidebar.number_input("f final [Hz]", 0.0, fs / 2.0, min(300.0, fs / 2.0), 1.0)
        )
    if kind == "Vocal /a/ sintética":
        kwargs["f0"] = float(
            st.sidebar.number_input(
                "f₀ [Hz]",
                50.0,
                min(400.0, fs / 2.0),
                min(120.0, fs / 4.0),
                1.0,
            )
        )
    if kind == "Ruido blanco":
        kwargs["noise_std"] = float(st.sidebar.number_input("σ del ruido", 0.0, 10.0, 0.25, 0.05))
        kwargs["seed"] = int(st.sidebar.number_input("Semilla", 0, 100000, 7, 1))

    return generate_bank_signal(kind, **kwargs)


def signal_source_controls() -> Signal | None:
    st.sidebar.header("1. Señal")
    source = st.sidebar.radio("Fuente", ["Banco didáctico", "CSV", "WAV", "Manual"], horizontal=True)

    try:
        if source == "Banco didáctico":
            return _bank_signal_controls()

        if source == "CSV":
            upload = st.sidebar.file_uploader("CSV: una columna de muestras", type=["csv"])
            fs = float(st.sidebar.number_input("fs del CSV [Hz]", 1.0, 192000.0, 1000.0, 10.0))
            if upload is None:
                st.info("Carga un CSV para comenzar.")
                return None
            return load_csv_signal(upload, fs=fs, name=upload.name)

        if source == "WAV":
            upload = st.sidebar.file_uploader("Archivo WAV", type=["wav"])
            if upload is None:
                st.info("Carga un WAV para comenzar.")
                return None
            return load_wav_signal(upload, name=upload.name)

        text = st.sidebar.text_area(
            "Muestras",
            value="1, 0, -1, 0, 1, 0, -1, 0",
            help="Separadores admitidos: coma, punto y coma o espacios.",
        )
        fs = float(st.sidebar.number_input("fs manual [Hz]", 1.0, 192000.0, 8.0, 1.0))
        return Signal(parse_manual_samples(text), fs=fs, name="Señal manual")
    except Exception as exc:
        st.sidebar.error(str(exc))
        return None


def dft_controls(signal: Signal) -> dict:
    st.sidebar.header("2. Análisis DFT")
    max_len = min(signal.n, 8192)
    default_len = min(signal.n, 512)
    frame_length = int(
        st.sidebar.slider(
            "Longitud de análisis L",
            min_value=1,
            max_value=max_len,
            value=max(1, default_len),
            step=1,
        )
    )
    max_start = max(0, signal.n - frame_length)
    start = int(
        st.sidebar.slider(
            "Inicio del marco",
            min_value=0,
            max_value=max_start,
            value=0,
            step=1,
        )
    )

    p2 = 1 << max(0, int(math.ceil(math.log2(max(1, frame_length)))))
    candidates = sorted(
        {
            frame_length,
            p2,
            min(8192, max(frame_length, 2 * p2)),
            min(8192, max(frame_length, 4 * p2)),
        }
    )
    candidates = [n for n in candidates if n >= frame_length]
    default_idx = candidates.index(p2) if p2 in candidates else 0
    n_fft = int(st.sidebar.selectbox("NFFT", candidates, index=default_idx))
    window = st.sidebar.selectbox(
        "Ventana",
        ["Rectangular", "Hann", "Hamming", "Blackman", "Bartlett", "Tukey"],
        index=1,
    )
    return {"start": start, "frame_length": frame_length, "n_fft": n_fft, "window": window}


def operation_signal_controls(fs: float, reference: np.ndarray, *, key_prefix: str = "op") -> np.ndarray:
    kind = st.selectbox(
        "Segunda señal / núcleo",
        ["Impulso", "Promedio móvil", "Pulso rectangular", "Seno corto", "Copia retardada de x", "Manual"],
        key=f"{key_prefix}_kind",
    )

    if kind == "Impulso":
        length = int(st.number_input("Longitud h", 1, 256, 9, 1, key=f"{key_prefix}_imp_len"))
        delay = int(st.slider("Posición del impulso", 0, length - 1, min(2, length - 1), key=f"{key_prefix}_imp_delay"))
        h = np.zeros(length)
        h[delay] = 1.0
        return h

    if kind == "Promedio móvil":
        length = int(st.slider("Longitud del promedio", 2, 128, 8, key=f"{key_prefix}_ma_len"))
        return np.ones(length) / length

    if kind == "Pulso rectangular":
        length = int(st.slider("Longitud del pulso", 1, 128, 16, key=f"{key_prefix}_rect_len"))
        return np.ones(length)

    if kind == "Seno corto":
        length = int(st.slider("Longitud", 4, 512, min(128, max(4, len(reference))), key=f"{key_prefix}_sine_len"))
        freq = float(st.number_input("Frecuencia [Hz]", 0.0, fs / 2.0, min(50.0, fs / 4.0), 1.0, key=f"{key_prefix}_sine_freq"))
        n = np.arange(length)
        return np.sin(2 * np.pi * freq * n / fs)

    if kind == "Copia retardada de x":
        delay = int(
            st.slider(
                "Retardo [muestras]",
                0,
                min(512, max(0, len(reference) - 1)),
                min(10, max(0, len(reference) - 1)),
                key=f"{key_prefix}_delay",
            )
        )
        return np.concatenate([np.zeros(delay), reference.copy()])

    text = st.text_area("Muestras de h / y", value="1, 1, 1", key=f"{key_prefix}_manual")
    return parse_manual_samples(text)
