# Changelog

## 0.3.0 — 2026-08-17

- Incorporación de la Transformada de Laplace interactiva.
- Visualización numérica del plano `s = σ + jω` mediante magnitud, fase y superficie 3D.
- Cálculo paso a paso de `x(t)e^{-σt}e^{-jωt}` e integración compleja acumulada.
- Animación muestra a muestra de la integral de Laplace.
- Ejemplos analíticos con polos, ceros y región de convergencia (ROC).
- Relación visual entre el eje `jω` de Laplace y la transformada de Fourier.
- Nuevas pruebas unitarias y documentación específica de Laplace.
- Corrección de claves de widgets para evitar colisiones entre pestañas de Streamlit.

## 0.2.0 — 2026-08-17

- Rediseño completo de la arquitectura.
- Banco ampliado de señales.
- CSV y WAV.
- Marco, ventanas y NFFT configurables.
- DFT directa y FFT.
- Magnitud/fase, Re/Im, plano complejo y polar.
- Mecánica de un bin DFT en 3D y animación.
- Comparación de ventanas.
- Convolución paso a paso.
- Autocorrelación y correlación cruzada.
- Pruebas y documentación docente.
