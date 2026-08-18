# Guía docente

## 1. Sinusoide exactamente sobre un bin

- `fs = 1000 Hz`, `L = 1000`, seno de `50 Hz`.
- Ventana rectangular.
- Identificar el bin dominante.
- Explorar ese `k` en **Cómo funciona la DFT**.

Objetivo: vincular coincidencia de frecuencia con alineación de vectores complejos.

## 2. Leakage

- Cambiar la frecuencia a `50.5 Hz`.
- Comparar Rectangular, Hann y Blackman.
- Discutir lóbulo principal y lóbulos laterales.

## 3. Zero-padding

- Mantener fija la señal y `L`.
- Comparar `NFFT=L`, `2L`, `4L`.
- Observar `Δf=fs/NFFT`.

Idea clave: mayor densidad espectral no equivale a nueva información temporal.

## 4. Fase

- Comparar seno y coseno.
- Introducir un desplazamiento temporal.
- Observar la fase de la componente dominante.

## 5. Dos tonos próximos

- Reducir progresivamente la separación entre tonos.
- Aumentar `L`.
- Comparar ventanas.

## 6. Convolución

- Usar un promedio móvil como `h[n]`.
- Desplazar el índice `m`.
- Verificar numéricamente cada `y[m]`.

## 7. Correlación y retardo

- Elegir `Copia retardada de x`.
- Fijar un retardo conocido.
- Verificar el máximo de `|rxy|`.

## 8. Señal tipo voz

- Seleccionar `Vocal /a/ sintética`.
- Usar `fs = 8000 Hz`.
- Modificar `f0`.
- Discutir periodicidad de excitación y envolvente espectral.


## 9. Laplace como Fourier ponderada

1. Elegir una señal senoidal o una vocal sintética.
2. Abrir **Laplace → Punto s: paso a paso**.
3. Fijar primero \(\sigma=0\) y modificar \(f\).
4. Luego mantener \(f\) y mover \(\sigma\) hacia valores positivos y negativos.
5. Observar:
   - la envolvente \(e^{-\sigma t}\),
   - el integrando complejo,
   - la trayectoria de la integral acumulada.

**Objetivo:** comprender que \(\sigma\) no es “otra frecuencia”, sino un parámetro
de crecimiento/decaimiento exponencial.

## 10. El plano s

1. Abrir **Laplace → Plano s**.
2. Observar el heatmap de \(|X(\sigma+j\omega)|\).
3. Identificar el eje \(\sigma=0\).
4. Comparar cortes para \(\sigma<0\), \(\sigma=0\) y \(\sigma>0\).
5. Rotar la superficie 3D.

**Pregunta:** ¿por qué al cambiar \(\sigma\) se modifica la importancia relativa de
las muestras tardías?

## 11. Región de convergencia

1. Abrir **Laplace → Casos analíticos y ROC**.
2. Comparar escalón, exponencial decreciente y exponencial creciente.
3. Identificar polos y frontera de la ROC.
4. Verificar si la ROC contiene el eje \(j\omega\).
5. Relacionar este hecho con la existencia de la transformada de Fourier.

**Caso recomendado:** comparar \(e^{-at}u(t)\) con \(e^{at}u(t)\). Ambos tienen
transformada de Laplace, pero sólo el primero posee transformada de Fourier ordinaria
para \(a>0\).
