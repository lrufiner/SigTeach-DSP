# Módulo interactivo de Transformada de Laplace

## Objetivo didáctico

El módulo no presenta la transformada de Laplace como una fórmula aislada, sino
como una extensión geométrica de la transformada de Fourier:

\[
X(s)=\int x(t)e^{-st}\,dt,\qquad s=\sigma+j\omega,
\]

\[
X(\sigma+j\omega)=\int \underbrace{x(t)e^{-\sigma t}}_{\text{ponderación real}}
\underbrace{e^{-j\omega t}}_{\text{rotación compleja}}\,dt.
\]

La aplicación permite modificar de manera independiente \(\sigma\) y \(\omega\)
para observar qué hace cada parte del núcleo.

## 1. Plano s

La pestaña **Plano s** calcula numéricamente la transformada sobre una grilla
rectangular:

\[
s_{pq}=\sigma_p+j\omega_q.
\]

Se muestran:

- heatmap de magnitud;
- heatmap de fase;
- cortes \(|X(\sigma+j\omega)|\) a \(\sigma\) fijo;
- superficie 3D de magnitud.

El eje vertical marcado por \(\sigma=0\) es el eje imaginario. Cuando una señal
infinita tiene una ROC que contiene dicho eje, la transformada de Fourier existe
como el corte

\[
X(j\omega).
\]

## 2. Evaluación en un punto s

Para un punto

\[
s_0=\sigma_0+j\omega_0,
\]

la app descompone el cálculo en cuatro etapas.

### Paso 1 — ponderación exponencial

Se comparan

\[
x(t),\qquad e^{-\sigma_0t},\qquad x(t)e^{-\sigma_0t}.
\]

- \(\sigma_0>0\): las muestras tardías se atenúan;
- \(\sigma_0=0\): no hay ponderación real y queda una integral de Fourier;
- \(\sigma_0<0\): las muestras tardías se amplifican.

### Paso 2 — rotación compleja

La señal ponderada se multiplica por

\[
e^{-j\omega_0 t}.
\]

Cada muestra pasa a ser una contribución compleja. La curva 3D usa los ejes

\[
(t,\Re\{x(t)e^{-s_0t}\},\Im\{x(t)e^{-s_0t}\}).
\]

### Paso 3 — integración acumulada

La integral se aproxima mediante la regla trapezoidal:

\[
X(s_0)\approx \sum_n q_nx[n]e^{-s_0nT_s}.
\]

La trayectoria compleja acumulada muestra cómo se construye el resultado final.

### Paso 4 — animación

La animación incorpora las contribuciones una a una y permite identificar
visualmente cancelación y alineación de vectores.

## 3. Región de convergencia

Una señal temporal truncada tiene duración finita; por ello su integral numérica
es finita para cualquier valor finito de \(s\). Esto no permite estudiar por sí
solo la convergencia asintótica de señales de duración infinita.

Para enseñar la ROC, SigTeach incluye pares analíticos:

### Escalón

\[
u(t)\longleftrightarrow \frac{1}{s},\qquad \Re\{s\}>0.
\]

### Exponencial causal decreciente

\[
e^{-at}u(t)\longleftrightarrow \frac{1}{s+a},\qquad \Re\{s\}>-a.
\]

### Exponencial causal creciente

\[
e^{at}u(t)\longleftrightarrow \frac{1}{s-a},\qquad \Re\{s\}>a.
\]

### Sinusoides amortiguadas

Los polos conjugados permiten visualizar simultáneamente frecuencia de
oscilación y amortiguamiento.

## 4. Actividad recomendada para introducir Laplace

1. Seleccionar un seno de baja frecuencia.
2. Usar inicialmente \(\sigma=0\).
3. Buscar una frecuencia \(f\) cercana a la frecuencia de la señal.
4. Observar la integral acumulada.
5. Mantener \(f\) y mover \(\sigma\) hacia valores positivos.
6. Repetir con \(\sigma<0\).
7. Pasar al plano s y localizar el eje \(j\omega\).
8. Abrir el ejemplo analítico de una exponencial decreciente.
9. Mover el parámetro \(a\) y observar el desplazamiento del polo y de la ROC.
10. Comparar con una exponencial creciente y discutir por qué su transformada de
    Fourier ordinaria no existe aunque su transformada de Laplace sí.

## Nota numérica

La visualización para señales cargadas utiliza un registro finito y cuadratura
trapezoidal. Es una aproximación de la transformada continua de una señal
interpolada a partir de las muestras; no debe confundirse con una transformada
Z ni con una DFT sin factor de escala temporal.
