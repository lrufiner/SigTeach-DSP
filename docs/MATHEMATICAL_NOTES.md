# Notas matemáticas

## DFT

\[
X[k]=\sum_{n=0}^{N-1}x[n]e^{-j2\pi kn/N}
\]

\[
x[n]=\frac{1}{N}\sum_{k=0}^{N-1}X[k]e^{j2\pi kn/N}.
\]

Si \(L<N_{\mathrm{FFT}}\), se aplica zero-padding desde \(L\).

## Frecuencia de un bin

\[
f_k = \frac{k f_s}{N_{\mathrm{FFT}}}.
\]

## Magnitud unilateral

\[
A[k] = \frac{|X[k]|}{\sum_n w[n]}.
\]

Se duplican los bins positivos salvo DC y Nyquist.

## Fase

\[
\phi[k]=\arg X[k].
\]

La fase se enmascara cuando la magnitud es numéricamente insignificante.

## Mecánica de un bin

\[
c_k[n]=x[n]w[n]e^{-j2\pi kn/N_{\mathrm{FFT}}}
\]

\[
S_k[m]=\sum_{n=0}^{m}c_k[n]
\]

y finalmente:

\[
S_k[L-1]=X[k].
\]

## Convolución

\[
y[m]=\sum_k x[k]h[m-k].
\]

## Correlación

\[
r_{xy}[\ell]=\sum_n x[n]y[n-\ell].
\]


## Transformada de Laplace

La convención utilizada es

\[
X(s)=\int_{-\infty}^{\infty}x(t)e^{-st}\,dt,
\qquad s=\sigma+j\omega.
\]

Separando las dos componentes del exponente:

\[
X(\sigma+j\omega)=\int x(t)e^{-\sigma t}e^{-j\omega t}\,dt.
\]

Esto permite interpretar Laplace como una transformada de Fourier de la señal
ponderada exponencialmente:

\[
x_\sigma(t)=x(t)e^{-\sigma t}.
\]

En la aplicación, un registro de muestras se interpreta como un segmento causal
finito que comienza en \(t=0\), y la integral se aproxima mediante la regla
trapezoidal:

\[
X(s)\approx \sum_n q_n x[n]e^{-s nT_s},
\]

donde \(q_n\) son los pesos de integración.

### Relación con Fourier

Si la región de convergencia contiene el eje imaginario, entonces

\[
X(j\omega)=\mathcal{F}\{x(t)\}.
\]

Por eso el corte \(\sigma=0\) del plano de Laplace es particularmente importante.

### Región de convergencia

La expresión algebraica racional de \(X(s)\) no determina por sí sola la señal:
la **ROC** forma parte de la especificación de la transformada. Para señales
causales racionales, la ROC se encuentra a la derecha del polo de mayor parte real.

Ejemplo:

\[
x(t)=e^{-at}u(t)
\quad\Longleftrightarrow\quad
X(s)=\frac{1}{s+a},
\qquad \Re\{s\}>-a.
\]

Un registro finito tiene integral finita para cualquier \(s\) finito; por ello la
app usa ejemplos analíticos de duración infinita para enseñar polos, ceros y ROC.
