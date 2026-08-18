from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from sigteach.signals.model import Signal


@dataclass(frozen=True)
class LaplacePointResult:
    """Numerical Laplace transform evaluated at one point s = sigma + j omega."""

    signal_name: str
    fs: float
    sigma: float
    omega: float
    t: np.ndarray
    samples: np.ndarray
    envelope: np.ndarray
    kernel: np.ndarray
    integrand: np.ndarray
    quadrature_weights: np.ndarray
    contributions: np.ndarray
    cumulative: np.ndarray
    value: complex

    @property
    def frequency_hz(self) -> float:
        return self.omega / (2.0 * np.pi)

    @property
    def s(self) -> complex:
        return complex(self.sigma, self.omega)


@dataclass(frozen=True)
class LaplaceGridResult:
    """Numerical Laplace transform on a rectangular s-plane grid."""

    signal_name: str
    fs: float
    sigmas: np.ndarray
    omegas: np.ndarray
    values: np.ndarray  # shape: (len(sigmas), len(omegas))

    @property
    def frequencies_hz(self) -> np.ndarray:
        return self.omegas / (2.0 * np.pi)


@dataclass(frozen=True)
class CanonicalLaplaceInfo:
    name: str
    formula_latex: str
    signal_latex: str
    poles: np.ndarray
    zeros: np.ndarray
    roc_boundary: float
    roc_side: str  # "right" or "left"
    explanation: str


def _quadrature_weights_uniform(n: int, fs: float) -> np.ndarray:
    """Trapezoidal quadrature weights for a uniformly sampled causal record."""
    if n <= 0:
        raise ValueError("n must be positive.")
    dt = 1.0 / float(fs)
    if n == 1:
        return np.array([dt], dtype=float)
    w = np.full(n, dt, dtype=float)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def _safe_real_exponential(exponent: np.ndarray) -> np.ndarray:
    """Avoid floating-point overflow while preserving the didactic trend."""
    return np.exp(np.clip(exponent, -700.0, 700.0))


def laplace_point(signal: Signal, *, sigma: float, omega: float) -> LaplacePointResult:
    r"""Approximate X(s)=∫ x(t)e^{-st}dt for a finite causal record.

    The record is interpreted as starting at t=0. Integration is approximated
    with the trapezoidal rule. For a finite record this numerical transform is
    finite for every finite s; ROC questions for infinite-duration signals are
    handled separately by the canonical analytic examples.
    """
    t = signal.t
    x = signal.samples.astype(float, copy=False)
    sigma = float(sigma)
    omega = float(omega)

    envelope = _safe_real_exponential(-sigma * t)
    kernel = envelope * np.exp(-1j * omega * t)
    integrand = x.astype(complex) * kernel
    q = _quadrature_weights_uniform(signal.n, signal.fs)
    contributions = integrand * q
    cumulative = np.cumsum(contributions)
    value = complex(cumulative[-1])

    return LaplacePointResult(
        signal_name=signal.name,
        fs=signal.fs,
        sigma=sigma,
        omega=omega,
        t=t,
        samples=x.copy(),
        envelope=envelope,
        kernel=kernel,
        integrand=integrand,
        quadrature_weights=q,
        contributions=contributions,
        cumulative=cumulative,
        value=value,
    )


def laplace_grid(
    signal: Signal,
    *,
    sigmas: np.ndarray,
    omegas: np.ndarray,
) -> LaplaceGridResult:
    """Vectorized numerical Laplace transform on a rectangular grid.

    `values[i, j]` corresponds to s = sigmas[i] + j*omegas[j].
    """
    sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
    omegas = np.asarray(omegas, dtype=float).reshape(-1)
    if sigmas.size == 0 or omegas.size == 0:
        raise ValueError("sigmas and omegas must be non-empty.")

    t = signal.t
    qx = signal.samples * _quadrature_weights_uniform(signal.n, signal.fs)

    damping = _safe_real_exponential(-sigmas[:, None] * t[None, :])
    oscillation = np.exp(-1j * omegas[:, None] * t[None, :])
    weighted = damping * qx[None, :]
    values = weighted @ oscillation.T

    return LaplaceGridResult(
        signal_name=signal.name,
        fs=signal.fs,
        sigmas=sigmas,
        omegas=omegas,
        values=values,
    )


def canonical_laplace_info(
    kind: str,
    *,
    a: float = 1.0,
    frequency_hz: float = 1.0,
) -> CanonicalLaplaceInfo:
    """Metadata for classical right-sided continuous-time Laplace examples."""
    a = float(a)
    w0 = 2.0 * np.pi * float(frequency_hz)

    if kind == "Escalón u(t)":
        return CanonicalLaplaceInfo(
            name=kind,
            signal_latex=r"x(t)=u(t)",
            formula_latex=r"X(s)=\frac{1}{s}",
            poles=np.array([0.0 + 0.0j]),
            zeros=np.array([], dtype=complex),
            roc_boundary=0.0,
            roc_side="right",
            explanation="La integral converge sólo si Re{s}>0.",
        )
    if kind == "Exponencial decreciente":
        return CanonicalLaplaceInfo(
            name=kind,
            signal_latex=rf"x(t)=e^{{-{a:g}t}}u(t)",
            formula_latex=rf"X(s)=\frac{{1}}{{s+{a:g}}}",
            poles=np.array([-a + 0.0j]),
            zeros=np.array([], dtype=complex),
            roc_boundary=-a,
            roc_side="right",
            explanation="Para una señal causal decreciente, la ROC está a la derecha del polo.",
        )
    if kind == "Exponencial creciente":
        return CanonicalLaplaceInfo(
            name=kind,
            signal_latex=rf"x(t)=e^{{{a:g}t}}u(t)",
            formula_latex=rf"X(s)=\frac{{1}}{{s-{a:g}}}",
            poles=np.array([a + 0.0j]),
            zeros=np.array([], dtype=complex),
            roc_boundary=a,
            roc_side="right",
            explanation="La exponencial puede crecer y aun así tener transformada: se necesita Re{s}>a.",
        )
    if kind == "Coseno amortiguado":
        return CanonicalLaplaceInfo(
            name=kind,
            signal_latex=rf"x(t)=e^{{-{a:g}t}}\cos({w0:.4g}t)u(t)",
            formula_latex=rf"X(s)=\frac{{s+{a:g}}}{{(s+{a:g})^2+({w0:.4g})^2}}",
            poles=np.array([-a + 1j * w0, -a - 1j * w0]),
            zeros=np.array([-a + 0.0j]),
            roc_boundary=-a,
            roc_side="right",
            explanation="Los polos complejos codifican simultáneamente amortiguamiento y oscilación.",
        )
    if kind == "Seno amortiguado":
        return CanonicalLaplaceInfo(
            name=kind,
            signal_latex=rf"x(t)=e^{{-{a:g}t}}\sin({w0:.4g}t)u(t)",
            formula_latex=rf"X(s)=\frac{{{w0:.4g}}}{{(s+{a:g})^2+({w0:.4g})^2}}",
            poles=np.array([-a + 1j * w0, -a - 1j * w0]),
            zeros=np.array([], dtype=complex),
            roc_boundary=-a,
            roc_side="right",
            explanation="La frecuencia aparece como la parte imaginaria de los polos conjugados.",
        )
    raise ValueError(f"Unknown canonical Laplace example: {kind!r}")


def canonical_laplace_values(
    kind: str,
    s: np.ndarray,
    *,
    a: float = 1.0,
    frequency_hz: float = 1.0,
) -> np.ndarray:
    """Evaluate the analytic transform for a canonical example."""
    s = np.asarray(s, dtype=complex)
    a = float(a)
    w0 = 2.0 * np.pi * float(frequency_hz)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if kind == "Escalón u(t)":
            return 1.0 / s
        if kind == "Exponencial decreciente":
            return 1.0 / (s + a)
        if kind == "Exponencial creciente":
            return 1.0 / (s - a)
        if kind == "Coseno amortiguado":
            return (s + a) / ((s + a) ** 2 + w0**2)
        if kind == "Seno amortiguado":
            return w0 / ((s + a) ** 2 + w0**2)
    raise ValueError(f"Unknown canonical Laplace example: {kind!r}")
