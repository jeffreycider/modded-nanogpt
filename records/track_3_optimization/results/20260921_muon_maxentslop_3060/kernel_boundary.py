"""Positive-real-mode boundary for x[t+1] = x[t] - kappa sum c[k] x[t-k].

Use the smallest positive gain at every unit-circle phase crossing, including
theta=pi. This is a frozen-kernel boundary, not an estimator of live curvature.
"""
import math
import numpy as np


def stability_boundary(weights, grid_size=32768, retention=1.0):
    """First positive gain for x_next = retention*x - gain*sum c*x_past.

    Nonunit kernel mass is intentional: decay can add a lag-zero response.
    This does not certify negative/complex spatial modes or live trajectories.
    """
    c = np.asarray(weights, dtype=np.float64)
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all() or c.sum() <= 0:
        raise ValueError("expected a finite kernel with positive DC response")
    if grid_size < 8 * len(c):
        raise ValueError("phase grid too small for this kernel")
    if not -1 < retention <= 1:
        raise ValueError('expected a small-positive stable starting regime')
    # FFT evaluates H(theta)=sum c[k] exp(-ik theta), without an NxL array.
    theta = np.linspace(0, np.pi, grid_size + 1)
    response = np.fft.rfft(c, n=2 * grid_size)
    rhs = retention - np.exp(1j * theta)
    phase = np.imag(rhs * response.conj())
    ages = np.arange(len(c))

    def evaluate(t):
        h = np.dot(c, np.exp(-1j * ages * t))
        return (retention - np.exp(1j * t)) * h.conjugate(), h

    crossings = np.flatnonzero((phase[1:-1] * phase[2:]) <= 0) + 1
    candidates = []
    for index in crossings:
        lo, hi = theta[index], theta[index + 1]
        flo = phase[index]
        for _ in range(45):
            mid = (lo + hi) / 2
            fmid = evaluate(mid)[0].imag
            if flo * fmid <= 0:
                hi = mid
            else:
                lo, flo = mid, fmid
        t = (lo + hi) / 2
        numerator, h = evaluate(t)
        if abs(h) > 1e-12:
            gain = numerator.real / abs(h) ** 2
            if gain > 0:
                candidates.append((float(gain), float(t)))
    hpi = np.dot(c, (-1.0) ** ages)
    if hpi > 1e-12:
        candidates.append((float((retention + 1) / hpi), math.pi))
    if not candidates:
        raise ValueError("no positive-real-mode boundary found")
    return min(candidates)



class KernelBoundaryRatio:
    """Precompute the instantaneous ratio; no RNG or GPU operations."""
    def __init__(self, start, train_steps, kernel_at_step):
        self.start = start
        self.kappas = np.array([stability_boundary(kernel_at_step(t).numpy())[0]
                                for t in range(start, train_steps)])
        self.ratios = self.kappas / self.kappas[0]

    def __call__(self, step):
        return float(self.ratios[step-self.start])
