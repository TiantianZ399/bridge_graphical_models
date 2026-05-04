"""Estimators for the Markovization gap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from .bridges import straight_interpolation, straight_velocity

Array = NDArray[np.float64]
BridgeFn = Callable[[Array, Array, float], Array]
VelocityFn = Callable[[Array, Array, float | None], Array]


@dataclass(frozen=True)
class MarkovizationGapResult:
    """Estimated conditional-variance gap over a grid of bridge times."""

    times: Array
    gaps: Array
    auc: float
    k_neighbors: int


def _trapezoid(y: Array, x: Array) -> float:
    try:
        return float(np.trapezoid(y, x))  # numpy >= 2.0
    except AttributeError:  # pragma: no cover
        return float(np.trapz(y, x))


def estimate_markovization_gap(
    z: Array,
    x: Array,
    times: Array,
    k_neighbors: int = 32,
    bridge_fn: BridgeFn = straight_interpolation,
    velocity_fn: VelocityFn = straight_velocity,
) -> MarkovizationGapResult:
    """Estimate ``E Var(U_t | X_t)`` with a k-nearest-neighbor regression proxy.

    For each time ``t``, the estimator forms bridge states ``X_t`` and velocity
    targets ``U_t``. A k-NN conditional mean ``E[U_t | X_t]`` is estimated by
    averaging neighboring targets, and the residual squared norm is averaged.

    This is a diagnostic, not a statistically optimal estimator. It is useful for
    comparing bridge/coupling choices under the same sample budget.
    """
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if z.ndim != 2 or x.ndim != 2 or z.shape != x.shape:
        raise ValueError("z and x must have the same shape (n, d)")
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty one-dimensional array")
    if np.any((times < 0.0) | (times > 1.0)):
        raise ValueError("all times must lie in [0, 1]")
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")

    n = z.shape[0]
    k = min(int(k_neighbors), n)
    gaps = []
    for t in times:
        xt = bridge_fn(z, x, float(t))
        u = velocity_fn(z, x, float(t))
        nn = NearestNeighbors(n_neighbors=k).fit(xt)
        idx = nn.kneighbors(xt, return_distance=False)
        mean_u = u[idx].mean(axis=1)
        residual = u - mean_u
        gaps.append(float(np.mean(np.sum(residual**2, axis=1))))

    gap_array = np.asarray(gaps, dtype=np.float64)
    return MarkovizationGapResult(
        times=times.copy(), gaps=gap_array, auc=_trapezoid(gap_array, times), k_neighbors=k
    )
