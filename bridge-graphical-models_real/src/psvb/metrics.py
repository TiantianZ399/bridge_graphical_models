"""Lightweight metrics for latent generative benchmarks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _sqeuclidean(x: Array, y: Array) -> Array:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def median_heuristic_sigma(x: Array, y: Array, max_points: int = 1000, seed: int = 0) -> float:
    """Return an RBF bandwidth from the median pairwise distance."""

    rng = np.random.default_rng(seed)
    xy = np.concatenate([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)], axis=0)
    if xy.shape[0] > max_points:
        xy = xy[rng.choice(xy.shape[0], size=max_points, replace=False)]
    d2 = _sqeuclidean(xy, xy)
    vals = d2[np.triu_indices_from(d2, k=1)]
    vals = vals[vals > 0]
    med = float(np.sqrt(np.median(vals))) if vals.size else 1.0
    return max(med, 1e-6)


def rbf_mmd2(x: Array, y: Array, sigma: float | None = None, seed: int = 0) -> float:
    """Biased RBF MMD^2 between two point clouds."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have shape (n, d) and (m, d)")
    if sigma is None:
        sigma = median_heuristic_sigma(x, y, seed=seed)
    gamma = 1.0 / (2.0 * sigma**2)
    return float(
        np.exp(-gamma * _sqeuclidean(x, x)).mean()
        + np.exp(-gamma * _sqeuclidean(y, y)).mean()
        - 2.0 * np.exp(-gamma * _sqeuclidean(x, y)).mean()
    )


def mean_cov_error(x: Array, y: Array) -> tuple[float, float]:
    """Return mean-error norm and Frobenius covariance error."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean_err = float(np.linalg.norm(x.mean(axis=0) - y.mean(axis=0)))
    cx = np.cov(x, rowvar=False)
    cy = np.cov(y, rowvar=False)
    cov_err = float(np.linalg.norm(cx - cy, ord="fro"))
    return mean_err, cov_err
