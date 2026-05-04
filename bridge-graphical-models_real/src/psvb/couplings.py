"""Endpoint coupling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

Array = NDArray[np.float64]
IndexArray = NDArray[np.int64]


@dataclass(frozen=True)
class CouplingResult:
    """A paired endpoint sample representing an empirical coupling."""

    z: Array
    x: Array
    permutation: IndexArray
    mean_cost: float


def _check_pair_arrays(z: Array, x: Array) -> tuple[Array, Array]:
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if z.ndim != 2 or x.ndim != 2:
        raise ValueError("z and x must be two-dimensional arrays of shape (n, d)")
    if z.shape != x.shape:
        raise ValueError(f"z and x must have the same shape; got {z.shape} and {x.shape}")
    return z, x


def independent_pairing(z: Array, x: Array, seed: int | None = None, shuffle: bool = True) -> CouplingResult:
    """Create an empirical independent coupling by optionally shuffling target samples.

    If ``z`` and ``x`` were sampled independently, their raw order is already an
    independent empirical coupling. Shuffling is useful when input arrays may have
    correlated order.
    """
    z, x = _check_pair_arrays(z, x)
    n = z.shape[0]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n) if shuffle else np.arange(n)
    paired_x = x[permutation].copy()
    mean_cost = float(np.mean(np.sum((z - paired_x) ** 2, axis=1)))
    return CouplingResult(z=z.copy(), x=paired_x, permutation=permutation.astype(np.int64), mean_cost=mean_cost)


def minibatch_ot_pairing(z: Array, x: Array, power: float = 2.0) -> CouplingResult:
    """Compute a one-to-one minibatch OT-style coupling with the Hungarian algorithm.

    This solves the empirical assignment problem

    ``min_pi sum_i ||z_i - x_{pi(i)}||^power``.

    The implementation forms an O(n^2) cost matrix and is meant for small toy
    diagnostics. For large-scale training, use entropic, sliced, or batched OT.
    """
    z, x = _check_pair_arrays(z, x)
    if power <= 0:
        raise ValueError("power must be positive")

    distances = np.linalg.norm(z[:, None, :] - x[None, :, :], axis=-1) ** power
    row, col = linear_sum_assignment(distances)
    paired_x = np.empty_like(x)
    permutation = np.empty(z.shape[0], dtype=np.int64)
    paired_x[row] = x[col]
    permutation[row] = col.astype(np.int64)
    mean_cost = float(np.mean(distances[row, col]))
    return CouplingResult(z=z.copy(), x=paired_x, permutation=permutation, mean_cost=mean_cost)
