"""Synthetic distributions used in the toy diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def sample_standard_normal(n: int, dim: int = 2, seed: int | None = None) -> Array:
    """Sample ``n`` points from a standard normal distribution in ``dim`` dimensions."""
    if n <= 0:
        raise ValueError("n must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float64)


def sample_eight_gaussians(
    n: int,
    radius: float = 3.0,
    noise: float = 0.25,
    seed: int | None = None,
) -> Array:
    """Sample a 2-D eight-Gaussian mixture arranged on a circle.

    Parameters
    ----------
    n:
        Number of samples.
    radius:
        Radius of the circle containing the component centers.
    noise:
        Isotropic standard deviation of each component.
    seed:
        Optional random seed.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    rng = np.random.default_rng(seed)
    k = 8
    angles = 2.0 * np.pi * np.arange(k) / k
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    components = rng.integers(0, k, size=n)
    return (centers[components] + noise * rng.normal(size=(n, 2))).astype(np.float64)
