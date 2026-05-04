"""Bridge path primitives."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def straight_interpolation(z: Array, x: Array, t: float) -> Array:
    """Evaluate the straight-line bridge ``X_t = (1-t) z + t x``."""
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if z.shape != x.shape:
        raise ValueError("z and x must have the same shape")
    if not 0.0 <= float(t) <= 1.0:
        raise ValueError("t must lie in [0, 1]")
    return (1.0 - float(t)) * z + float(t) * x


def straight_velocity(z: Array, x: Array, t: float | None = None) -> Array:
    """Velocity target for the straight-line bridge: ``u_t = x - z``."""
    del t
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if z.shape != x.shape:
        raise ValueError("z and x must have the same shape")
    return x - z


def brownian_bridge_marginal(
    z: Array,
    x: Array,
    t: float,
    epsilon: float = 1.0,
    seed: int | None = None,
) -> Array:
    """Sample the one-time marginal of a Brownian bridge between paired endpoints.

    The marginal is

    ``X_t = (1-t) z + t x + sqrt(epsilon * t * (1-t)) * N(0, I)``.

    This samples a single marginal time, not a full correlated Brownian-bridge path.
    """
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if z.shape != x.shape:
        raise ValueError("z and x must have the same shape")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not 0.0 <= float(t) <= 1.0:
        raise ValueError("t must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    scale = np.sqrt(epsilon * float(t) * (1.0 - float(t)))
    return straight_interpolation(z, x, t) + scale * rng.normal(size=z.shape)
