"""Pure-NumPy velocity regression for latent bridge benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class LinearVelocityModel:
    weights: Array
    latent_dim: int

    def features(self, x: Array, t: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be two-dimensional")
        ones = np.ones((x.shape[0], 1), dtype=np.float64)
        return np.concatenate(
            [x, x * t, t, t * t, np.sin(np.pi * t), np.cos(np.pi * t), ones],
            axis=1,
        )

    def predict(self, x: Array, t: Array) -> Array:
        return self.features(x, t) @ self.weights


@dataclass(frozen=True)
class LinearTrainResult:
    train_mse: float
    eval_mse: float
    num_design_points: int
    ridge: float


def fit_linear_velocity_model(
    z: Array,
    x: Array,
    repeats: int = 8,
    ridge: float = 1e-4,
    seed: int = 0,
) -> tuple[LinearVelocityModel, LinearTrainResult]:
    """Fit a ridge-regression velocity model for straight-line bridges."""

    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if z.shape != x.shape or z.ndim != 2:
        raise ValueError("z and x must have shape (n, d)")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(seed)
    n, d = z.shape
    z_rep = np.repeat(z, repeats, axis=0)
    x_rep = np.repeat(x, repeats, axis=0)
    t = rng.random(n * repeats)
    xt = (1.0 - t[:, None]) * z_rep + t[:, None] * x_rep
    u = x_rep - z_rep
    dummy = LinearVelocityModel(weights=np.zeros((2 * d + 5, d)), latent_dim=d)
    phi = dummy.features(xt, t)
    gram = phi.T @ phi
    reg = ridge * np.eye(gram.shape[0], dtype=np.float64)
    weights = np.linalg.solve(gram + reg, phi.T @ u)
    model = LinearVelocityModel(weights=weights.astype(np.float64), latent_dim=d)
    pred = model.predict(xt, t)
    train_mse = float(np.mean(np.sum((pred - u) ** 2, axis=1)))
    # Fresh times on the original paired points.
    t_eval = rng.random(n)
    xt_eval = (1.0 - t_eval[:, None]) * z + t_eval[:, None] * x
    u_eval = x - z
    pred_eval = model.predict(xt_eval, t_eval)
    eval_mse = float(np.mean(np.sum((pred_eval - u_eval) ** 2, axis=1)))
    return model, LinearTrainResult(train_mse, eval_mse, int(phi.shape[0]), ridge)


def sample_linear_model(
    model: LinearVelocityModel,
    n: int,
    n_steps: int = 50,
    seed: int = 0,
) -> Array:
    """Sample terminal latents by Euler-integrating the linear velocity model."""

    rng = np.random.default_rng(seed)
    y = rng.normal(size=(n, model.latent_dim)).astype(np.float64)
    dt = 1.0 / float(n_steps)
    for i in range(n_steps):
        t = np.full(y.shape[0], i * dt, dtype=np.float64)
        y = y + dt * model.predict(y, t)
    return y.astype(np.float64)
