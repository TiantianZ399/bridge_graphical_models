"""Training helpers for real-data latent straight-bridge benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .torch_models import VelocityMLP, euler_sample

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TrainResult:
    final_train_loss: float
    eval_mse: float
    losses: list[float]


def _to_tensor(x: Array, device: str) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def train_straight_velocity_model(
    z: Array,
    x: Array,
    steps: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    seed: int = 0,
    device: str | None = None,
) -> tuple[nn.Module, TrainResult]:
    """Train v_theta on paired straight-line bridge samples."""

    if z.shape != x.shape or z.ndim != 2:
        raise ValueError("z and x must have shape (n, d)")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    z_t = _to_tensor(z, device)
    x_t = _to_tensor(x, device)
    n, d = z.shape
    model = VelocityMLP(latent_dim=d, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses: list[float] = []
    for _ in range(steps):
        idx_np = np_rng.integers(0, n, size=batch_size)
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        zb = z_t[idx]
        xb = x_t[idx]
        t = torch.rand(batch_size, device=device)
        xt = (1.0 - t[:, None]) * zb + t[:, None] * xb
        target = xb - zb
        pred = model(xt, t)
        loss = torch.mean(torch.sum((pred - target) ** 2, dim=1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    with torch.no_grad():
        t = torch.rand(n, device=device)
        xt = (1.0 - t[:, None]) * z_t + t[:, None] * x_t
        target = x_t - z_t
        pred = model(xt, t)
        eval_mse = float(torch.mean(torch.sum((pred - target) ** 2, dim=1)).cpu())
    return model, TrainResult(
        final_train_loss=float(np.mean(losses[-min(50, len(losses)) :])),
        eval_mse=eval_mse,
        losses=losses,
    )


def sample_model_latents(
    model: nn.Module,
    n: int,
    latent_dim: int,
    n_steps: int = 50,
    seed: int = 0,
    device: str | None = None,
) -> Array:
    """Sample terminal latents from a trained ODE velocity model."""

    if n <= 0:
        raise ValueError("n must be positive")
    torch.manual_seed(seed)
    if device is None:
        device = next(model.parameters()).device.type
    z0 = torch.randn(n, latent_dim, device=device)
    model.eval()
    y = euler_sample(model, z0, n_steps=n_steps)
    return y.detach().cpu().numpy().astype(np.float64)
