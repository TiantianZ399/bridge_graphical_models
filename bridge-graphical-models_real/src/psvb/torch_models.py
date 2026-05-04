"""Small PyTorch models for latent flow-matching benchmarks."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for scalar time in [0, 1]."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityMLP(nn.Module):
    """A compact velocity network v_theta(x_t, t)."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, time_dim: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        layers: list[nn.Module] = []
        in_dim = latent_dim + time_dim
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time(t)], dim=-1))


@torch.no_grad()
def euler_sample(model: nn.Module, z0: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
    """Sample an ODE path with explicit Euler from t=0 to t=1."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    x = z0.clone()
    dt = 1.0 / float(n_steps)
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
        x = x + dt * model(x, t)
    return x
