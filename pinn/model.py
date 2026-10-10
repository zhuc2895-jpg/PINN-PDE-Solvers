"""Coordinate MLP with the two knobs worth exploring first.

1. ``activation``   -- tanh (the safe default for second-order PDEs), sin
                       (SIREN-style), silu, ...
2. ``n_fourier``    -- 0 keeps the plain MLP, >0 prepends a random Fourier
                       feature map, the cheapest known cure for spectral bias on
                       smooth/periodic PDE solutions.

ReLU is offered but only valid for first-order PDEs: its second derivative is 0.
"""

from __future__ import annotations

import math
from typing import Dict, List, Type

import torch
import torch.nn as nn

__all__ = ["MLP", "FourierFeatures", "Sine", "ACTIVATIONS"]


class Sine(nn.Module):
    """sin(x); ``nn.Sin`` only exists in newer torch releases, so define it here."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


ACTIVATIONS: Dict[str, Type[nn.Module]] = {
    "tanh": nn.Tanh,
    "sin": Sine,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,  # only valid for first-order PDEs: its 2nd derivative is 0
}


class FourierFeatures(nn.Module):
    """gamma(x) = [sin(2*pi*B x), cos(2*pi*B x)], B ~ N(0, sigma^2)."""

    def __init__(self, in_dim: int, n_features: int, sigma: float = 1.0, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(in_dim, n_features, generator=gen) * sigma)
        self.out_dim = 2 * n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class MLP(nn.Module):
    """Fully connected net mapping ``(N, in_dim)`` coordinates to ``(N, out_dim)``."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        *,
        width: int = 64,
        depth: int = 4,
        activation: str = "tanh",
        n_fourier: int = 0,
        fourier_sigma: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; pick one of {sorted(ACTIVATIONS)}"
            )

        torch.manual_seed(seed)
        self.activation = activation
        self.encoder: nn.Module
        if n_fourier > 0:
            self.encoder = FourierFeatures(in_dim, n_fourier, fourier_sigma, seed)
            net_in = self.encoder.out_dim
        else:
            self.encoder = nn.Identity()
            net_in = in_dim

        layers: List[nn.Module] = []
        prev = net_in
        for _ in range(depth):
            layers.append(nn.Linear(prev, width))
            layers.append(ACTIVATIONS[activation]())
            prev = width
        layers.append(nn.Linear(prev, out_dim))  # linear output head
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        if self.activation != "sin":
            return
        # SIREN (Sitzmann et al. 2020): a plain sine net saturates unless the
        # weight scale is matched to the frequency, hence the special init.
        w0 = 30.0
        first = True
        for m in self.net:
            if not isinstance(m, nn.Linear):
                continue
            fan_in = m.weight.shape[1]
            bound = (1.0 / fan_in) if first else math.sqrt(6.0 / fan_in) / w0
            with torch.no_grad():
                m.weight.uniform_(-bound, bound)
                m.bias.uniform_(-bound, bound)
            first = False

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoder(coords))
