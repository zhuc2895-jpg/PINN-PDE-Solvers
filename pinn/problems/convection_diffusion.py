"""2-D steady convection-diffusion, homogeneous Dirichlet, manufactured solution.

    beta . grad(u) - eps * Laplacian(u) = f
    u_exact = sin(pi x) sin(pi y),  beta = (1, 1), eps = 0.1  (Pe ~ 10)

``source`` is derived from the *instance's* beta/eps, not from the module-level
defaults, so ``ConvectionDiffusion2D(beta=..., eps=...)`` stays self-consistent.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from ..derivatives import grad_all, laplacian
from ..problem import Data, PointSet, inputs_from
from ..sampling import uniform, unit_square_boundary

__all__ = ["ConvectionDiffusion2D"]

BETA: Tuple[float, float] = (1.0, 1.0)
EPS = 0.1


class ConvectionDiffusion2D:
    name = "2d-convection-diffusion"
    coords = ("x", "y")
    out_dim = 1

    def __init__(self, beta: Tuple[float, float] = BETA, eps: float = EPS):
        self.beta = beta
        self.eps = eps

    def sample(
        self, n_interior: int, n_boundary: int, n_initial: int, device: torch.device
    ) -> Data:
        return {
            "interior": PointSet({"x": uniform(n_interior, 0.0, 1.0, device),
                                  "y": uniform(n_interior, 0.0, 1.0, device)}),
            "boundary": PointSet(unit_square_boundary(n_boundary, 0.0, 1.0, device)),
        }

    def losses(self, model: torch.nn.Module, data: Data) -> Dict[str, torch.Tensor]:
        inside = data["interior"].coords
        u = model(inputs_from(inside, self.coords))
        grads = grad_all(u, inside)
        convection = self.beta[0] * grads["x"] + self.beta[1] * grads["y"]
        residual = convection - self.eps * laplacian(u, inside) - self.source(inside)
        bc = model(inputs_from(data["boundary"].coords, self.coords))
        return {"pde": torch.mean(residual**2), "bc": torch.mean(bc**2)}

    def loss_weights(self) -> Dict[str, float]:
        return {"pde": 1.0, "bc": 1.0}

    def source(self, coords: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, y = coords["x"], coords["y"]
        conv = self.beta[0] * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y) \
            + self.beta[1] * math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)
        diff = self.eps * 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        return conv + diff

    @staticmethod
    def exact(coords: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        return {"u": torch.sin(math.pi * coords["x"]) * torch.sin(math.pi * coords["y"])}
