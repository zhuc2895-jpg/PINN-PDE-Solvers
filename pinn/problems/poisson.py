"""2-D Poisson, homogeneous Dirichlet, manufactured solution.

    -Laplacian(u) = f,   u = 0 on the boundary of [0,1]^2
    u_exact = sin(pi x) sin(pi y),   f = 2 pi^2 sin(pi x) sin(pi y)
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from ..derivatives import laplacian
from ..problem import Data, PointSet, inputs_from
from ..sampling import uniform, unit_square_boundary

__all__ = ["Poisson2D"]


class Poisson2D:
    name = "2d-poisson"
    coords = ("x", "y")
    out_dim = 1

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
        residual = -laplacian(u, inside) - self.source(inside)
        bc = model(inputs_from(data["boundary"].coords, self.coords))
        return {
            "pde": torch.mean(residual**2),
            "bc": torch.mean(bc**2),
        }

    def loss_weights(self) -> Dict[str, float]:
        return {"pde": 1.0, "bc": 1.0}

    @staticmethod
    def source(coords: Dict[str, torch.Tensor]) -> torch.Tensor:
        return 2.0 * math.pi**2 * torch.sin(math.pi * coords["x"]) * torch.sin(math.pi * coords["y"])

    @staticmethod
    def exact(coords: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        return {"u": torch.sin(math.pi * coords["x"]) * torch.sin(math.pi * coords["y"])}
