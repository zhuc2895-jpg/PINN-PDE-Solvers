"""1-D viscous Burgers, time dependent.

    u_t + u u_x - nu u_xx = 0,   x in [-1, 1], t in [0, 1]
    u(x, 0) = -sin(pi x),        u(+-1, t) = 0,   nu = 0.01 / pi

No closed-form solution, so ``exact`` returns None and evaluation has to be
against a numerical reference rather than an analytic formula.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from ..derivatives import d
from ..problem import Data, PointSet, inputs_from
from ..sampling import strip_boundary, uniform

__all__ = ["Burgers1D"]

NU = 0.01 / math.pi


class Burgers1D:
    name = "1d-burgers"
    coords = ("x", "t")
    out_dim = 1
    nu = NU

    def sample(
        self, n_interior: int, n_boundary: int, n_initial: int, device: torch.device
    ) -> Data:
        x_ic = uniform(n_initial, -1.0, 1.0, device)
        return {
            "interior": PointSet({"x": uniform(n_interior, -1.0, 1.0, device),
                                  "t": uniform(n_interior, 0.0, 1.0, device)}),
            "boundary": PointSet(strip_boundary(n_boundary, -1.0, 1.0, 0.0, 1.0, device)),
            "initial": PointSet(
                {"x": x_ic, "t": torch.zeros_like(x_ic)},
                {"u": -torch.sin(math.pi * x_ic)},
            ),
        }

    def losses(self, model: torch.nn.Module, data: Data) -> Dict[str, torch.Tensor]:
        inside = data["interior"].coords
        u = model(inputs_from(inside, self.coords))
        u_x = d(u, inside["x"])
        u_t = d(u, inside["t"])
        u_xx = d(u_x, inside["x"])
        residual = u_t + u * u_x - self.nu * u_xx

        bc = model(inputs_from(data["boundary"].coords, self.coords))

        ic_set = data["initial"]
        ic = model(inputs_from(ic_set.coords, self.coords))
        return {
            "pde": torch.mean(residual**2),
            "bc": torch.mean(bc**2),
            "ic": torch.mean((ic - ic_set.targets["u"]) ** 2),
        }

    def loss_weights(self) -> Dict[str, float]:
        return {"pde": 1.0, "bc": 1.0, "ic": 1.0}

    @staticmethod
    def exact(coords: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        return None  # no analytic solution; evaluate against a numerical reference
