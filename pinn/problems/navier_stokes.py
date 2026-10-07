"""2-D steady lid-driven cavity (Navier-Stokes), Re = 100.

This is the stress test for the seam: three field outputs, three PDE residuals,
and four boundary regions with different targets -- all through the same
``sample`` / ``losses`` / ``loss_weights`` interface as 1-D Burgers.

No analytic solution exists, so ``exact`` returns None and accuracy has to be
judged against the Ghia et al. (1982) centreline benchmark table.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..derivatives import d, grad_all
from ..problem import Data, PointSet, inputs_from
from ..sampling import uniform

__all__ = ["LidDrivenCavity"]

RE = 100.0


class LidDrivenCavity:
    name = "2d-lid-driven-cavity"
    coords = ("x", "y")
    out_dim = 3  # (u, v, p)

    def __init__(self, re: float = RE):
        self.re = re
        self.nu = 1.0 / re

    def loss_weights(self) -> Dict[str, float]:
        # the original script hard-codes W_CONT = W_MOM = 1, W_BC = 10
        return {"cont": 1.0, "mom_x": 1.0, "mom_y": 1.0, "bc": 10.0}

    def sample(
        self, n_interior: int, n_boundary: int, n_initial: int, device: torch.device
    ) -> Data:
        per_wall = max(n_boundary // 4, 1)
        walls = []
        for wall in range(4):
            s = uniform(per_wall, 0.0, 1.0, device)
            ones, zeros = torch.ones_like(s), torch.zeros_like(s)
            if wall == 0:    # lid, y = 1:  u = 1, v = 0
                walls.append({"x": s, "y": ones, "tu": ones, "tv": zeros})
            elif wall == 1:  # bottom, y = 0
                walls.append({"x": s, "y": zeros, "tu": zeros, "tv": zeros})
            elif wall == 2:  # left, x = 0
                walls.append({"x": zeros, "y": s, "tu": zeros, "tv": zeros})
            else:            # right, x = 1
                walls.append({"x": ones, "y": s, "tu": zeros, "tv": zeros})

        coords = {k: torch.cat([w[k] for w in walls], dim=0) for k in ("x", "y")}
        targets = {k: torch.cat([w[k] for w in walls], dim=0) for k in ("tu", "tv")}
        return {
            "interior": PointSet({"x": uniform(n_interior, 0.0, 1.0, device),
                                  "y": uniform(n_interior, 0.0, 1.0, device)}),
            "boundary": PointSet(coords, {"u": targets["tu"], "v": targets["tv"]}),
        }

    def losses(self, model: torch.nn.Module, data: Data) -> Dict[str, torch.Tensor]:
        inside = data["interior"].coords
        out = model(inputs_from(inside, self.coords))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        du = grad_all(u, inside)
        dv = grad_all(v, inside)
        dp = grad_all(p, inside)

        cont = du["x"] + dv["y"]
        mom_x = (u * du["x"] + v * du["y"] + dp["x"]
                 - self.nu * (d(du["x"], inside["x"]) + d(du["y"], inside["y"])))
        mom_y = (u * dv["x"] + v * dv["y"] + dp["y"]
                 - self.nu * (d(dv["x"], inside["x"]) + d(dv["y"], inside["y"])))

        bc_set = data["boundary"]
        bc_out = model(inputs_from(bc_set.coords, self.coords))
        bc = torch.mean((bc_out[:, 0:1] - bc_set.targets["u"]) ** 2
                        + (bc_out[:, 1:2] - bc_set.targets["v"]) ** 2)

        return {
            "cont": torch.mean(cont**2),
            "mom_x": torch.mean(mom_x**2),
            "mom_y": torch.mean(mom_y**2),
            "bc": bc,
        }

    @staticmethod
    def exact(coords: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        return None  # compare against the Ghia et al. (1982) centreline table
