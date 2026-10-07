"""Shared evaluation: build a grid, predict on it, measure relative L2 error.

Accuracy checking is the part of this repository worth protecting.  A PINN's
loss is a mean residual over collocation points, not an error estimate, so
"is it right?" has to be answered against a reference -- an analytic solution
where one exists, a published benchmark table where one does not.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .problem import inputs_from

__all__ = ["unit_square_grid", "predict_field", "relative_l2", "ghia_u_centreline"]


def unit_square_grid(
    n: int, device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """An ``n x n`` grid on ``[0, 1]^2`` as ``(X, Y, coords)``.

    ``X`` and ``Y`` keep their ``(n, n)`` shape for plotting; ``coords`` is the
    flattened ``(n*n, 1)`` pair the model expects.
    """
    axis = torch.linspace(0.0, 1.0, n, device=device)
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    return X, Y, {"x": X.reshape(-1, 1), "y": Y.reshape(-1, 1)}


def predict_field(
    model: nn.Module,
    problem: object,
    coords: Dict[str, torch.Tensor],
    column: int = 0,
) -> torch.Tensor:
    """Evaluate ``model`` on ``coords`` with autograd disabled, one field at a time."""
    model.eval()
    with torch.no_grad():
        out = model(inputs_from(coords, problem.coords))
    return out[:, column : column + 1].cpu()


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> float:
    """``||pred - true||_2 / ||true||_2`` over the whole field."""
    return float(torch.norm(pred - true) / torch.norm(true))


def ghia_u_centreline() -> Tuple[torch.Tensor, torch.Tensor]:
    """Ghia et al. (1982), Re=100: u along the vertical centreline x = 0.5."""
    y = [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
         0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000]
    u = [1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
         -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775,
         -0.04192, -0.03717, 0.00000]
    return torch.tensor(y, dtype=torch.float32), torch.tensor(u, dtype=torch.float32)
