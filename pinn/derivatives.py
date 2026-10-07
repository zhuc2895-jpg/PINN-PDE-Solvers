"""Single home for every autograd derivative the PINNs need.

Each PDE needs first and second derivatives of its fields with respect to the
coordinates, and every one of them needs the same rule: keep the graph alive so
a second derivative can be taken and the result can still be backpropagated.
Stating that rule once, here, is why ``d`` / ``grad_all`` / ``laplacian`` exist
instead of the same ``torch.autograd.grad`` call at every residual.
"""

from __future__ import annotations

from typing import Dict

import torch

__all__ = ["d", "grad_all", "laplacian"]


def d(y: torch.Tensor, x: torch.Tensor, *, create_graph: bool = True) -> torch.Tensor:
    """Return ``dy/dx`` without collapsing the graph needed for higher orders."""
    if not x.requires_grad:
        x.requires_grad_(True)
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def grad_all(y: torch.Tensor, coords: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """First derivatives of one field with respect to every coordinate.

    ``grad_all(u, {"x": x, "y": y})`` -> ``{"x": u_x, "y": u_y}``.
    """
    return {name: d(y, c) for name, c in coords.items()}


def laplacian(y: torch.Tensor, coords: Dict[str, torch.Tensor]) -> torch.Tensor:
    """``sum_i d2y/dxi2`` -- the operator Poisson and the diffusion term share."""
    first = grad_all(y, coords)
    return sum(d(g, coords[name]) for name, g in first.items())

