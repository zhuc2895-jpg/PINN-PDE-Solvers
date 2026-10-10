"""The seam: everything PDE-specific lives behind one protocol.

The trainer only ever sees

    sample()        -> named coordinate sets
    losses()        -> {"pde": ..., "bc": ..., "ic": ...} unweighted scalars
    loss_weights()  -> per-term weights
    exact()         -> ground truth, when a closed form exists

so it has no idea whether the problem is steady 2-D Poisson or time-dependent
Burgers, how many field outputs there are, or how many BC regions exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

import torch

__all__ = ["PointSet", "Data", "PDEProblem", "inputs_from"]


@dataclass
class PointSet:
    """Coordinates of one point group, e.g. interior / boundary / initial."""

    coords: Dict[str, torch.Tensor]
    targets: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Targets are data, never part of the autograd graph.  A target that
        # still carries a graph gets reused on the next epoch and blows up with
        # "Trying to backward through the graph a second time" -- detaching here
        # makes that class of bug unrepresentable instead of debugging it twice.
        self.targets = {name: t.detach() for name, t in self.targets.items()}

    def __len__(self) -> int:
        return len(next(iter(self.coords.values())))


#: keyed by "interior" / "boundary" / "initial" (only the ones the PDE needs)
Data = Dict[str, PointSet]  # typing.Dict: runtime alias, must work on py3.8


@runtime_checkable
class PDEProblem(Protocol):
    """A PDE plus its sampling and residual definitions."""

    name: str
    coords: Tuple[str, ...]  # e.g. ("x", "y") or ("x", "t")
    out_dim: int             # 1 for u, 3 for (u, v, p)

    def sample(
        self,
        n_interior: int,
        n_boundary: int,
        n_initial: int,
        device: torch.device,
    ) -> Data: ...

    def losses(self, model: torch.nn.Module, data: Data) -> Dict[str, torch.Tensor]: ...

    def loss_weights(self) -> Dict[str, float]: ...

    def exact(self, coords: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """Ground truth if a manufactured/reference solution exists, else None."""
        ...


def inputs_from(coords: Dict[str, torch.Tensor], order: Tuple[str, ...]) -> torch.Tensor:
    """Stack a coord dict into the ``(N, in_dim)`` tensor the MLP expects."""
    return torch.cat([coords[name] for name in order], dim=1)
