"""One training loop for every PDE.

The original scripts each carried their own copy of the Adam + StepLR +
accumulate-history + print block; the only real differences were the loss names
and the hard-coded weights.  Here the loop is generic and the weights come from
the problem, so resampling, adaptive weighting, logging and L-BFGS fine-tuning
all become things that are implemented once and apply to every PDE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .device import resolve_device
from .problem import Data, PDEProblem

__all__ = ["TrainConfig", "train"]


@dataclass
class TrainConfig:
    epochs: int = 1500
    lr: float = 1e-3
    lr_step: Optional[int] = 500   # StepLR step_size; None disables the scheduler
    lr_gamma: float = 0.5
    log_every: int = 250
    n_interior: int = 512
    n_boundary: int = 128
    n_initial: int = 0
    resample_every: Optional[int] = None  # None = fixed collocation points
    device: str = "cpu"
    seed: int = 42
    verbose: bool = True
    extra_weights: Dict[str, float] = field(default_factory=dict)


def train(
    problem: PDEProblem, model: nn.Module, config: TrainConfig
) -> Dict[str, List[float]]:
    """Train ``model`` against ``problem`` and return per-term loss history."""
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_step, gamma=config.lr_gamma
        )
        if config.lr_step
        else None
    )

    def draw() -> Data:
        return problem.sample(
            config.n_interior, config.n_boundary, config.n_initial, device
        )

    data = draw()
    weights = {**problem.loss_weights(), **config.extra_weights}
    history: Dict[str, List[float]] = {}

    if config.verbose:
        print(f"[{problem.name}] device={device} epochs={config.epochs}")
        print(f"[{problem.name}] loss weights: {weights}")

    for epoch in range(1, config.epochs + 1):
        if config.resample_every and epoch % config.resample_every == 0:
            data = draw()

        optimizer.zero_grad()
        terms = problem.losses(model, data)
        total = sum(weights.get(k, 1.0) * v for k, v in terms.items())
        total.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for k, v in terms.items():
            history.setdefault(k, []).append(float(v.detach()))
        history.setdefault("total", []).append(float(total.detach()))

        if config.verbose and epoch % config.log_every == 0:
            parts = "  ".join(f"{k}={history[k][-1]:.3e}" for k in terms)
            print(f"  {epoch:6d}  total={history['total'][-1]:.3e}  {parts}")

    return history
