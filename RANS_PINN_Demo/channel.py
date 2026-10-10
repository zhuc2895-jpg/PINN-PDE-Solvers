"""``ChannelProblem``: the demo posed behind the existing ``PDEProblem`` seam.

The network does not approximate the velocity field -- it corrects a closure.
``U`` comes out of the coupling solve in :mod:`RANS_PINN_Demo.solver`, so the
loss compares *solved* profiles against sparse observations.  That is why
``pinn.trainer.train`` can drive this unchanged: the four protocol methods are
still the whole contract, and the trainer never learns that the field it is
helping to fit is not the field the network emits.

The learned correction sees the local shear, which is what makes the inner
solve a real fixed point rather than a quadrature -- see
``docs/adr/0001-solver-in-the-loop-training.md``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from pinn.problem import Data, PointSet

from .closure import BaselineClosure, Closure, TruthClosure
from .solver import solve_channel

__all__ = ["ChannelClosure", "ChannelProblem", "stratified_points", "RE_TAU_SCALE"]

#: Re_tau is divided by this before the network sees it, so the input is O(1).
RE_TAU_SCALE = 1000.0
#: Relative observation noise.  Applied to training points only.
NOISE_LEVEL = 0.01
DEFAULT_RE_TAUS = (180.0, 395.0)
#: Data budgets the nested point sets are defined for.
DATA_BUDGETS = (8, 12, 16)


def _strata(Re_tau: float) -> Tuple[Tuple[float, float], ...]:
    """Four wall-normal bands in wall units, inner to outer."""
    return (
        (3.0, 30.0),  # buffer layer
        (30.0, 0.15 * Re_tau),  # log layer
        (0.15 * Re_tau, 0.6 * Re_tau),  # outer layer
        # Stops short of the centreline: the evaluation grid ends exactly at
        # y = h, so an observation there would coincide with a scored point.
        (0.6 * Re_tau, 0.98 * Re_tau),
    )


def _four_points(lo: float, hi: float) -> torch.Tensor:
    """Four points spanning ``[lo, hi]``, including both ends."""
    t = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    return lo * (hi / lo) ** t if lo > 0.0 else hi * t


#: Which of each band's four points a budget keeps.  Chosen so the budgets are
#: *nested* -- 8 subset of 12 subset of 16 -- rather than three unrelated clouds.
#: Otherwise the data-volume curve would be measuring placement and volume at
#: the same time, and neither would be attributable.
_NESTED_KEEP = {2: (0, 3), 3: (0, 2, 3), 4: (0, 1, 2, 3)}


def stratified_points(n_per_Re_tau: int, Re_tau: float) -> torch.Tensor:
    """Observation points as ``y/h``, stratified and nested in the budget."""
    if n_per_Re_tau not in DATA_BUDGETS:
        raise ValueError(
            f"data budget must be one of {DATA_BUDGETS} so the sets stay nested, "
            f"got {n_per_Re_tau}"
        )
    keep = _NESTED_KEEP[n_per_Re_tau // 4]
    bands = [_four_points(lo, hi)[list(keep)] for lo, hi in _strata(Re_tau)]
    return (torch.cat(bands) / Re_tau).reshape(-1, 1)


def _interpolate(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear ``y(x_new)`` on a sorted grid; differentiable in ``y``."""
    idx = torch.searchsorted(x, x_new).clamp(1, x.numel() - 1)
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    weight = (x_new - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


class ChannelClosure:
    """``nu_t = nu_t_base * exp(delta)``, with ``delta`` from the network.

    Multiplicative on purpose: ``delta = 0`` reproduces the baseline exactly, so
    training starts from the model under test rather than from noise, and the
    result is positive wherever the baseline is.
    """

    def __init__(self, model: torch.nn.Module, base: Optional[Closure] = None):
        self.model = model
        self.base = base if base is not None else BaselineClosure()

    def __call__(
        self,
        y: torch.Tensor,
        Re_tau: float,
        shear: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        zero_shear = torch.zeros_like(y)
        shear = zero_shear if shear is None else shear
        re_tau = torch.full_like(y, Re_tau / RE_TAU_SCALE)
        features = torch.cat([y, shear, re_tau], dim=1).to(
            next(self.model.parameters()).dtype
        )
        delta = self.model(features).to(y.dtype)
        return self.base(y, Re_tau) * torch.exp(delta)


class ChannelProblem:
    """A :class:`pinn.problem.PDEProblem`: closure correction on a 1-D channel."""

    name = "1d-channel-closure"
    coords = ("y", "Re_tau")
    out_dim = 1

    def __init__(
        self,
        truth: Optional[Closure] = None,
        base: Optional[Closure] = None,
        Re_taus: Sequence[float] = DEFAULT_RE_TAUS,
        noise: float = NOISE_LEVEL,
        seed: int = 0,
        solve_max_iter: int = 30,
    ):
        self.truth = truth if truth is not None else TruthClosure()
        self.base = base if base is not None else BaselineClosure()
        self.Re_taus = tuple(Re_taus)
        self.noise = noise
        self.seed = seed
        self.solve_max_iter = solve_max_iter

    def reference(self, y: torch.Tensor, Re_tau: float) -> torch.Tensor:
        """The profile the observations are drawn from: the truth closure, solved."""
        solved = solve_channel(self.truth, Re_tau)
        return _interpolate(
            y.reshape(-1), solved.y.reshape(-1), solved.U.reshape(-1)
        ).reshape(-1, 1)

    def sample(self, n_interior: int, n_boundary: int, n_initial: int, device) -> Data:
        """``n_interior`` observations *per* Reynolds number, with noise added.

        The seam's boundary/initial slots are unused: this problem is driven by
        sparse data, not by boundary conditions in a field the network emits.
        """
        generator = torch.Generator().manual_seed(self.seed)
        ys, res, targets = [], [], []
        for Re_tau in self.Re_taus:
            y = stratified_points(n_interior, Re_tau).to(device)
            observed = self.reference(y, Re_tau).to(device)
            if self.noise:
                observed = observed * (
                    1.0 + self.noise * torch.randn(y.shape, generator=generator, dtype=y.dtype)
                )
            ys.append(y)
            res.append(torch.full_like(y, Re_tau))
            targets.append(observed)
        return {
            "interior": PointSet(
                {"y": torch.cat(ys), "Re_tau": torch.cat(res)},
                {"U": torch.cat(targets)},
            )
        }

    def losses(self, model: torch.nn.Module, data: Data) -> Dict[str, torch.Tensor]:
        observations = data["interior"]
        y_obs = observations.coords["y"]
        Re_obs = observations.coords["Re_tau"]
        U_observed = observations.targets["U"]
        closure = ChannelClosure(model, self.base)

        per_Re_tau = []
        for Re_tau in self.Re_taus:
            selected = (Re_obs == Re_tau).reshape(-1)
            solved = solve_channel(closure, Re_tau, max_iter=self.solve_max_iter)
            predicted = _interpolate(
                y_obs[selected].reshape(-1),
                solved.y.reshape(-1),
                solved.U.reshape(-1),
            )
            per_Re_tau.append(((predicted - U_observed[selected].reshape(-1)) ** 2).mean())
        return {"data": torch.stack(per_Re_tau).mean()}

    def loss_weights(self) -> Dict[str, float]:
        return {"data": 1.0}

    @staticmethod
    def exact(coords) -> None:
        """No closed form: the reference is the truth closure, solved."""
        return None
