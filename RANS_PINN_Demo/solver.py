"""The coupling solver: a fixed point on ``nu_t``, then a quadrature for ``U``.

Non-dimensionalisation is the same as :mod:`RANS_PINN_Demo.closure`: ``h = 1``,
``u_tau = 1``, ``nu = 1 / Re_tau``.  Integrating the total-stress relation from
the wall gives

    dU/dy = u_tau**2 * (1 - y/h) / (nu + nu_t)

so ``U`` is a cumulative quadrature *once ``nu_t`` is known*.  The fixed point
is needed because a learned closure may depend on the local shear, which depends
on ``U``, which depends on ``nu_t``: a closure that depends only on ``y`` would
make this loop a fiction, and the loop is the thing H3 is about (see
``docs/adr/0001-solver-in-the-loop-training.md``).

``nu_t`` is the field being iterated; ``U`` is a deterministic functional of it,
so a small fixed-point residual is what makes the whole coupled state settled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .closure import Closure

__all__ = ["SolveResult", "solve_channel", "shear_plus", "wall_clustered_grid", "U_TAU", "H"]

U_TAU = 1.0
H = 1.0


@dataclass
class SolveResult:
    """Everything the caller needs to judge the solve, not just the answer."""

    y: torch.Tensor
    U: torch.Tensor
    nu_t: torch.Tensor
    iterations: int
    converged: bool
    #: ``max|F(nu_t) - nu_t|`` per iteration, evaluated *before* relaxation.
    #: This is the honest measure of distance from the fixed point.
    residual_norms: List[float] = field(default_factory=list)
    #: ``max|nu_t_next - nu_t|`` per iteration -- the relaxed step actually taken.
    step_norms: List[float] = field(default_factory=list)


def wall_clustered_grid(n: int = 801, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """``y = t**2``: quadratic clustering at the wall, where ``nu_t`` and
    ``dU/dy`` both change fastest."""
    t = torch.linspace(0.0, 1.0, n, dtype=dtype)
    return (t**2).reshape(-1, 1)


def _cumulative_trapezoid(f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``int_0^y f`` at each grid point, with the value at ``y = 0`` fixed at 0."""
    segments = 0.5 * (f[1:] + f[:-1]) * (y[1:] - y[:-1])
    out = torch.zeros_like(f)
    out[1:] = torch.cumsum(segments, dim=0)
    return out


def _du_dy(nu_t: torch.Tensor, y: torch.Tensor, Re_tau: float) -> torch.Tensor:
    """Mean shear implied by ``nu_t``, straight from the total-stress relation.

    The one place that relation is written; both the shear handed to the closure
    and the final quadrature go through it.
    """
    nu = 1.0 / Re_tau
    return U_TAU**2 * (1.0 - y / H) / (nu + nu_t)


def shear_plus(nu_t: torch.Tensor, y: torch.Tensor, Re_tau: float) -> torch.Tensor:
    """Wall-unit mean shear ``dU+/dy+`` implied by ``nu_t``."""
    return _du_dy(nu_t, y, Re_tau) / Re_tau


def _is_monotone_decreasing(residual: List[float]) -> bool:
    """True when the whole residual history is non-increasing.

    A small relative slack absorbs round-off.  Requiring the *whole* history,
    not just the last few steps, is the point: an iteration that ever grew --
    an oscillation, a limit cycle -- stays unconverged and has to be reported.
    """
    return all(b <= a * (1.0 + 1e-6) for a, b in zip(residual, residual[1:]))


def solve_channel(
    closure: Closure,
    Re_tau: float,
    y: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> SolveResult:
    """Iterate ``nu_t`` to a fixed point, then integrate the profile.

    ``alpha`` is the under-relaxation factor: ``1.0`` takes the closure's full
    step, smaller values damp it.  ``converged`` is False whenever the iteration
    ran out of budget *or* the residual never settled -- the caller is never
    handed the last iterate dressed up as an answer.

    Convergence requires the *residual* to be below ``tol``, not merely the step.
    The step is ``alpha * residual``, so judging by the step alone would let any
    caller manufacture convergence by shrinking ``alpha``.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    if y is None:
        y = wall_clustered_grid()
    y = y.to(torch.float64)

    nu_t = closure(y, Re_tau).to(torch.float64)
    residual_norms: List[float] = []
    step_norms: List[float] = []
    converged = False

    for _ in range(max_iter):
        shear = shear_plus(nu_t, y, Re_tau)
        target = closure(y, Re_tau, shear=shear).to(torch.float64)
        residual = target - nu_t
        step = alpha * residual
        residual_norms.append(float(residual.abs().max()))
        step_norms.append(float(step.abs().max()))
        nu_t = nu_t + step

        if (
            residual_norms[-1] < tol
            and step_norms[-1] < tol
            and _is_monotone_decreasing(residual_norms)
        ):
            converged = True
            break

    return SolveResult(
        y=y,
        U=_cumulative_trapezoid(_du_dy(nu_t, y, Re_tau), y),
        nu_t=nu_t,
        iterations=len(residual_norms),
        converged=converged,
        residual_norms=residual_norms,
        step_norms=step_norms,
    )
