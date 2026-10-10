"""The coupling solver: fixed point on ``nu_t``, then a quadrature for ``U``.

The seam is :func:`solve_channel` -- a closure in, a converged profile (or an
honest refusal) out.  Nothing here reaches inside the iteration; the tests pin
the physics the solver claims and the failure mode it must not hide.
"""

from typing import Optional

import numpy as np
import torch

from RANS_PINN_Demo.closure import TruthClosure
from RANS_PINN_Demo.solver import solve_channel, wall_clustered_grid


class ConstantEddyViscosity:
    """A prescribed closure that ignores ``shear`` -- converges in one step."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, y: torch.Tensor, Re_tau: float, shear: Optional[torch.Tensor] = None):
        return torch.full_like(y, self.value)


def test_solver_matches_the_analytic_profile_for_a_constant_eddy_viscosity():
    """With ``nu_t`` constant the momentum equation integrates in closed form.

        U(y) = u_tau**2 * (y - y**2 / 2) / (nu + nu_t)

    The integrand is linear in ``y``, and the trapezoid rule is exact for linear
    integrands, so this is an equality claim, not a tolerance choice.
    """
    nu_t_c = 0.05
    Re_tau = 395.0
    nu = 1.0 / Re_tau

    result = solve_channel(ConstantEddyViscosity(nu_t_c), Re_tau)
    expected = (result.y - result.y**2 / 2.0) / (nu + nu_t_c)

    assert result.converged
    assert torch.allclose(result.U, expected, rtol=1e-10, atol=1e-12)

class ShearDependentEddyViscosity:
    """``nu_t = base + eps * S+``: only solvable by iterating.

    Positive ``eps`` closes a loop -- more ``nu_t`` means less shear means less
    ``nu_t`` -- which contracts for small ``eps`` and diverges for large.
    """

    def __init__(self, base: float = 0.05, eps: float = 1e-3):
        self.base = base
        self.eps = eps

    def __call__(self, y: torch.Tensor, Re_tau: float, shear: Optional[torch.Tensor] = None):
        s = torch.zeros_like(y) if shear is None else shear
        return torch.full_like(y, self.base) + self.eps * s


def _shear_plus_from(nu_t: torch.Tensor, y: torch.Tensor, Re_tau: float) -> torch.Tensor:
    """``dU+/dy+`` implied by ``nu_t``, straight from the total-stress relation.

    Derived here rather than imported so the fixed-point check does not lean on
    the solver's own arithmetic.
    """
    nu = 1.0 / Re_tau
    return ((1.0 - y) / (nu + nu_t)) / Re_tau


def test_a_shear_dependent_closure_is_iterated_to_a_fixed_point():
    """The loop must be real: with a shear-dependent closure the answer has to
    satisfy ``nu_t = F(y, S+(nu_t))``, which one quadrature cannot produce."""
    closure = ShearDependentEddyViscosity()
    Re_tau = 395.0

    result = solve_channel(closure, Re_tau, alpha=1.0, tol=1e-12)

    assert result.converged
    assert result.iterations > 1
    shear = _shear_plus_from(result.nu_t, result.y, Re_tau)
    assert torch.allclose(closure(result.y, Re_tau, shear=shear), result.nu_t, atol=1e-12)


def test_the_reported_residual_history_is_monotone_on_a_converged_solve():
    result = solve_channel(ShearDependentEddyViscosity(), 395.0, tol=1e-12)
    assert result.converged
    assert all(b <= a for a, b in zip(result.residual_norms, result.residual_norms[1:]))


def test_under_relaxation_buys_iterations():
    closure = ShearDependentEddyViscosity(eps=1e-4)
    full = solve_channel(closure, 395.0, alpha=1.0, tol=1e-14)
    damped = solve_channel(closure, 395.0, alpha=0.5, tol=1e-14)
    assert full.converged and damped.converged
    assert damped.iterations > full.iterations


class AntiDampedEddyViscosity:
    """``nu_t = base - k * S+``: positive feedback, and no fixed point.

    More ``nu_t`` means less shear means *more* ``nu_t``.  Unlike a positive
    coefficient -- whose eigenvalue approaches -1 from below and so never truly
    diverges -- this form has no real fixed point for large ``k``.  It is the
    shape of failure H3 has to be able to report.
    """

    def __init__(self, base: float = 0.05, k: float = 5.0):
        self.base = base
        self.k = k

    def __call__(self, y: torch.Tensor, Re_tau: float, shear: Optional[torch.Tensor] = None):
        s = torch.zeros_like(y) if shear is None else shear
        return torch.full_like(y, self.base) - self.k * s


def test_a_diverging_coupling_is_reported_not_hidden():
    """H3 is the claim that the frozen correction does not diverge; a solver
    that cannot say "this diverged" cannot support it."""
    result = solve_channel(AntiDampedEddyViscosity(), 395.0, alpha=1.0, tol=1e-10, max_iter=60)

    assert not result.converged
    assert result.iterations == 60
    # ...and the history must show it rather than looking like a settled solve.
    assert not (result.residual_norms[-1] < result.residual_norms[0])

def test_the_default_grid_resolves_a_real_eddy_viscosity_profile():
    """``nu_t`` spans four decades between the wall and the centre, so "the
    answer" is grid-dependent and the grid has to be defended.

    The reference is the same physics on a far finer grid rather than a recorded
    number, so this stays a statement about resolution and not about one output.
    """
    truth = TruthClosure()
    Re_tau = 590.0

    coarse = solve_channel(truth, Re_tau)
    fine = solve_channel(truth, Re_tau, y=wall_clustered_grid(n=20001))

    assert coarse.converged and fine.converged
    reference = np.interp(
        coarse.y.numpy().ravel(), fine.y.numpy().ravel(), fine.U.numpy().ravel()
    )
    gap = np.linalg.norm(coarse.U.numpy().ravel() - reference) / np.linalg.norm(reference)
    assert gap < 1e-4

def test_a_tiny_relaxation_factor_cannot_manufacture_convergence():
    """The relaxed step is ``alpha * residual``, so a small ``alpha`` makes the
    step small no matter how far from a fixed point the iterate is.

    Judging convergence by the step alone therefore lets the caller dial in a
    "converged" answer for a closure that has no fixed point at all.  The
    residual itself has to be small -- that is the bind the criterion needs.
    """
    closure = AntiDampedEddyViscosity()  # no fixed point
    result = solve_channel(closure, 395.0, alpha=1e-10, tol=1e-10, max_iter=20)

    assert not result.converged
    assert result.residual_norms[-1] > 1e-2
