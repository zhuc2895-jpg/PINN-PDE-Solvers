"""Residual integrity: plug a known solution into each PDE and the residual must vanish.

These are the tests worth keeping. Each one is checked against an independent
source of truth -- a closed-form solution whose derivatives are known by hand --
so a passing test means the operator coded in the problem really is the operator
named in its docstring, not merely that some loss went down.
"""

import math

import torch
import torch.nn as nn

from pinn.problems import (
    Burgers1D,
    ConvectionDiffusion2D,
    LidDrivenCavity,
    Poisson2D,
)

CPU = torch.device("cpu")


class Analytic(nn.Module):
    """Wraps a closed-form field so it can be handed to ``problem.losses``."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, coords):  # coords: (N, n_dims)
        return self.fn(coords)


def _sin_sin(coords):
    return torch.sin(math.pi * coords[:, 0:1]) * torch.sin(math.pi * coords[:, 1:2])


def test_poisson_residual_vanishes_on_the_manufactured_solution():
    problem = Poisson2D()
    data = problem.sample(n_interior=64, n_boundary=32, n_initial=0, device=CPU)
    terms = problem.losses(Analytic(_sin_sin), data)
    assert terms["pde"].item() < 1e-8
    assert terms["bc"].item() < 1e-12


def test_convection_diffusion_residual_vanishes_for_non_default_coefficients():
    # Non-default beta/eps: catches a source term that quietly uses module-level
    # constants instead of the instance's own coefficients.
    problem = ConvectionDiffusion2D(beta=(2.0, 0.5), eps=0.3)
    data = problem.sample(n_interior=64, n_boundary=32, n_initial=0, device=CPU)
    terms = problem.losses(Analytic(_sin_sin), data)
    assert terms["pde"].item() < 1e-8
    assert terms["bc"].item() < 1e-12


def test_navier_stokes_residuals_match_the_hand_derived_expressions():
    # A polynomial field with known derivatives, whose residuals are worked out
    # by hand below. It is not a solution -- that is the point: the test checks
    # the operator, not just that a solution scores zero.
    #
    #   u = x^2 y,  v = x y^2,  p = x + y
    #   cont  = u_x + v_y            = 2xy + 2xy          = 4xy
    #   mom_x = u u_x + v u_y + p_x - nu (u_xx + u_yy)
    #         = 2x^3y^2 + x^3y^2 + 1 - 2 nu y            = 3x^3y^2 + 1 - 2 nu y
    #   mom_y = u v_x + v v_y + p_y - nu (v_xx + v_yy)
    #         = x^2y^3 + 2x^2y^3 + 1 - 2 nu x            = 3x^2y^3 + 1 - 2 nu x
    problem = LidDrivenCavity()
    data = problem.sample(n_interior=200, n_boundary=8, n_initial=0, device=CPU)

    def field(coords):
        x, y = coords[:, 0:1], coords[:, 1:2]
        return torch.cat([x**2 * y, x * y**2, x + y], dim=1)

    terms = problem.losses(Analytic(field), data)
    x = data["interior"].coords["x"]
    y = data["interior"].coords["y"]
    nu = problem.nu

    assert torch.allclose(terms["cont"], torch.mean((4.0 * x * y) ** 2), atol=1e-12)
    assert torch.allclose(
        terms["mom_x"], torch.mean((3.0 * x**3 * y**2 + 1.0 - 2.0 * nu * y) ** 2), atol=1e-12
    )
    assert torch.allclose(
        terms["mom_y"], torch.mean((3.0 * x**2 * y**3 + 1.0 - 2.0 * nu * x) ** 2), atol=1e-12
    )


def test_burgers_residual_matches_the_hand_derived_expression():
    # u = x^2 t  =>  u_t = x^2,  u_x = 2xt,  u_xx = 2t
    #   residual = u_t + u u_x - nu u_xx = x^2 + 2x^3 t^2 - 2 nu t
    problem = Burgers1D()
    data = problem.sample(n_interior=200, n_boundary=8, n_initial=8, device=CPU)

    def field(coords):
        x, t = coords[:, 0:1], coords[:, 1:2]
        return x**2 * t

    terms = problem.losses(Analytic(field), data)
    x = data["interior"].coords["x"]
    t = data["interior"].coords["t"]
    nu = problem.nu
    expected = torch.mean((x**2 + 2.0 * x**3 * t**2 - 2.0 * nu * t) ** 2)
    assert torch.allclose(terms["pde"], expected, atol=1e-12)


def test_exact_solutions_are_reported_only_where_one_exists():
    assert set(Poisson2D.exact({"x": torch.rand(3, 1), "y": torch.rand(3, 1)})) == {"u"}
    assert ConvectionDiffusion2D.exact is not None
    assert LidDrivenCavity.exact({"x": torch.rand(3, 1), "y": torch.rand(3, 1)}) is None
    assert Burgers1D.exact({"x": torch.rand(3, 1), "t": torch.rand(3, 1)}) is None

