"""2-D wall shear must be inferred, not supplied as a prescribed u_tau."""

from __future__ import annotations

import math

import pytest

from RANS_PINN_Demo.wall_observables import (
    friction_velocity_from_wall_shear,
    wall_force_balance_residual,
    wall_shear_from_gradient,
    wall_shear_relative_error,
)


def test_wall_shear_and_friction_velocity_are_inferred_from_gradient():
    tau_w = wall_shear_from_gradient(0.5, 1.0)
    observation = friction_velocity_from_wall_shear(tau_w, density=2.0)
    assert tau_w == pytest.approx(0.5)
    assert observation.u_tau == pytest.approx(math.sqrt(0.25))
    assert observation.tau_w == pytest.approx(0.5)


def test_wall_shear_relative_error_and_force_balance_are_independent_checks():
    assert wall_shear_relative_error(0.95, 1.0) == pytest.approx(0.05)
    assert wall_force_balance_residual(1.0, -0.99) == pytest.approx(0.01)
    assert wall_force_balance_residual(0.0, 0.0) == 0.0


def test_wall_shear_inputs_reject_nonphysical_values():
    with pytest.raises(ValueError):
        wall_shear_from_gradient(0.0, 1.0)
    with pytest.raises(ValueError):
        wall_shear_from_gradient(1.0, float("nan"))
    with pytest.raises(ValueError):
        wall_shear_relative_error(1.0, 0.0)