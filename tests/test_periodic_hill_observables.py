"""T10.2: independent 2-D wall and integral observables.

Wall shear is computed from the *solved* wall-normal velocity gradient and the
effective viscosity, with ``u_tau`` inferred as an output (ADR-0003).  The
integral force balance is a separate check; the flat-plate limit is the
regression that ties the reporter to the discrete scheme.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from RANS_PINN_Demo.periodic_hill import KINEMATIC_VISCOSITY, PeriodicHillProfile
from RANS_PINN_Demo.periodic_hill_observables import (
    ForceBudget,
    StationObservation,
    force_balance,
    friction_velocity_profile,
    reference_wall_shear_fit,
    separation_reattachment,
    wall_shear_observations,
    wall_shear_profile,
)


def _flat_solution(
    *,
    nx: int = 8,
    ny: int = 6,
    dx: float = 0.5,
    dy: float = 0.5,
    nu: float = 0.5,
    f0: float = 1.0,
    mixing_length: bool = False,
):
    """A genuinely flat channel field object (downstream flat run, b=0)."""
    x_u = 2.0 + np.arange(nx) * dx  # stays inside the flat run b(x)=0
    y_c = (np.arange(ny) + 0.5) * dy
    return SimpleNamespace(
        u=np.zeros((nx, ny)),
        v=np.zeros((nx, ny + 1)),
        pressure=np.zeros((nx, ny)),
        x_u=x_u,
        y_c=y_c,
        dx=dx,
        dy=dy,
        nu=nu,
        f0=f0,
        length=nx * dx,
        height=ny * dy,
        mixing_length=mixing_length,
    )


def _parabola(solution, forcing: float):
    """Fill ``u`` with the analytic pressure-driven channel parabola."""
    y = solution.y_c
    solution.u[:] = forcing / (2.0 * solution.nu) * (y * (solution.height - y))
    return forcing * solution.height / 2.0  # exact wall shear stress


def _reference_profile(y_over_h, u_over_ub, x_over_h=5.0):
    zeros = np.zeros_like(np.asarray(y_over_h, dtype=float))
    return PeriodicHillProfile(
        x_over_h=x_over_h,
        y_over_h=np.asarray(y_over_h, dtype=float),
        u_over_ub=np.asarray(u_over_ub, dtype=float),
        v_over_ub=zeros,
        uu_over_ub2=zeros,
        vv_over_ub2=zeros,
        uv_over_ub2=zeros,
        k_over_ub2=zeros,
    )


# ---------------------------------------------------------------------------
# Local wall shear and inferred friction velocity
# ---------------------------------------------------------------------------
def test_wall_shear_from_gradient_and_inferred_u_tau():
    """An analytic parabola has an exactly recoverable wall gradient."""
    solution = _flat_solution(nu=0.5)
    tau_exact = _parabola(solution, forcing=2.0)
    profile = wall_shear_profile(solution)

    assert profile.tau_w.shape == (solution.u.shape[0],)
    assert np.allclose(profile.tau_w, tau_exact)
    assert np.allclose(profile.u_tau, np.sqrt(np.abs(profile.tau_w)))


def test_friction_velocity_profile_matches_scalar_inverse():
    tau = np.array([0.25, -0.25, 4.0])
    assert np.allclose(friction_velocity_profile(tau), [0.5, 0.5, 2.0])


def test_wall_shear_scales_with_effective_viscosity():
    solution = _flat_solution()
    solution.u[:, 0] = 0.3
    baseline = wall_shear_profile(solution)
    doubled = wall_shear_profile(
        solution, wall_effective_viscosity=np.full(solution.u.shape[0], 2.0 * solution.nu)
    )
    assert np.allclose(doubled.tau_w, 2.0 * baseline.tau_w)


def test_wall_shear_rejects_bad_effective_viscosity_shape():
    solution = _flat_solution()
    solution.u[:, 0] = 0.3
    with pytest.raises(ValueError):
        wall_shear_profile(solution, wall_effective_viscosity=np.ones(3))


# ---------------------------------------------------------------------------
# Independent LES reference wall shear
# ---------------------------------------------------------------------------
def test_reference_wall_shear_fit_is_independent_linear_slope():
    profile = _reference_profile([0.0, 0.1, 0.5], [0.0, 0.2, 0.9])
    assert reference_wall_shear_fit(profile, wall_normal_window=0.15) == pytest.approx(
        KINEMATIC_VISCOSITY * 2.0
    )


def test_reference_wall_shear_fit_recovers_analytic_slope():
    eta = np.array([0.0, 0.05, 0.1, 0.2, 0.4])
    slope = 3.0
    u = slope * eta + 0.5 * eta**2
    profile = _reference_profile(eta, u)
    assert reference_wall_shear_fit(profile, wall_normal_window=0.15) == pytest.approx(
        KINEMATIC_VISCOSITY * slope, rel=1e-6
    )


# ---------------------------------------------------------------------------
# Per-station error
# ---------------------------------------------------------------------------
def test_wall_shear_observations_report_relative_error_and_u_tau():
    solution = _flat_solution(nu=1.0)
    tau_pred = _parabola(solution, forcing=2.0)
    reference = _reference_profile([0.0, 0.5, 1.0], [0.0, 0.4, 0.8], x_over_h=5.0)
    tau_ref = KINEMATIC_VISCOSITY * 0.8

    observations = wall_shear_observations(solution, {5.0: reference}, stations=(5.0,))
    assert len(observations) == 1
    obs = observations[0]
    assert isinstance(obs, StationObservation)
    assert obs.tau_w_reference == pytest.approx(tau_ref)
    assert obs.tau_w_rans == pytest.approx(tau_pred)
    assert obs.relative_error == pytest.approx(abs(tau_pred - tau_ref) / abs(tau_ref))
    assert obs.u_tau_rans == pytest.approx(math.sqrt(abs(tau_pred)))
    assert obs.u_tau_reference == pytest.approx(math.sqrt(abs(tau_ref)))


def test_wall_shear_observations_skip_stations_without_reference():
    solution = _flat_solution()
    solution.u[:, 0] = 0.5
    assert wall_shear_observations(solution, {}, stations=(1.0,)) == []


# ---------------------------------------------------------------------------
# Integral force balance
# ---------------------------------------------------------------------------
def test_force_balance_flat_plate_limit_vanishes():
    """Flat plate: integrated wall shear balances the body force to machine zero."""
    solution = _flat_solution(nu=0.5, f0=2.0)
    _parabola(solution, forcing=solution.f0)

    budget = force_balance(solution)
    assert isinstance(budget, ForceBudget)
    assert budget.pressure_hill == pytest.approx(0.0)
    assert budget.shear_wall == pytest.approx(-budget.body, rel=1e-9)
    assert budget.residual < 1e-9


def test_hill_pressure_force_is_zero_for_uniform_pressure():
    solution = _flat_solution()
    solution.pressure[:] = 3.0
    assert force_balance(solution).pressure_hill == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Separation / reattachment
# ---------------------------------------------------------------------------
def test_separation_reattachment_sign_change_returns_positions():
    solution = _flat_solution(nx=6, ny=3, dx=1.0, dy=1.0, nu=1.0)
    # tau sign pattern along x: +, +, -, -, +, +
    solution.u[:, 0] = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    separation, reattachment = separation_reattachment(solution)
    assert separation == pytest.approx(3.5)
    assert reattachment == pytest.approx(5.5)


def test_separation_reattachment_returns_none_without_sign_change():
    solution = _flat_solution(nx=4)
    solution.u[:, 0] = 1.0
    assert separation_reattachment(solution) == (None, None)
