"""Contracts for the executable six-gate periodic-hill acceptance."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from RANS_PINN_Demo.periodic_hill import KINEMATIC_VISCOSITY, PeriodicHillProfile
from RANS_PINN_Demo.periodic_hill_acceptance import (
    AcceptanceReport,
    GateResult,
    flat_plate_limit_gate,
    force_balance_gate,
    grid_sensitivity_gate,
    mass_conservation_gate,
    periodicity_gate,
    reference_separation_reattachment,
    separation_gate,
)
from RANS_PINN_Demo.periodic_hill_solver import PeriodicHillSolution, PeriodicHillSolver


def _solution(
    *,
    nx: int = 6,
    ny: int = 3,
    dx: float = 1.0,
    dy: float = 1.0,
    u: Optional[np.ndarray] = None,
    y_c: Optional[np.ndarray] = None,
    f0: float = 1.0,
    nu: float = 1.0,
    length: float = 9.0,
    height: float = 3.0,
) -> PeriodicHillSolution:
    if u is None:
        u = np.zeros((nx, ny), dtype=float)
    return PeriodicHillSolution(
        u=np.asarray(u, dtype=float),
        v=np.zeros((nx, ny + 1), dtype=float),
        pressure=np.zeros((nx, ny), dtype=float),
        x_u=np.arange(nx, dtype=float) * dx,
        y_c=(
            np.asarray(y_c, dtype=float)
            if y_c is not None
            else (np.arange(ny, dtype=float) + 0.5) * dy
        ),
        dx=dx,
        dy=dy,
        f0=f0,
        nu=nu,
        length=length,
        height=height,
    )


def _profile(x: float, tau_w: float) -> PeriodicHillProfile:
    y = np.array([0.0, 0.1, 0.2])
    slope = tau_w / KINEMATIC_VISCOSITY
    return PeriodicHillProfile(
        x_over_h=x,
        y_over_h=y,
        u_over_ub=slope * y,
        v_over_ub=np.zeros_like(y),
        uu_over_ub2=np.zeros_like(y),
        vv_over_ub2=np.zeros_like(y),
        uv_over_ub2=np.zeros_like(y),
        k_over_ub2=np.zeros_like(y),
    )


def test_reference_separation_uses_independent_wall_shear():
    profiles = {
        2.0: _profile(2.0, 1.0),
        3.0: _profile(3.0, -1.0),
        4.0: _profile(4.0, -1.0),
        5.0: _profile(5.0, 1.0),
    }
    assert reference_separation_reattachment(profiles) == pytest.approx((2.5, 4.5))


def test_flat_plate_gate_accepts_exact_profiles_and_rejects_two_percent():
    ny = 4
    y = (np.arange(ny, dtype=float) + 0.5)
    exact = 1.0 / (2.0 * 1.0) * y * (3.0 - y)
    good = {
        "G1": _solution(ny=ny, y_c=y, u=np.tile(exact, (6, 1)), height=3.0),
        "G2": _solution(ny=ny, y_c=y, u=np.tile(exact, (6, 1)), height=3.0),
    }
    assert flat_plate_limit_gate(good).passed is True

    bad = {
        name: _solution(ny=ny, y_c=y, u=np.tile(exact * 1.021, (6, 1)), height=3.0)
        for name in good
    }
    result = flat_plate_limit_gate(bad)
    assert result.passed is False
    assert result.metrics["G1"]["max_relative_error"] > 0.02


def test_mass_conservation_gate_uses_the_strict_dt_scaled_threshold():
    values = {"G1": 1.0e-8, "G2": 9.9e-7, "G3": 1.0e-8}
    assert mass_conservation_gate(values, dt=0.01).passed is True
    values["G2"] = 1.0e-6
    assert mass_conservation_gate(values, dt=0.01).passed is False
    assert mass_conservation_gate({"G2": 1.0e-8}, dt=0.01).passed is False
    assert mass_conservation_gate(
        {"G1": 0.0, "G2": float("nan"), "G3": 0.0}, dt=0.01
    ).passed is False


def test_periodicity_gate_requires_finite_periodic_fields():
    solution = _solution(nx=10, ny=4, dx=0.9, length=9.0)
    assert periodicity_gate({"G1": solution, "G2": solution, "G3": solution}).passed
    solution.u[0, 0] = np.nan
    result = periodicity_gate({"G1": solution, "G2": solution, "G3": solution})
    assert result.passed is False


def test_periodicity_gate_rejects_a_one_sided_solver_operator(monkeypatch):
    def one_sided_derivative(self, field, velocity, dx):
        out = np.empty_like(field)
        out[0] = (field[1] - field[0]) / dx
        out[-1] = (field[-1] - field[-2]) / dx
        out[1:-1] = (field[2:] - field[:-2]) / (2.0 * dx)
        return out

    monkeypatch.setattr(PeriodicHillSolver, "_x_derivative", one_sided_derivative)
    solution = _solution(nx=10, ny=4, dx=0.9, length=9.0)
    phase = 2.0 * np.pi * np.arange(10, dtype=float) / 10.0
    solution.u[:] = np.sin(phase)[:, None]
    solution.v[:] = np.cos(phase)[:, None]
    solution.pressure[:] = np.sin(phase)[:, None]
    result = periodicity_gate({"G1": solution, "G2": solution, "G3": solution})
    assert result.passed is False


def test_separation_gate_compares_both_crossings_to_reference():
    y = np.array([0.0, 0.5, 1.0, 1.5])
    slopes = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    solution = _solution(nx=6, ny=4, u=slopes[:, None] * y[None, :], height=3.0)
    references = {
        2.0: _profile(2.0, 1.0),
        3.0: _profile(3.0, -1.0),
        4.0: _profile(4.0, -1.0),
        5.0: _profile(5.0, 1.0),
    }
    result = separation_gate(solution, references)
    assert result.passed is True
    assert result.metrics["separation"] == pytest.approx(1.5)
    assert result.metrics["reattachment"] == pytest.approx(3.5)


def test_grid_sensitivity_gate_rejects_profiles_above_ten_percent(monkeypatch):
    import RANS_PINN_Demo.periodic_hill_acceptance as acceptance

    monkeypatch.setattr(acceptance, "separation_reattachment", lambda solution: (2.0, 4.5))
    coarse = _solution(nx=6, ny=4, u=np.ones((6, 4)))
    fine = _solution(nx=6, ny=4, u=np.ones((6, 4)))
    assert grid_sensitivity_gate(coarse, fine).passed is True

    fine.u[:] = 1.12
    result = grid_sensitivity_gate(coarse, fine)
    assert result.passed is False
    assert result.metrics["relative_l2"] == pytest.approx(0.12 / 1.12, rel=1e-6)


def test_force_balance_gate_accepts_analytic_flat_channel():
    nx, ny = 8, 6
    dx = dy = 0.5
    y = (np.arange(ny, dtype=float) + 0.5) * dy
    u = np.tile(1.0 / (2.0 * 0.5) * y * (3.0 - y), (nx, 1))
    solution = _solution(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        u=u,
        f0=1.0,
        nu=0.5,
        length=nx * dx,
        height=ny * dy,
    )
    solution.x_u = 2.0 + np.arange(nx, dtype=float) * dx
    result = force_balance_gate(solution)
    assert result.passed is True
    assert result.metrics["residual"] < 0.05


def test_report_requires_all_solver_runs_to_converge():
    gates = tuple(
        GateResult(key=key, title=key, passed=True, metrics={}, detail="pass")
        for key in (
            "flat_plate",
            "mass_conservation",
            "periodicity",
            "separation_reattachment",
            "grid_sensitivity",
            "force_balance",
        )
    )
    statuses = {"flat_G1": "converged", "flat_G2": "converged", "G1": "converged", "G2": "converged", "G3": "converged"}
    assert AcceptanceReport({}, statuses, gates).passed is True
    statuses["G2"] = "max_iter"
    assert AcceptanceReport({}, statuses, gates).passed is False
