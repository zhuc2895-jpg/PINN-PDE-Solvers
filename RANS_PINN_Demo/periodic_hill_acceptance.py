"""Executable acceptance for the six pre-training periodic-hill solver gates.

The thresholds in this module are copied from section 4.4 of
``docs/2d-periodic-hill-plan.md``.  The runner deliberately fails closed:
missing data, non-finite fields, missing separation/reattachment crossings, or
a failed reference-data lookup cannot silently pass a gate.

The six gates are evaluated from one primary solve per grid plus two flat-plate
verification solves.  This avoids re-running a grid for every check and makes
the JSON report a direct record of the fields that were judged.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .periodic_hill import (
    BULK_VELOCITY,
    HILL_HEIGHT,
    PERIOD,
    PeriodicHillProfile,
    hill_height,
    load_reference,
)
from .periodic_hill_observables import (
    force_balance,
    reference_wall_shear_fit,
    separation_reattachment,
    wall_shear_profile,
)
from .periodic_hill_solver import PeriodicHillSolution, PeriodicHillSolver

__all__ = [
    "DEFAULT_GRIDS",
    "DEFAULT_REFERENCE_DIR",
    "FLAT_PLATE_RELATIVE_ERROR_LIMIT",
    "GRID_REATTACHMENT_LIMIT",
    "GRID_RELATIVE_L2_LIMIT",
    "MASS_DIVERGENCE_COEFFICIENT",
    "REFERENCE_POSITION_TOLERANCE",
    "FORCE_BALANCE_LIMIT",
    "GateResult",
    "AcceptanceReport",
    "SixGateInputs",
    "reference_separation_reattachment",
    "flat_plate_limit_gate",
    "mass_conservation_gate",
    "periodicity_gate",
    "separation_gate",
    "grid_sensitivity_gate",
    "force_balance_gate",
    "evaluate_six_gates",
    "run_solver_acceptance",
    "main",
]

DEFAULT_GRIDS: Mapping[str, Tuple[int, int]] = {
    "G1": (48, 24),
    "G2": (96, 48),
    "G3": (192, 96),
}
DEFAULT_REFERENCE_DIR = Path("RANS_PINN_Demo/reference/periodic_hill/extracted")
DEFAULT_OUTPUT = Path("RANS_PINN_Demo/results/periodic_hill_acceptance.json")

FLAT_PLATE_RELATIVE_ERROR_LIMIT = 0.02
MASS_DIVERGENCE_COEFFICIENT = 1e-8
REFERENCE_POSITION_TOLERANCE = 2.0 * HILL_HEIGHT
GRID_REATTACHMENT_LIMIT = 1.0 * HILL_HEIGHT
GRID_RELATIVE_L2_LIMIT = 0.10
FORCE_BALANCE_LIMIT = 0.05


def _safe_float(value: object) -> Optional[float]:
    """Return a JSON-safe float, converting non-finite values to ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_array(value: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(value, dtype=float))))


@dataclass(frozen=True)
class GateResult:
    """One pre-registered solver gate and the measurements that decided it."""

    key: str
    title: str
    passed: bool
    metrics: Dict[str, Any]
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "passed": bool(self.passed),
            "detail": self.detail,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class AcceptanceReport:
    """Complete six-gate report."""

    config: Dict[str, Any]
    run_status: Dict[str, str]
    gates: Tuple[GateResult, ...]

    @property
    def run_status_ok(self) -> bool:
        required = {"flat_G1", "flat_G2", "G1", "G2", "G3"}
        return required.issubset(self.run_status) and all(
            status == "converged" for status in self.run_status.values()
        )

    @property
    def passed(self) -> bool:
        return (
            len(self.gates) == 6
            and all(gate.passed for gate in self.gates)
            and self.run_status_ok
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": "periodic-hill-solver-acceptance/v1",
            "passed": self.passed,
            "run_status_ok": self.run_status_ok,
            "config": self.config,
            "run_status": self.run_status,
            "gates": [gate.as_dict() for gate in self.gates],
        }

    def write_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def to_text(self) -> str:
        lines = ["periodic-hill solver acceptance: " + ("PASS" if self.passed else "FAIL")]
        for gate in self.gates:
            lines.append(
                "{key}: {status} -- {detail}".format(
                    key=gate.key, status="PASS" if gate.passed else "FAIL", detail=gate.detail
                )
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class SixGateInputs:
    """Fields and diagnostics required to evaluate the six gates without rerunning."""

    flat_solutions: Mapping[str, PeriodicHillSolution]
    hill_solutions: Mapping[str, PeriodicHillSolution]
    max_divergence: Mapping[str, float]
    reference_profiles: Mapping[float, PeriodicHillProfile]
    dt: float
    run_status: Mapping[str, str] = field(default_factory=dict)


def _crossing(x0: float, value0: float, x1: float, value1: float) -> float:
    return float(x0 - value0 * (x1 - x0) / (value1 - value0))


def reference_separation_reattachment(
    profiles: Mapping[float, PeriodicHillProfile],
) -> Tuple[Optional[float], Optional[float]]:
    """Locate the reference bubble from the independent LES wall-shear fit."""

    if not profiles:
        return None, None
    x = np.array(sorted(profiles), dtype=float)
    try:
        tau = np.array(
            [reference_wall_shear_fit(profiles[float(xi)]) for xi in x], dtype=float
        )
    except (TypeError, ValueError):
        return None, None
    separation: Optional[float] = None
    reattachment: Optional[float] = None
    for index in range(x.size - 1):
        left, right = float(tau[index]), float(tau[index + 1])
        if not (math.isfinite(left) and math.isfinite(right)):
            continue
        if separation is None and left > 0.0 and right < 0.0:
            separation = _crossing(float(x[index]), left, float(x[index + 1]), right)
        elif separation is not None and reattachment is None and left < 0.0 and right > 0.0:
            reattachment = _crossing(float(x[index]), left, float(x[index + 1]), right)
            break
    return separation, reattachment


def flat_plate_limit_gate(
    solutions: Mapping[str, PeriodicHillSolution],
) -> GateResult:
    """Gate 1: G1 and G2 flat-channel profiles match the analytic parabola."""

    metrics: Dict[str, Any] = {}
    passed = True
    missing = [name for name in ("G1", "G2") if name not in solutions]
    if missing:
        return GateResult(
            "flat_plate",
            "flat-plate limit",
            False,
            {"missing_grids": missing},
            "missing flat-plate solves: " + ", ".join(missing),
        )
    for name in ("G1", "G2"):
        solution = solutions[name]
        y = np.asarray(solution.y_c, dtype=float)
        profile = np.asarray(solution.u, dtype=float).mean(axis=0)
        exact = float(solution.f0) / (2.0 * float(solution.nu)) * y * (
            float(solution.height) - y
        )
        denominator = float(np.max(np.abs(exact))) if exact.size else 0.0
        if denominator <= 0.0:
            error = float("inf")
        else:
            error = float(np.max(np.abs(profile - exact)) / denominator)
        finite = _finite_array(profile) and _finite_array(exact) and math.isfinite(error)
        reynolds_ok = math.isclose(float(solution.nu), 1.0, rel_tol=0.0, abs_tol=1e-12)
        gate_passed = finite and reynolds_ok and error < FLAT_PLATE_RELATIVE_ERROR_LIMIT
        passed = passed and gate_passed
        metrics[name] = {
            "max_relative_error": _safe_float(error),
            "reynolds_number": _safe_float(1.0 / float(solution.nu)),
            "reynolds_ok": bool(reynolds_ok),
            "passed": bool(gate_passed),
        }
    return GateResult(
        "flat_plate",
        "flat-plate limit",
        passed,
        metrics,
        "maximum relative error must be < 2% on G1 and G2 at Re=1",
    )


def mass_conservation_gate(
    max_divergence: Mapping[str, float], *, dt: float
) -> GateResult:
    """Gate 2: every post-projection divergence is below ``1e-8/dt``."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    threshold = MASS_DIVERGENCE_COEFFICIENT / float(dt)
    metrics: Dict[str, Any] = {"threshold": threshold}
    required = ("G1", "G2", "G3")
    missing = [name for name in required if name not in max_divergence]
    metrics["missing_grids"] = missing
    passed = not missing
    for name, value in max_divergence.items():
        finite = math.isfinite(float(value))
        gate_passed = finite and float(value) < threshold
        passed = passed and gate_passed
        metrics[name] = {
            "max_divergence": _safe_float(value),
            "passed": bool(gate_passed),
        }
    return GateResult(
        "mass_conservation",
        "mass conservation",
        passed,
        metrics,
        "each post-projection max divergence must be < 1e-8/dt",
    )


def _actual_x_operator_residual(
    solution: PeriodicHillSolution, field: np.ndarray
) -> float:
    """Check roll-equivariance of the solver's actual x derivative.

    A one-sided implementation changes when the same field is rolled by one
    cell; the solver's cyclic implementation does not.  This tests the actual
    ``PeriodicHillSolver`` stencil rather than redefining a second operator in
    the acceptance module.
    """

    values = np.asarray(field, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        return float("inf")
    solver = PeriodicHillSolver(
        nx=values.shape[0],
        ny=np.asarray(solution.pressure).shape[1],
        length=float(solution.length),
        height=float(solution.height),
        re_h=1.0 / float(solution.nu),
        mixing_length=False,
        dt=1.0,
        controller=False,
    )
    velocity = np.ones_like(values)
    derivative = solver._x_derivative(values, velocity, float(solution.dx))
    shifted = solver._x_derivative(
        np.roll(values, 1, axis=0), np.roll(velocity, 1, axis=0), float(solution.dx)
    )
    commutator = shifted - np.roll(derivative, 1, axis=0)
    scale = max(1.0, float(np.max(np.abs(derivative))))
    return float(np.max(np.abs(commutator)) / scale)


def periodicity_gate(
    solutions: Mapping[str, PeriodicHillSolution],
) -> GateResult:
    """Gate 3: every field has a periodic x discretization and periodic geometry."""

    metrics: Dict[str, Any] = {}
    passed = True
    missing = [name for name in ("G1", "G2", "G3") if name not in solutions]
    if missing:
        return GateResult(
            "periodicity",
            "periodicity",
            False,
            {"missing_grids": missing},
            "missing primary solves: " + ", ".join(missing),
        )
    # Check the published period map at the two representations of the seam.
    geometry_residual = abs(
        float(hill_height(np.array([PERIOD]))[0])
        - float(hill_height(np.array([0.0]))[0])
    )
    metrics["geometry_period_residual"] = _safe_float(geometry_residual)
    if geometry_residual > 1e-12:
        passed = False
    for name in ("G1", "G2", "G3"):
        solution = solutions[name]
        residual = 0.0
        finite = True
        for field_name in ("u", "v", "pressure"):
            values = np.asarray(getattr(solution, field_name), dtype=float)
            field_finite = _finite_array(values)
            finite = finite and field_finite
            if field_finite:
                residual = max(residual, _actual_x_operator_residual(solution, values))
            else:
                residual = float("nan")
        x = np.asarray(solution.x_u, dtype=float)
        expected_x = np.arange(x.size, dtype=float) * float(solution.dx)
        coordinate_residual = float(np.max(np.abs(x - expected_x))) if x.size else float("inf")
        coordinate_ok = (
            x.size > 0
            and coordinate_residual <= 1e-12
            and abs(float(x[-1]) - (float(solution.length) - float(solution.dx))) <= 1e-12
        )
        gate_passed = (
            finite
            and math.isfinite(residual)
            and residual <= 1e-12
            and coordinate_ok
        )
        passed = passed and gate_passed
        metrics[name] = {
            "periodic_operator_residual": _safe_float(residual),
            "coordinate_residual": _safe_float(coordinate_residual),
            "finite_fields": bool(finite),
            "passed": bool(gate_passed),
        }
    return GateResult(
        "periodicity",
        "periodicity",
        passed,
        metrics,
        "x operators must be cyclic and the hill geometry must close at x=0/Lx",
    )


def separation_gate(
    solution: PeriodicHillSolution,
    reference_profiles: Mapping[float, PeriodicHillProfile],
) -> GateResult:
    """Gate 4: G2 has a bubble within +/-2h of the independent LES positions."""

    reference_separation, reference_reattachment = reference_separation_reattachment(
        reference_profiles
    )
    separation, reattachment = separation_reattachment(solution)
    metrics: Dict[str, Any] = {
        "reference_separation": _safe_float(reference_separation)
        if reference_separation is not None
        else None,
        "reference_reattachment": _safe_float(reference_reattachment)
        if reference_reattachment is not None
        else None,
        "separation": _safe_float(separation) if separation is not None else None,
        "reattachment": _safe_float(reattachment) if reattachment is not None else None,
        "position_tolerance": REFERENCE_POSITION_TOLERANCE,
        "separation_domain_limit": 8.0 * HILL_HEIGHT,
    }
    if reference_separation is None or reference_reattachment is None:
        return GateResult(
            "separation_reattachment",
            "separation/reattachment",
            False,
            metrics,
            "independent reference does not bracket one separation and one reattachment",
        )
    if separation is None or reattachment is None:
        return GateResult(
            "separation_reattachment",
            "separation/reattachment",
            False,
            metrics,
            "G2 does not contain both a separation and a reattachment crossing",
        )
    separation_error = abs(float(separation) - float(reference_separation))
    reattachment_error = abs(float(reattachment) - float(reference_reattachment))
    domain_ok = 0.0 <= float(separation) <= 8.0 * HILL_HEIGHT and 0.0 <= float(
        reattachment
    ) <= PERIOD
    passed = (
        domain_ok
        and separation_error <= REFERENCE_POSITION_TOLERANCE
        and reattachment_error <= REFERENCE_POSITION_TOLERANCE
    )
    metrics["separation_error"] = _safe_float(separation_error)
    metrics["reattachment_error"] = _safe_float(reattachment_error)
    metrics["domain_ok"] = bool(domain_ok)
    return GateResult(
        "separation_reattachment",
        "separation/reattachment",
        passed,
        metrics,
        "G2 must bracket a bubble within +/-2h of the independent reference",
    )


def _periodic_column(at_x: float, x: np.ndarray, values: np.ndarray, length: float) -> np.ndarray:
    extended_x = np.concatenate([x, np.array([x[0] + length], dtype=float)])
    extended_values = np.concatenate([values, values[:1, :]], axis=0)
    return np.array(
        [np.interp(at_x, extended_x, extended_values[:, j]) for j in range(extended_values.shape[1])],
        dtype=float,
    )


def _velocity_profile_relative_l2(
    coarse: PeriodicHillSolution, fine: PeriodicHillSolution
) -> float:
    coarse_x = np.asarray(coarse.x_u, dtype=float)
    fine_x = np.asarray(fine.x_u, dtype=float)
    coarse_y = np.asarray(coarse.y_c, dtype=float)
    fine_y = np.asarray(fine.y_c, dtype=float)
    coarse_u = np.asarray(coarse.u, dtype=float)
    fine_u = np.asarray(fine.u, dtype=float)
    coarse_at_fine = np.empty_like(fine_u, dtype=float)
    for index, x_value in enumerate(fine_x):
        coarse_column = _periodic_column(
            float(x_value), coarse_x, coarse_u, float(coarse.length)
        )
        coarse_at_fine[index, :] = np.interp(fine_y, coarse_y, coarse_column)
    barrier = hill_height(fine_x)
    mask = np.broadcast_to(
        fine_y[None, :] > barrier[:, None], fine_u.shape
    ).copy()
    fine_values = fine_u[mask]
    coarse_values = coarse_at_fine[mask]
    denominator = float(np.linalg.norm(fine_values))
    if denominator <= 0.0:
        return float("inf")
    return float(np.linalg.norm(fine_values - coarse_values) / denominator)


def grid_sensitivity_gate(
    g2: PeriodicHillSolution, g3: PeriodicHillSolution
) -> GateResult:
    """Gate 5: G2/G3 reattachment differs by <1h and mainstream L2 is <10%."""

    _, g2_reattachment = separation_reattachment(g2)
    _, g3_reattachment = separation_reattachment(g3)
    metrics: Dict[str, Any] = {
        "g2_reattachment": _safe_float(g2_reattachment)
        if g2_reattachment is not None
        else None,
        "g3_reattachment": _safe_float(g3_reattachment)
        if g3_reattachment is not None
        else None,
        "reattachment_limit": GRID_REATTACHMENT_LIMIT,
        "relative_l2_limit": GRID_RELATIVE_L2_LIMIT,
    }
    if g2_reattachment is None or g3_reattachment is None:
        metrics["reattachment_difference"] = None
        metrics["relative_l2"] = None
        return GateResult(
            "grid_sensitivity",
            "grid sensitivity",
            False,
            metrics,
            "both G2 and G3 must have an in-cycle reattachment location",
        )
    reattachment_difference = abs(float(g2_reattachment) - float(g3_reattachment))
    relative_l2 = _velocity_profile_relative_l2(g2, g3)
    metrics["reattachment_difference"] = _safe_float(reattachment_difference)
    metrics["relative_l2"] = _safe_float(relative_l2)
    passed = (
        math.isfinite(reattachment_difference)
        and math.isfinite(relative_l2)
        and reattachment_difference < GRID_REATTACHMENT_LIMIT
        and relative_l2 < GRID_RELATIVE_L2_LIMIT
    )
    return GateResult(
        "grid_sensitivity",
        "grid sensitivity",
        passed,
        metrics,
        "G2/G3 reattachment separation <1h and full-fluid relative L2 <10%",
    )


def force_balance_gate(solution: PeriodicHillSolution) -> GateResult:
    """Gate 6: G2 satisfies the periodic integral streamwise force balance."""

    budget = force_balance(solution)
    local = wall_shear_profile(solution)
    residual = float(budget.residual)
    passed = math.isfinite(residual) and residual < FORCE_BALANCE_LIMIT
    metrics = {
        "body": _safe_float(budget.body),
        "pressure_hill": _safe_float(budget.pressure_hill),
        "shear_wall": _safe_float(budget.shear_wall),
        "residual": _safe_float(residual),
        "limit": FORCE_BALANCE_LIMIT,
        "tau_w_x_over_h": [_safe_float(value) for value in local.x_over_h],
        "tau_w": [_safe_float(value) for value in local.tau_w],
        "u_tau": [_safe_float(value) for value in local.u_tau],
    }
    return GateResult(
        "force_balance",
        "force balance",
        passed,
        metrics,
        "G2 periodic streamwise force-balance residual must be <5%",
    )


def _missing_gate(key: str, title: str, reason: str) -> GateResult:
    return GateResult(key, title, False, {"error": reason}, reason)


def evaluate_six_gates(inputs: SixGateInputs) -> AcceptanceReport:
    """Evaluate all six gates in the pre-registered order."""

    gates = [
        flat_plate_limit_gate(inputs.flat_solutions),
        mass_conservation_gate(inputs.max_divergence, dt=inputs.dt),
        periodicity_gate(inputs.hill_solutions),
    ]
    g2 = inputs.hill_solutions.get("G2")
    g3 = inputs.hill_solutions.get("G3")
    if g2 is None:
        gates.append(
            _missing_gate(
                "separation_reattachment",
                "separation/reattachment",
                "missing G2 solve",
            )
        )
    else:
        gates.append(separation_gate(g2, inputs.reference_profiles))
    if g2 is None or g3 is None:
        gates.append(
            _missing_gate(
                "grid_sensitivity",
                "grid sensitivity",
                "missing G2 or G3 solve",
            )
        )
    else:
        gates.append(grid_sensitivity_gate(g2, g3))
    if g2 is None:
        gates.append(_missing_gate("force_balance", "force balance", "missing G2 solve"))
    else:
        gates.append(force_balance_gate(g2))
    config = {
        "dt": float(inputs.dt),
        "flat_plate_error_limit": FLAT_PLATE_RELATIVE_ERROR_LIMIT,
        "mass_divergence_coefficient": MASS_DIVERGENCE_COEFFICIENT,
        "separation_tolerance": REFERENCE_POSITION_TOLERANCE,
        "grid_reattachment_limit": GRID_REATTACHMENT_LIMIT,
        "grid_relative_l2_limit": GRID_RELATIVE_L2_LIMIT,
        "force_balance_limit": FORCE_BALANCE_LIMIT,
    }
    return AcceptanceReport(config=config, run_status=dict(inputs.run_status), gates=tuple(gates))


def _snapshot(
    solver: PeriodicHillSolver,
    *,
    q_history: Sequence[float],
    change_history: Sequence[float],
    iterations: int,
    status: str,
) -> PeriodicHillSolution:
    return PeriodicHillSolution(
        u=solver.u.copy(),
        v=solver.v.copy(),
        pressure=solver.pressure.copy(),
        x_u=solver.x_u.copy(),
        y_c=solver.y_c.copy(),
        dx=solver.dx,
        dy=solver.dy,
        f0=solver.f0,
        nu=solver.nu,
        length=solver.length,
        height=solver.height,
        q_history=list(q_history),
        change_history=list(change_history),
        iterations=iterations,
        converged=status == "converged",
    )


def _solver_divergence(solver: PeriodicHillSolver) -> float:
    divergence = (
        (solver.u[solver.ip, :] - solver.u) / solver.dx
        + (solver.v[:, 1:] - solver.v[:, :-1]) / solver.dy
    )
    fluid = np.asarray(solver.chi_c, dtype=float) < 0.5
    if not np.any(fluid):
        return 0.0
    values = np.abs(np.asarray(divergence, dtype=float)[fluid])
    if not np.all(np.isfinite(values)):
        return float("nan")
    return float(np.max(values))


def _march(
    solver: PeriodicHillSolver,
    *,
    initial_u: float,
    max_steps: int,
    tol: float,
    patience: int,
) -> Tuple[PeriodicHillSolution, float, str]:
    solver.u[:] = float(initial_u)
    q_history = []
    change_history = []
    max_divergence = 0.0
    stable = 0
    status = "max_iter"
    iterations = 0
    for step in range(max_steps):
        iterations = step + 1
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            q, change = solver.step()
        q_history.append(float(q))
        change_history.append(float(change))
        if not (
            math.isfinite(float(q))
            and math.isfinite(float(change))
            and _finite_array(solver.u)
            and _finite_array(solver.v)
            and _finite_array(solver.pressure)
        ):
            status = "diverged"
            break
        divergence = _solver_divergence(solver)
        if not math.isfinite(divergence):
            status = "diverged"
            break
        max_divergence = max(max_divergence, divergence)
        if math.isfinite(change) and change < tol:
            stable += 1
        else:
            stable = 0
        if stable >= patience:
            status = "converged"
            break
    return (
        _snapshot(
            solver,
            q_history=q_history,
            change_history=change_history,
            iterations=iterations,
            status=status,
        ),
        max_divergence,
        status,
    )


def _flat_solver(nx: int, ny: int, dt: float) -> PeriodicHillSolver:
    solver = PeriodicHillSolver(
        nx=nx,
        ny=ny,
        re_h=1.0,
        mixing_length=False,
        dt=dt,
        f0=1.0,
        controller=False,
    )
    solver.b_u[:] = 0.0
    solver.b_v[:] = 0.0
    solver.b_c[:] = 0.0
    solver.chi_u[:] = 0.0
    solver.chi_v[:] = 0.0
    solver.chi_c[:] = 0.0
    return solver


def run_solver_acceptance(
    reference_directory: Path,
    *,
    output: Optional[Path] = None,
    max_steps: int = 14000,
    flat_max_steps: int = 12000,
    hill_dt: float = 0.01,
    flat_dt: float = 0.001,
    tol: float = 1e-7,
    patience: int = 20,
) -> AcceptanceReport:
    """Run the baseline once per grid and evaluate all six gates."""

    profiles = load_reference(Path(reference_directory))
    flat_solutions: Dict[str, PeriodicHillSolution] = {}
    hill_solutions: Dict[str, PeriodicHillSolution] = {}
    max_divergence: Dict[str, float] = {}
    run_status: Dict[str, str] = {}

    for name in ("G1", "G2"):
        nx, ny = DEFAULT_GRIDS[name]
        flat_solver = _flat_solver(nx, ny, flat_dt)
        flat_solution, _, flat_status = _march(
            flat_solver,
            initial_u=0.0,
            max_steps=flat_max_steps,
            tol=tol,
            patience=patience,
        )
        flat_solutions[name] = flat_solution
        run_status["flat_" + name] = flat_status

    for name, (nx, ny) in DEFAULT_GRIDS.items():
        solver = PeriodicHillSolver(
            nx=nx,
            ny=ny,
            dt=hill_dt,
            f0=0.036,
            controller=False,
            mixing_length=True,
        )
        solution, divergence, status = _march(
            solver,
            initial_u=BULK_VELOCITY,
            max_steps=max_steps,
            tol=tol,
            patience=patience,
        )
        hill_solutions[name] = solution
        max_divergence[name] = divergence
        run_status[name] = status

    config = {
        "reference_directory": str(Path(reference_directory)),
        "max_steps": int(max_steps),
        "flat_max_steps": int(flat_max_steps),
        "hill_dt": float(hill_dt),
        "flat_dt": float(flat_dt),
        "tol": float(tol),
        "patience": int(patience),
        "grids": {name: {"nx": nx, "ny": ny} for name, (nx, ny) in DEFAULT_GRIDS.items()},
    }
    report = evaluate_six_gates(
        SixGateInputs(
            flat_solutions=flat_solutions,
            hill_solutions=hill_solutions,
            max_divergence=max_divergence,
            reference_profiles=profiles,
            dt=hill_dt,
            run_status=run_status,
        )
    )
    report.config.update(config)
    if output is not None:
        report.write_json(Path(output))
    return report


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="extracted ERCOFTAC case-081 directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path",
    )
    parser.add_argument("--max-steps", type=int, default=14000)
    parser.add_argument("--flat-max-steps", type=int, default=12000)
    parser.add_argument("--hill-dt", type=float, default=0.01)
    parser.add_argument("--flat-dt", type=float, default=0.001)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        report = run_solver_acceptance(
            args.reference_dir,
            output=args.output,
            max_steps=args.max_steps,
            flat_max_steps=args.flat_max_steps,
            hill_dt=args.hill_dt,
            flat_dt=args.flat_dt,
            tol=args.tol,
            patience=args.patience,
        )
    except FileNotFoundError as exc:
        print(str(exc))
        print(
            "Fetch the independent reference first: "
            "python -m RANS_PINN_Demo.fetch_periodic_hill_reference"
        )
        return 2
    print(report.to_text())
    print("JSON report: " + str(args.output))
    return 0 if report.passed else 1
