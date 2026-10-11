"""Independent wall and integral observables for the 2-D periodic-hill case.

This is the T10.2 observable seam.  It answers the two questions the 1-D channel
demo could not:

1. **Local wall shear.**  ``tau_w = (nu + nu_t)|_wall * (du/dn)|_wall`` is
   computed from the *solved* wall-normal velocity gradient and the effective
   viscosity at the wall.  The friction velocity ``u_tau = sqrt(abs(tau_w))``
   is an *output* of that computation -- no function here accepts a prescribed
   ``u_tau``, which is the ADR-0003 boundary against the 1-D construction
   identity ``tau_w = u_tau**2``.
2. **Integral force balance.**  For a periodic domain the streamwise momentum
   budget must close: ``F_body + F_pressure,hill + F_shear,wall = 0``.  The
   residual ``r_balance`` is reported *next to* the local wall shear, never in
   place of it.

The two are deliberately separate checks: a field can match the local wall
shear and still violate the integral balance, and vice versa.  Looking at both
is the point of the 2-D entry condition.

The wall-normal gradient is a local quadratic fit through the first fluid
samples evaluated at the wall; a raw two-point difference only sees the first
two cells and carries an O(dy) bias on the coarse grid.  The wall shear itself
is the physical quantity ``(nu + nu_t)|_wall * (du/dn)|_wall``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

import numpy as np

from .periodic_hill import KINEMATIC_VISCOSITY, PeriodicHillProfile, hill_height
from .wall_observables import friction_velocity_from_wall_shear, wall_shear_relative_error

__all__ = [
    "WallShearProfile",
    "StationObservation",
    "ForceBudget",
    "wall_shear_profile",
    "friction_velocity_profile",
    "reference_wall_shear_fit",
    "wall_shear_observations",
    "force_balance",
    "hill_wall_diagnostics",
    "separation_reattachment",
]


# ---------------------------------------------------------------------------
# Wall shear from a solved field
# ---------------------------------------------------------------------------
def _wall_gradient_from_solution(
    u: np.ndarray,
    y_c: np.ndarray,
    hill: np.ndarray,
    chi: np.ndarray,
    dy: float,
) -> np.ndarray:
    """One-sided wall-normal gradient ``du/dn`` at the lower wall.

    A local quadratic is fitted through the first three fluid samples and its
    slope at the wall is used; the raw two-point difference only sees the first
    two cells and carries an O(dy) bias on the coarse grid.  The wall sits
    ``dy/2`` below the first cell centre.  With fewer than three fluid samples
    the reflected-ghost estimate ``2 u_first / dy`` is used.
    """
    nx = u.shape[0]
    gradient = np.full(nx, np.nan)
    for i in range(nx):
        fluid = np.where((1.0 - chi[i]) > 0.5)[0]
        if fluid.size == 0:
            continue
        j0 = int(fluid[0])
        eta0 = y_c[j0] - hill[i]
        if fluid.size >= 3:
            j1, j2 = int(fluid[1]), int(fluid[2])
            eta = np.array([eta0, y_c[j1] - hill[i], y_c[j2] - hill[i]])
            samples = np.array([u[i, j0], u[i, j1], u[i, j2]])
            coeffs = np.polyfit(eta, samples, 2)
            gradient[i] = float(np.polyder(coeffs)[-1])
        else:
            gradient[i] = 2.0 * u[i, j0] / dy
    return gradient


@dataclass(frozen=True)
class WallShearProfile:
    """Per-station wall shear with the inferred friction velocity.

    ``u_tau`` is stored as a field of the result, never accepted as an input:
    that asymmetry is the ADR-0003 interface rule.
    """

    x_over_h: np.ndarray
    tau_w: np.ndarray
    u_tau: np.ndarray
    effective_viscosity: np.ndarray


def wall_shear_profile(
    solution: object,
    wall_effective_viscosity: Optional[np.ndarray] = None,
) -> WallShearProfile:
    """Compute ``tau_w`` and the inferred ``u_tau`` from a solved 2-D field.

    ``wall_effective_viscosity`` is ``(nu + nu_t)|_wall``; when omitted the
    molecular ``nu`` carried by the solution is used, which is exact for the
    baseline mixing-length closure (it enforces ``nu_t = 0`` at the wall).  A
    learned correction must pass its own wall value.  ``u_tau`` is derived from
    ``tau_w`` and is never an input.
    """
    u = np.asarray(solution.u, dtype=float)
    nx = u.shape[0]
    x = np.asarray(solution.x_u, dtype=float)
    y_c = np.asarray(solution.y_c, dtype=float)
    dy = float(solution.dy)
    nu = float(solution.nu)
    hill = hill_height(x)
    chi = (y_c[None, :] < hill[:, None]).astype(float)

    if wall_effective_viscosity is None:
        mu_wall = np.full(nx, nu, dtype=float)
    else:
        mu_wall = np.asarray(wall_effective_viscosity, dtype=float)
        if mu_wall.shape != (nx,):
            raise ValueError("wall_effective_viscosity must have shape (nx,)")
        if np.any(~np.isfinite(mu_wall)) or np.any(mu_wall <= 0.0):
            raise ValueError("wall_effective_viscosity must be finite and positive")

    gradient = _wall_gradient_from_solution(u, y_c, hill, chi, dy)
    tau_w = mu_wall * gradient
    u_tau = np.array(
        [
            friction_velocity_from_wall_shear(float(tau)).u_tau
            if math.isfinite(tau)
            else float("nan")
            for tau in tau_w
        ],
        dtype=float,
    )
    return WallShearProfile(
        x_over_h=x,
        tau_w=tau_w,
        u_tau=u_tau,
        effective_viscosity=mu_wall,
    )


def friction_velocity_profile(tau_w: np.ndarray) -> np.ndarray:
    """Infer ``u_tau = sqrt(abs(tau_w))`` from an already-computed stress field."""
    return np.sqrt(np.abs(np.asarray(tau_w, dtype=float)))


# ---------------------------------------------------------------------------
# Independent reference wall shear from the LES near-wall profile
# ---------------------------------------------------------------------------
def reference_wall_shear_fit(
    profile: PeriodicHillProfile,
    wall_normal_window: float = 0.15,
) -> float:
    """Fit ``tau_w`` from the LES near-wall profile, independently of RANS.

    A local polynomial is fitted to the near-wall samples in the LES wall-normal
    coordinate and its slope at the wall is multiplied by the molecular
    viscosity.  This uses only the independent reference data; nothing here
    touches the solver or its closure.  A linear fit (two samples) is the
    degenerate case; the default quadratic fit over the
    ``y - b(x) <= wall_normal_window`` band is more robust to free-stream
    curvature.
    """
    eta = np.asarray(profile.wall_normal, dtype=float)
    u = np.asarray(profile.u_over_ub, dtype=float)
    order = np.argsort(eta)
    eta = eta[order]
    u = u[order]
    band = eta <= float(wall_normal_window)
    if band.sum() < 2:
        band[:2] = True
    eta_w = eta[band]
    u_w = u[band]
    degree = min(2, eta_w.size - 1)
    coeffs = np.polyfit(eta_w, u_w, degree)
    slope = float(np.polyder(coeffs)[-1]) if coeffs.size > 1 else 0.0
    return KINEMATIC_VISCOSITY * slope


# ---------------------------------------------------------------------------
# Per-station observations and error
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StationObservation:
    """One wall station: local stress, inferred ``u_tau``, and reference error."""

    x_over_h: float
    tau_w_rans: float
    u_tau_rans: float
    tau_w_reference: float
    u_tau_reference: float
    relative_error: float


def _interp_profile(values: np.ndarray, x: np.ndarray, station: float) -> float:
    order = np.argsort(x)
    return float(np.interp(station, x[order], np.asarray(values)[order]))


def wall_shear_observations(
    solution: object,
    reference_profiles: Mapping[float, PeriodicHillProfile],
    *,
    stations: Optional[Tuple[float, ...]] = None,
    wall_effective_viscosity: Optional[np.ndarray] = None,
) -> List[StationObservation]:
    """Per-station ``e_tau = |tau_w^rans - tau_w^ref| / |tau_w^ref|``.

    The reference stress is fit independently from the LES near-wall profile.
    Only stations that have both a reference profile and a finite RANS stress
    are reported.
    """
    profile = wall_shear_profile(solution, wall_effective_viscosity)
    station_list = stations if stations is not None else tuple(sorted(reference_profiles))
    observations: List[StationObservation] = []
    for station in station_list:
        reference = reference_profiles.get(station)
        if reference is None:
            continue
        tau_rans = _interp_profile(profile.tau_w, profile.x_over_h, station)
        if not math.isfinite(tau_rans):
            continue
        tau_ref = reference_wall_shear_fit(reference)
        error = wall_shear_relative_error(tau_rans, tau_ref)
        observations.append(
            StationObservation(
                x_over_h=float(station),
                tau_w_rans=tau_rans,
                u_tau_rans=friction_velocity_from_wall_shear(tau_rans).u_tau,
                tau_w_reference=tau_ref,
                u_tau_reference=friction_velocity_from_wall_shear(tau_ref).u_tau,
                relative_error=error,
            )
        )
    return observations


def _wall_shear_forces(
    solution: object,
    wall_effective_viscosity: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Return the lower/upper wall shear force per unit width (retarding).

    Each wall force is ``-tau_w * ds`` with the same curvature-aware wall
    gradient the local observable reports.  The lower wall follows the hill, so
    its wetted perimeter is used; the upper wall is flat.
    """
    u = np.asarray(solution.u, dtype=float)
    x_u = np.asarray(solution.x_u, dtype=float)
    y_c = np.asarray(solution.y_c, dtype=float)
    dy = float(solution.dy)
    dx = float(solution.dx)
    nx, ny = u.shape
    height = float(getattr(solution, "height", 0.0)) or ny * dy
    b = hill_height(x_u)
    chi = (y_c[None, :] < b[:, None]).astype(float)

    if wall_effective_viscosity is None:
        mu_bottom = np.full(nx, float(solution.nu), dtype=float)
    else:
        mu_bottom = np.asarray(wall_effective_viscosity, dtype=float)
        if mu_bottom.shape != (nx,):
            raise ValueError("wall_effective_viscosity must have shape (nx,)")
        if np.any(~np.isfinite(mu_bottom)) or np.any(mu_bottom <= 0.0):
            raise ValueError("wall_effective_viscosity must be finite and positive")

    grad_bottom = _wall_gradient_from_solution(u, y_c, b, chi, dy)
    tau_bottom = mu_bottom * grad_bottom
    slope = np.gradient(b, x_u, axis=0)
    ds_bottom = dx * np.sqrt(1.0 + slope**2)
    lower = -float(np.nansum(tau_bottom * ds_bottom))

    # Upper wall: mirror the field so the same lower-wall routine sees a flat
    # wall at eta = 0; the outward normal flips, giving a retarding force with
    # the same sign as the lower wall.
    eta_top = height - y_c[::-1]
    grad_top = _wall_gradient_from_solution(
        u[:, ::-1], eta_top, np.zeros(nx), chi[:, ::-1], dy
    )
    tau_top = float(solution.nu) * grad_top
    upper = -float(np.nansum(tau_top * dx))
    return lower, upper


# ---------------------------------------------------------------------------
# Integral streamwise momentum balance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ForceBudget:
    """Per-unit-width streamwise force budget over the periodic cell.

    ``body`` drives the flow; ``pressure_hill`` is the hill pressure drag and
    ``shear_wall`` the integrated wall shear.  The residual uses the
    pre-registered scale-free normalisation.
    """

    body: float
    pressure_hill: float
    shear_wall: float
    residual: float


def _hill_pressure_force(
    pressure: np.ndarray,
    x_u: np.ndarray,
    y_c: np.ndarray,
    b: np.ndarray,
    dx: float,
) -> float:
    """Streamwise pressure drag on the stair-step hill (per unit width).

    Each terrain step at column ``i`` exposes a vertical face of height
    ``b(x_{i+1}) - b(x_i)``.  The fluid force from that face is ``-p dA_x``
    (high pressure on a windward face pushes the fluid backwards), evaluated
    with the pressure in the wall-adjacent fluid cell of that column.
    """
    nx = b.size
    fluid = y_c[None, :] > b[:, None] + 1e-12
    first = np.argmax(fluid, axis=1)
    p_face = np.asarray(pressure, dtype=float)[np.arange(nx), first]
    heights = np.roll(b, -1) - b
    return float(-np.sum(p_face * heights) * dx)


def force_balance(
    solution: object,
    wall_effective_viscosity: Optional[np.ndarray] = None,
) -> ForceBudget:
    """Integral longitudinal force balance for a solved periodic-hill field.

    ``r_balance = |F_body + F_pressure,hill + F_shear,wall| / max(|F_body|,
    |F_shear,wall|)``.  The wall shear is the same gradient-based observable the
    local wall-shear report uses, so both are independent checks of one field.
    """
    u = np.asarray(solution.u, dtype=float)
    pressure = np.asarray(solution.pressure, dtype=float)
    x_u = np.asarray(solution.x_u, dtype=float)
    y_c = np.asarray(solution.y_c, dtype=float)
    nx, ny = u.shape
    dx = float(solution.dx)
    dy = float(solution.dy)
    height = float(getattr(solution, "height", 0.0)) or ny * dy
    length = float(getattr(solution, "length", 0.0)) or nx * dx

    b_u = hill_height(x_u)
    # f0 acts only on fluid u-cells; subtract the solid hill area.
    solid_area = float(np.sum(y_c[None, :] < b_u[:, None] + 1e-12)) * dx * dy
    body = float(solution.f0) * (height * length - solid_area)
    pressure_hill = _hill_pressure_force(pressure, x_u, y_c, b_u, dx)
    lower, upper = _wall_shear_forces(solution, wall_effective_viscosity)
    shear_wall = lower + upper

    imbalance = body + pressure_hill + shear_wall
    scale = max(abs(body), abs(shear_wall))
    residual = abs(imbalance) if scale <= 1e-12 else abs(imbalance) / scale
    return ForceBudget(
        body=body,
        pressure_hill=pressure_hill,
        shear_wall=shear_wall,
        residual=float(residual),
    )


# ---------------------------------------------------------------------------
# Separation / reattachment from wall-shear sign changes
# ---------------------------------------------------------------------------
def hill_wall_diagnostics(
    solution: object,
    reference_profiles: Optional[Mapping[float, PeriodicHillProfile]] = None,
    *,
    stations: Optional[Tuple[float, ...]] = None,
    wall_effective_viscosity: Optional[np.ndarray] = None,
) -> Tuple[WallShearProfile, ForceBudget, List[StationObservation]]:
    """Report local wall shear and the integral force balance *together*.

    The local observable and the integral balance are independent checks of the
    same field and must be reported side by side, never as substitutes.  Returns
    ``(wall_shear_profile, force_budget, station_observations)``; the station
    list is empty when no reference profiles are supplied.
    """
    profile = wall_shear_profile(solution, wall_effective_viscosity)
    budget = force_balance(solution, wall_effective_viscosity)
    observations = (
        wall_shear_observations(
            solution,
            reference_profiles,
            stations=stations,
            wall_effective_viscosity=wall_effective_viscosity,
        )
        if reference_profiles
        else []
    )
    return profile, budget, observations


def separation_reattachment(solution: object) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(x_sep, x_reattach)`` by linearly interpolating sign changes.

    The periodic cell starts at the hill crest.  Separation is the first
    positive-to-negative crossing of ``tau_w`` after the crest; reattachment is
    the first negative-to-positive crossing after that.  Zero-valued nodes are
    skipped and the last non-zero sign is carried forward, so a root landing
    exactly on a grid node is still detected by the bracketing samples.
    """
    profile = wall_shear_profile(solution)
    x = profile.x_over_h
    tau = profile.tau_w
    separation: Optional[float] = None
    reattachment: Optional[float] = None
    prev_sign = 0
    prev_x = 0.0
    prev_tau = 0.0
    for i in range(len(x)):
        if not math.isfinite(tau[i]):
            continue
        current = float(tau[i])
        sign = 1 if current > 0.0 else (-1 if current < 0.0 else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            if current != prev_tau:
                xc = float(prev_x + prev_tau / (prev_tau - current) * (x[i] - prev_x))
            else:
                xc = float(x[i])
            if separation is None and prev_sign > 0 and sign < 0:
                separation = xc
            elif (
                separation is not None
                and reattachment is None
                and prev_sign < 0
                and sign > 0
            ):
                reattachment = xc
                break
        prev_sign = sign
        prev_x = float(x[i])
        prev_tau = current
    return separation, reattachment
