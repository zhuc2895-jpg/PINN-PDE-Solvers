"""Observable wall-shear criteria for the 2-D RANS extension.

The 1-D channel demo cannot validate wall shear: ``u_tau`` is an input, so the
wall stress is fixed by construction.  A 2-D solver must instead compute the
wall-normal velocity gradient, multiply it by the effective viscosity at the
wall, infer friction velocity from that stress, and compare the result with an
independent reference and the integral force balance.

The functions here deliberately take a stress or a gradient, never a prescribed
``u_tau``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "WallShearObservation",
    "wall_shear_from_gradient",
    "friction_velocity_from_wall_shear",
    "wall_shear_relative_error",
    "wall_force_balance_residual",
]


@dataclass(frozen=True)
class WallShearObservation:
    """A wall-shear observation with friction velocity inferred, not supplied."""

    tau_w: float
    u_tau: float
    density: float


def wall_shear_from_gradient(
    effective_viscosity: float,
    wall_normal_gradient: float,
) -> float:
    """Return ``tau_w = mu_eff * du_t/dn`` at a wall face.

    The caller is responsible for the wall-normal orientation and sign
    convention.  ``effective_viscosity`` must be the material viscosity plus any
    eddy viscosity actually active at the wall; it must be positive.
    """
    if not math.isfinite(effective_viscosity) or effective_viscosity <= 0.0:
        raise ValueError("effective_viscosity must be finite and positive")
    if not math.isfinite(wall_normal_gradient):
        raise ValueError("wall_normal_gradient must be finite")
    return effective_viscosity * wall_normal_gradient


def friction_velocity_from_wall_shear(
    tau_w: float,
    density: float = 1.0,
) -> WallShearObservation:
    """Infer friction velocity from an independently computed wall stress."""
    if not math.isfinite(tau_w):
        raise ValueError("tau_w must be finite")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density must be finite and positive")
    return WallShearObservation(
        tau_w=tau_w,
        u_tau=math.sqrt(abs(tau_w) / density),
        density=density,
    )


def wall_shear_relative_error(tau_w_predicted: float, tau_w_reference: float) -> float:
    """Relative wall-shear error against an independent high-fidelity value."""
    if not math.isfinite(tau_w_predicted) or not math.isfinite(tau_w_reference):
        raise ValueError("wall shear values must be finite")
    if tau_w_reference == 0.0:
        raise ValueError("reference wall shear must be non-zero")
    return abs(tau_w_predicted - tau_w_reference) / abs(tau_w_reference)


def wall_force_balance_residual(
    pressure_force: float,
    wall_shear_force: float,
) -> float:
    """Normalised residual of ``F_pressure + F_shear = 0``.

    For a periodic domain, the streamwise pressure force and integrated wall
    shear force must balance.  The residual is scale-free and returns zero only
    when the available force scales are both zero (the degenerate case).
    """
    if not math.isfinite(pressure_force) or not math.isfinite(wall_shear_force):
        raise ValueError("force terms must be finite")
    scale = max(abs(pressure_force), abs(wall_shear_force))
    if scale == 0.0:
        return 0.0
    return abs(pressure_force + wall_shear_force) / scale