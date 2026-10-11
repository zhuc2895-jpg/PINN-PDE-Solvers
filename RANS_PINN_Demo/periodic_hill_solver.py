"""Coarse-grid 2-D RANS baseline for the periodic-hill benchmark.

The solver is intentionally small and transparent: a staggered MAC projection
method on a uniform Cartesian grid, a sharp stair-step immersed boundary for
the hill, second-order upwind convection, and a two-equation-free mixing-length
eddy viscosity.  It is *not* a validated general-purpose CFD code.  Its job in
this project is to provide a stable, reproducible, deliberately imperfect
baseline whose closure error can be measured against the independent LES.

The independent reference is loaded through :mod:`RANS_PINN_Demo.periodic_hill`.
No reference field is ever fed into the solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .periodic_hill import (
    BULK_VELOCITY,
    CHANNEL_HEIGHT,
    KINEMATIC_VISCOSITY,
    PERIOD,
    REFERENCE_STATIONS,
    hill_height,
)

__all__ = [
    "PeriodicHillSolution",
    "PeriodicHillSolver",
    "solve_periodic_hill",
    "wall_shear_profile",
    "separation_reattachment",
]


def _thomas(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve one tridiagonal system with complex right-hand sides."""
    n = diagonal.size
    cp = np.zeros(n - 1, dtype=complex)
    dp = np.zeros(n, dtype=complex)
    cp[0] = upper[0] / diagonal[0]
    dp[0] = rhs[0] / diagonal[0]
    for i in range(1, n):
        m = diagonal[i] - lower[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / m
    x = np.zeros(n, dtype=complex)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _solve_pressure(
    rhs: np.ndarray,
    dx: float,
    dy: float,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Pressure Poisson solve, periodic in x and Neumann in y.

    The k=0 Fourier mode is singular because the pressure is defined up to a
    constant.  A tiny diagonal regularisation plus removal of the mean gives a
    well-defined representative; the velocity correction is unaffected.
    """
    nx, ny = rhs.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    rhs_hat = np.fft.fft(rhs, axis=0)
    p_hat = np.zeros_like(rhs_hat)
    for mi in range(nx):
        k2 = kx[mi] ** 2
        diagonal = np.full(ny, -2.0 / dy**2 - k2)
        lower = np.full(ny - 1, 1.0 / dy**2)
        upper = np.full(ny - 1, 1.0 / dy**2)
        # Neumann: p_{-1}=p_0 and p_N=p_{N-1}
        diagonal[0] = -1.0 / dy**2 - k2
        diagonal[-1] = -1.0 / dy**2 - k2
        if abs(kx[mi]) < 1e-14:
            diagonal = diagonal - eps
        p_hat[mi, :] = _thomas(lower, diagonal, upper, rhs_hat[mi, :])
    pressure = np.real(np.fft.ifft(p_hat, axis=0))
    return pressure - pressure.mean()


@dataclass
class PeriodicHillSolution:
    """Everything needed to judge a coarse periodic-hill solve."""

    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray
    x_u: np.ndarray
    y_c: np.ndarray
    dx: float
    dy: float
    f0: float
    nu: float
    q_history: List[float] = field(default_factory=list)
    change_history: List[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False

    @property
    def q(self) -> float:
        return float(self.q_history[-1]) if self.q_history else float("nan")

    @property
    def separation_x(self) -> Optional[float]:
        return separation_reattachment(self)[0]

    @property
    def reattachment_x(self) -> Optional[float]:
        return separation_reattachment(self)[1]


class PeriodicHillSolver:
    """A small MAC solver for the 2-D incompressible RANS equations.

    Parameters are deliberately exposed so that the same object can run the
    flat-channel verification case (set ``hill`` to zero) and the periodic hill.
    """

    def __init__(
        self,
        nx: int = 96,
        ny: int = 48,
        *,
        re_h: float = 10595.0,
        length: float = PERIOD,
        height: float = CHANNEL_HEIGHT,
        mixing_length: bool = True,
        kappa: float = 0.41,
        l_cap_fraction: float = 0.09,
        dt: float = 0.01,
        f0: float = 0.036,
        q_target: float = BULK_VELOCITY * (CHANNEL_HEIGHT - 1.0),
        controller_gain: float = 1e-3,
        controller: bool = False,
        convective_scheme: str = "sou",
        dtype: np.dtype = np.float64,
    ) -> None:
        if nx < 4 or ny < 4:
            raise ValueError("nx and ny must be at least 4")
        if not 0.0 <= l_cap_fraction:
            raise ValueError("l_cap_fraction must be non-negative")
        if convective_scheme not in {"sou", "upwind"}:
            raise ValueError("convective_scheme must be 'sou' or 'upwind'")

        self.nx = int(nx)
        self.ny = int(ny)
        self.length = float(length)
        self.height = float(height)
        self.re_h = float(re_h)
        self.nu = 1.0 / self.re_h
        self.dx = self.length / self.nx
        self.dy = self.height / self.ny
        self.mixing_length = bool(mixing_length)
        self.kappa = float(kappa)
        self.l_cap_fraction = float(l_cap_fraction)
        self.dt = float(dt)
        self.f0 = float(f0)
        self.q_target = float(q_target)
        self.controller_gain = float(controller_gain)
        self.controller = bool(controller)
        self.convective_scheme = convective_scheme
        self.dtype = dtype

        # MAC grid:
        #   u[i,j] lives at x=i*dx, y=(j+0.5)*dy
        #   v[i,j] lives at x=(i+0.5)*dx, y=j*dy
        #   p[i,j] lives at cell centres
        self.x_u = np.arange(self.nx, dtype=dtype) * self.dx
        self.x_c = (np.arange(self.nx, dtype=dtype) + 0.5) * self.dx
        self.y_c = (np.arange(self.ny, dtype=dtype) + 0.5) * self.dy
        self.y_v = np.arange(self.ny + 1, dtype=dtype) * self.dy

        self.u = np.zeros((self.nx, self.ny), dtype=dtype)
        self.v = np.zeros((self.nx, self.ny + 1), dtype=dtype)
        self.pressure = np.zeros((self.nx, self.ny), dtype=dtype)

        self._build_masks()

        self.ip = (np.arange(self.nx) + 1) % self.nx
        self.im = (np.arange(self.nx) - 1) % self.nx

    # ------------------------------------------------------------------
    # Geometry and masks
    # ------------------------------------------------------------------
    def _build_masks(self) -> None:
        b_u = hill_height(self.x_u)
        b_v = hill_height(self.x_c)
        b_c = hill_height(self.x_c)
        self.b_u = b_u
        self.b_v = b_v
        self.b_c = b_c
        # Sharp immersed boundary: 1 inside the hill, 0 in the fluid.  The
        # stair-step is the price paid for a Cartesian coarse-grid baseline.
        self.chi_u = (self.y_c[None, :] < b_u[:, None]).astype(self.dtype)
        self.chi_v = (self.y_v[None, :] < b_v[:, None]).astype(self.dtype)
        self.chi_c = (self.y_c[None, :] < b_c[:, None]).astype(self.dtype)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def strain_rate_magnitude(self) -> np.ndarray:
        """Cell-centred strain-rate magnitude used by the mixing-length model."""
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        # du/dy at cell centres, with an odd ghost at each no-slip wall.
        u_ext = np.empty((nx, ny + 2), dtype=self.dtype)
        u_ext[:, 1:-1] = self.u
        u_ext[:, 0] = -self.u[:, 0]
        u_ext[:, -1] = -self.u[:, -1]
        du_dy = (u_ext[:, 2:] - u_ext[:, :-2]) / (2.0 * dy)
        # dv/dx at cell centres: v is staggered in x.
        v_c = 0.5 * (self.v[:, :-1] + self.v[self.im, :-1])
        dv_dx = (np.roll(v_c, -1, axis=0) - v_c) / dx
        # dv/dy at cell centres: v is staggered in y.
        dv_dy = (self.v[:, 1:] - self.v[:, :-1]) / dy
        return np.sqrt(2.0 * (dv_dx**2 + du_dy**2 + 0.5 * (du_dy + dv_dx) ** 2))

    def eddy_viscosity(self, strain: Optional[np.ndarray] = None) -> np.ndarray:
        """Mixing-length eddy viscosity ``nu_t`` at cell centres."""
        if not self.mixing_length:
            return np.zeros((self.nx, self.ny), dtype=self.dtype)
        if strain is None:
            strain = self.strain_rate_magnitude()
        wall_distance = np.minimum(
            self.y_c[None, :] - self.b_c[:, None],
            self.height - self.y_c[None, :],
        )
        wall_distance = np.maximum(wall_distance, 0.0)
        length = np.minimum(
            self.kappa * wall_distance,
            self.l_cap_fraction * 0.5 * self.height,
        )
        nu_t = length**2 * strain
        # No eddy viscosity inside the solid or at the walls.
        return nu_t * (1.0 - self.chi_c)

    def flow_rate(self) -> float:
        """Volume flow rate per unit width, in hill-height/bulk-velocity units."""
        return float(
            np.sum(self.u * (1.0 - self.chi_u)) * self.dx * self.dy / self.length
        )

    # ------------------------------------------------------------------
    # One pseudo-time step
    # ------------------------------------------------------------------
    def step(self) -> Tuple[float, float]:
        """Advance one pseudo-time step and return ``(Q, max_change)``."""
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        u_old = self.u.copy()
        v_old = self.v.copy()

        strain = self.strain_rate_magnitude()
        nu_eff = self.nu + self.eddy_viscosity(strain)
        # Face interpolations.
        nu_u = 0.5 * (nu_eff + nu_eff[self.im, :])
        nu_v = np.zeros((nx, ny + 1), dtype=self.dtype)
        nu_v[:, 1:ny] = 0.5 * (nu_eff[:, :-1] + nu_eff[:, 1:])

        # ----- u momentum -----
        v_at_u = 0.25 * (
            self.v[:, :-1]
            + self.v[:, 1:]
            + self.v[self.im, :-1]
            + self.v[self.im, 1:]
        )
        du_dx = self._x_derivative(self.u, self.u, dx)
        du_dy = self._y_derivative_u(self.u, v_at_u, dy)
        conv_u = -(self.u * du_dx + v_at_u * du_dy)

        u_ext = np.empty((nx, ny + 2), dtype=self.dtype)
        u_ext[:, 1:-1] = self.u
        u_ext[:, 0] = -self.u[:, 0]
        u_ext[:, -1] = -self.u[:, -1]
        lap_u = (self.u[self.ip, :] - 2.0 * self.u + self.u[self.im, :]) / dx**2 + (
            u_ext[:, 2:] - 2.0 * u_ext[:, 1:-1] + u_ext[:, :-2]
        ) / dy**2
        u_star = self.u + self.dt * (conv_u + nu_u * lap_u + self.f0)
        u_star[self.chi_u > 0.5] = 0.0

        # ----- v momentum -----
        u_at_v = np.zeros((nx, ny + 1), dtype=self.dtype)
        u_at_v[:, 1:ny] = 0.25 * (
            self.u[:, :-1]
            + self.u[self.ip, :-1]
            + self.u[:, 1:]
            + self.u[self.ip, 1:]
        )
        dv_dx = self._x_derivative(self.v, u_at_v, dx)
        dv_dy = self._y_derivative_v(self.v, dy)
        conv_v = -(u_at_v * dv_dx + self.v * dv_dy)

        vjm = np.empty_like(self.v)
        vjp = np.empty_like(self.v)
        vjm[:, 0] = 0.0
        vjm[:, 1:] = self.v[:, :-1]
        vjp[:, -1] = 0.0
        vjp[:, :-1] = self.v[:, 1:]
        lap_v = (self.v[self.ip, :] - 2.0 * self.v + self.v[self.im, :]) / dx**2 + (
            vjp - 2.0 * self.v + vjm
        ) / dy**2
        v_star = self.v + self.dt * (conv_v + nu_v * lap_v)
        v_star[self.chi_v > 0.5] = 0.0
        v_star[:, 0] = 0.0
        v_star[:, -1] = 0.0

        # ----- projection -----
        divergence = (u_star[self.ip, :] - u_star) / dx + (v_star[:, 1:] - v_star[:, :-1]) / dy
        divergence = divergence - divergence.mean()
        pressure = _solve_pressure(divergence / self.dt, dx, dy)
        self.pressure = pressure
        self.u = u_star - self.dt * (pressure - pressure[self.im, :]) / dx
        self.v = v_star.copy()
        self.v[:, 1:ny] = v_star[:, 1:ny] - self.dt * (
            pressure[:, 1:] - pressure[:, :-1]
        ) / dy
        # Re-impose the no-slip solid and the top/bottom v walls.
        self.u[self.chi_u > 0.5] = 0.0
        self.v[self.chi_v > 0.5] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0

        q = self.flow_rate()
        if self.controller:
            self.f0 += self.controller_gain * (self.q_target - q)
        change = max(
            float(np.max(np.abs(self.u - u_old))),
            float(np.max(np.abs(self.v - v_old))),
        )
        return q, change

    def _x_derivative(self, field: np.ndarray, velocity: np.ndarray, dx: float) -> np.ndarray:
        """Second-order upwind or first-order upwind derivative in x."""
        if self.convective_scheme == "upwind":
            return np.where(
                velocity >= 0.0,
                field - field[self.im, :],
                field[self.ip, :] - field,
            ) / dx
        fm1 = np.roll(field, 1, axis=0)
        fm2 = np.roll(field, 2, axis=0)
        fp1 = np.roll(field, -1, axis=0)
        fp2 = np.roll(field, -2, axis=0)
        return np.where(
            velocity >= 0.0,
            (3.0 * field - 4.0 * fm1 + fm2) / (2.0 * dx),
            (-3.0 * field + 4.0 * fp1 - fp2) / (2.0 * dx),
        )

    def _y_derivative_u(self, field: np.ndarray, velocity: np.ndarray, dy: float) -> np.ndarray:
        """Derivative in y for u, with odd ghost values at the walls."""
        nx, ny = self.nx, self.ny
        if self.convective_scheme == "upwind":
            jm = np.empty_like(field)
            jp = np.empty_like(field)
            jm[:, 0] = -field[:, 0]
            jm[:, 1:] = field[:, :-1]
            jp[:, -1] = -field[:, -1]
            jp[:, :-1] = field[:, 1:]
            return np.where(velocity >= 0.0, field - jm, jp - field) / dy
        ext = np.empty((nx, ny + 4), dtype=self.dtype)
        ext[:, 0] = -field[:, 1]
        ext[:, 1] = -field[:, 0]
        ext[:, 2 : ny + 2] = field
        ext[:, ny + 2] = -field[:, ny - 1]
        ext[:, ny + 3] = -field[:, ny - 2]
        out = np.empty_like(field)
        for j in range(ny):
            c = ext[:, j + 2]
            m1 = ext[:, j + 1]
            m2 = ext[:, j]
            p1 = ext[:, j + 3]
            p2 = ext[:, j + 4]
            out[:, j] = np.where(
                velocity[:, j] >= 0.0,
                (3.0 * c - 4.0 * m1 + m2) / (2.0 * dy),
                (-3.0 * c + 4.0 * p1 - p2) / (2.0 * dy),
            )
        return out

    def _y_derivative_v(self, field: np.ndarray, dy: float) -> np.ndarray:
        """Derivative in y for v, with wall data v=0 at j=0 and j=ny."""
        nx, ny = self.nx, self.ny
        if self.convective_scheme == "upwind":
            jm = np.empty_like(field)
            jp = np.empty_like(field)
            jm[:, 0] = 0.0
            jm[:, 1:] = field[:, :-1]
            jp[:, -1] = 0.0
            jp[:, :-1] = field[:, 1:]
            return np.where(field >= 0.0, field - jm, jp - field) / dy
        out = np.zeros_like(field)
        for j in range(1, ny):
            c = field[:, j]
            if j == 1:
                m1 = np.zeros(nx, dtype=self.dtype)
                m2 = np.zeros(nx, dtype=self.dtype)
            elif j == 2:
                m1 = field[:, 1]
                m2 = np.zeros(nx, dtype=self.dtype)
            else:
                m1 = field[:, j - 1]
                m2 = field[:, j - 2]
            if j == ny - 1:
                p1 = np.zeros(nx, dtype=self.dtype)
                p2 = np.zeros(nx, dtype=self.dtype)
            elif j == ny - 2:
                p1 = field[:, ny - 1]
                p2 = np.zeros(nx, dtype=self.dtype)
            else:
                p1 = field[:, j + 1]
                p2 = field[:, j + 2]
            out[:, j] = np.where(
                c >= 0.0,
                (3.0 * c - 4.0 * m1 + m2) / (2.0 * dy),
                (-3.0 * c + 4.0 * p1 - p2) / (2.0 * dy),
            )
        return out

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        max_steps: int = 30000,
        tol: float = 1e-7,
        patience: int = 20,
        callback: Optional[object] = None,
    ) -> PeriodicHillSolution:
        """Pseudo-time march to a steady state.

        ``callback``, if given, is called as ``callback(step, q, change)`` every
        step.  The default is no output.
        """
        stable = 0
        change = float("inf")
        q = self.flow_rate()
        q_history: List[float] = []
        change_history: List[float] = []
        for step in range(max_steps):
            q, change = self.step()
            q_history.append(float(q))
            change_history.append(float(change))
            if callback is not None:
                callback(step, q, change)
            if math.isfinite(change) and change < tol:
                stable += 1
            else:
                stable = 0
            if stable >= patience:
                return PeriodicHillSolution(
                    u=self.u.copy(),
                    v=self.v.copy(),
                    pressure=self.pressure.copy(),
                    x_u=self.x_u.copy(),
                    y_c=self.y_c.copy(),
                    dx=self.dx,
                    dy=self.dy,
                    f0=self.f0,
                    nu=self.nu,
                    q_history=q_history,
                    change_history=change_history,
                    iterations=step + 1,
                    converged=True,
                )
        return PeriodicHillSolution(
            u=self.u.copy(),
            v=self.v.copy(),
            pressure=self.pressure.copy(),
            x_u=self.x_u.copy(),
            y_c=self.y_c.copy(),
            dx=self.dx,
            dy=self.dy,
            f0=self.f0,
            nu=self.nu,
            q_history=q_history,
            change_history=change_history,
            iterations=max_steps,
            converged=False,
        )


def solve_periodic_hill(
    *,
    nx: int = 96,
    ny: int = 48,
    max_steps: int = 30000,
    dt: float = 0.01,
    f0: float = 0.036,
    controller: bool = True,
    controller_gain: float = 1e-3,
    mixing_length: bool = True,
    tol: float = 1e-7,
    patience: int = 20,
) -> PeriodicHillSolution:
    """Convenience wrapper around :class:`PeriodicHillSolver`."""
    solver = PeriodicHillSolver(
        nx=nx,
        ny=ny,
        dt=dt,
        f0=f0,
        controller=controller,
        controller_gain=controller_gain,
        mixing_length=mixing_length,
    )
    solver.u[:] = BULK_VELOCITY
    return solver.run(max_steps=max_steps, tol=tol, patience=patience)


def wall_shear_profile(
    solution: PeriodicHillSolution,
    wall_effective_viscosity: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(x_u, tau_w)`` for a solved periodic-hill field.

    The wall is the first solid/fluid interface above the hill.  A linear fit
    through the first two fluid u samples is extrapolated to the wall and
    multiplied by the effective viscosity at the wall.  The default uses the
    molecular viscosity because the mixing-length model enforces ``nu_t=0`` at
    the wall; a learned correction must pass its wall effective viscosity
    explicitly through ``wall_effective_viscosity`` and preserve that limit.
    """
    u = solution.u
    x = solution.x_u
    y = solution.y_c
    dx = solution.dx
    dy = solution.dy
    nx, ny = u.shape
    # Reconstruct the hill height from the solution coordinates.  The first
    # fluid index is the first cell whose centre is above b(x).
    b = hill_height(x)
    if wall_effective_viscosity is None:
        wall_viscosity = np.full(nx, solution.nu, dtype=float)
    else:
        wall_viscosity = np.asarray(wall_effective_viscosity, dtype=float)
        if wall_viscosity.shape != (nx,):
            raise ValueError("wall_effective_viscosity must have shape (nx,)")
    tau = np.full(nx, np.nan)
    for i in range(nx):
        fluid = np.where(y > b[i] + 1e-12)[0]
        if fluid.size < 2:
            continue
        j0 = int(fluid[0])
        j1 = int(fluid[1])
        # Linear fit u = a + b*y through the first two fluid samples; the
        # wall gradient is the slope at y=b.
        slope = (u[i, j1] - u[i, j0]) / (y[j1] - y[j0])
        tau[i] = wall_viscosity[i] * slope
    return x, tau


def separation_reattachment(
    solution: PeriodicHillSolution,
) -> Tuple[Optional[float], Optional[float]]:
    """Return approximate ``(x_sep, x_reattach)`` from wall-shear sign changes.

    The periodic domain starts at the hill crest.  Separation is the first
    positive-to-negative crossing after the crest; reattachment is the first
    negative-to-positive crossing after that.
    """
    x, tau = wall_shear_profile(solution)
    finite = np.isfinite(tau)
    if finite.sum() < 3:
        return None, None
    # Work on the ordered x grid and interpolate sign changes linearly.
    sign = np.sign(tau)
    separation = None
    reattachment = None
    for i in range(len(x) - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        if sign[i] == sign[i + 1] or sign[i] == 0 or sign[i + 1] == 0:
            continue
        # Linear zero crossing.
        t = tau[i] / (tau[i] - tau[i + 1])
        xc = x[i] + t * (x[i + 1] - x[i])
        if separation is None and tau[i] > 0.0 and tau[i + 1] < 0.0:
            separation = float(xc)
        elif separation is not None and reattachment is None and tau[i] < 0.0 and tau[i + 1] > 0.0:
            reattachment = float(xc)
            break
    return separation, reattachment
