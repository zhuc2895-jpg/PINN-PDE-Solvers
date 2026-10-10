"""T3: invert the reference profile and separate the error sources.

The forward solver integrates

    dU/dy = (1 - y) / (nu + nu_t)

with a cumulative trapezoid.  That discrete form can be inverted exactly: if
``A_i`` is the average derivative over segment ``i``, then

    A_i = (f_i + f_{i-1}) / 2

so, starting from the physical wall condition ``nu_t(0) = 0``, each nodal
derivative ``f_i`` is recovered by recurrence.  The result replays the supplied
profile to floating-point roundoff -- this is a consistency check, not a fit.

One point is deliberately different: at the centreline the total stress is zero,
so every finite ``nu_t`` produces the same zero mean shear.  The inversion marks
that point unidentifiable instead of inventing a value.

The second half of the module separates the baseline-to-truth velocity gap into
two parts:

* closure error: baseline and truth on the same coarse grid;
* discretisation error: coarse truth versus a fine-grid truth reference.

The split is additive by construction.  It is not an orthogonality claim, so the
table reports both relative norms and magnitude shares.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from pinn.evaluate import relative_l2
from pinn.plotting import plt, save_figure

from .channel import _interpolate
from .closure import BaselineClosure, TruthClosure
from .solver import solve_channel, wall_clustered_grid

__all__ = [
    "EddyViscosityInversion",
    "TabulatedClosure",
    "ErrorDecomposition",
    "BandSensitivity",
    "IdentifiabilityReport",
    "InversionAnalysis",
    "invert_eddy_viscosity",
    "replay_error",
    "decompose_errors",
    "identifiability_report",
    "analyze_inversion",
    "save_inversion_artifacts",
]

DEFAULT_RE_TAUS: Tuple[float, ...] = (180.0, 395.0)
DEFAULT_COARSE_N = 801
DEFAULT_FINE_N = 20001

#: Practical sensitivity floor after normalising by the strongest band.  A band
#: below this is reported as practically unidentifiable at the available
#: profile resolution, even though the exact centreline is the only point with
#: identically zero sensitivity.
IDENTIFIABILITY_FLOOR = 1e-3
IDENTIFIABILITY_TAPER_PLUS = 5.0
IDENTIFIABILITY_PERTURBATION = 0.05


def _column(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1, 1)


def _nested_bands(Re_tau: float) -> Tuple[Tuple[float, float], ...]:
    """Bands including both the wall and the exact centreline point."""
    return (
        (0.0, 3.0),
        (3.0, 0.15 * Re_tau),
        (0.15 * Re_tau, 0.40 * Re_tau),
        (0.40 * Re_tau, 0.70 * Re_tau),
        (0.70 * Re_tau, 0.95 * Re_tau),
        (0.95 * Re_tau, Re_tau),
    )


@dataclass
class EddyViscosityInversion:
    """Nodal inversion result; ``nu_t`` is NaN where it is unidentifiable."""

    y: torch.Tensor
    Re_tau: float
    nu_t: torch.Tensor
    identifiable: torch.Tensor
    stress: torch.Tensor
    wall_derivative: float


class TabulatedClosure:
    """A prescribed ``nu_t`` table exposed through the normal ``Closure`` seam."""

    def __init__(self, y: torch.Tensor, nu_t: torch.Tensor):
        self.y = _column(y).to(torch.float64).reshape(-1)
        nu_t = _column(nu_t).to(torch.float64).reshape(-1)
        if self.y.shape != nu_t.shape:
            raise ValueError("y and nu_t must have the same flattened shape")
        # Any finite value is equivalent where total stress is zero; replay uses
        # zero there and never hides the unidentifiable point in the inversion.
        self.nu_t = torch.nan_to_num(nu_t, nan=0.0)

    def __call__(
        self,
        y: torch.Tensor,
        Re_tau: float,
        shear: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = y.reshape(-1).to(self.y.dtype)
        idx = torch.searchsorted(self.y, target).clamp(1, self.y.numel() - 1)
        x0, x1 = self.y[idx - 1], self.y[idx]
        y0, y1 = self.nu_t[idx - 1], self.nu_t[idx]
        weight = (target - x0) / (x1 - x0)
        return (y0 + weight * (y1 - y0)).reshape(-1, 1)


def invert_eddy_viscosity(
    U: torch.Tensor,
    y: torch.Tensor,
    Re_tau: float,
    *,
    u_tau: float = 1.0,
    h: float = 1.0,
) -> EddyViscosityInversion:
    """Recover ``nu_t`` from a velocity profile on a solver grid.

    The profile must start at the wall ``y=0`` and be monotone in ``y``.  At the
    centreline the stress factor is zero, so the local value of ``nu_t`` is not
    observable from ``U``; that node is returned as NaN and flagged.
    """
    if Re_tau <= 0.0 or u_tau <= 0.0 or h <= 0.0:
        raise ValueError("Re_tau, u_tau and h must be positive")
    y = _column(y).to(torch.float64)
    U = _column(U).to(torch.float64)
    if y.shape != U.shape:
        raise ValueError("U and y must have the same shape")
    if y.numel() < 2:
        raise ValueError("at least two grid points are required")
    if not torch.isfinite(y).all() or not torch.isfinite(U).all():
        raise ValueError("U and y must be finite")
    if abs(float(y[0])) > 1e-12 * max(1.0, h):
        raise ValueError("inversion requires the wall grid point y[0] = 0")
    dy = torch.diff(y.reshape(-1))
    if not torch.all(dy > 0.0):
        raise ValueError("y must be strictly increasing")

    nu = 1.0 / Re_tau
    stress = u_tau**2 * (1.0 - y / h)
    f = torch.empty_like(y)

    # The wall condition nu_t(0)=0 fixes the first nodal derivative exactly.
    f[0] = stress[0] / nu
    for index in range(1, y.numel()):
        average = (U[index] - U[index - 1]) / (y[index] - y[index - 1])
        f[index] = 2.0 * average - f[index - 1]

    identifiable = stress.abs() > 1e-12 * u_tau**2
    safe_f = torch.where(identifiable, f, torch.ones_like(f))
    nu_t = torch.full_like(f, float("nan"))
    nu_t = torch.where(identifiable, stress / safe_f - nu, nu_t)
    identifiable = identifiable & torch.isfinite(nu_t)

    return EddyViscosityInversion(
        y=y,
        Re_tau=float(Re_tau),
        nu_t=nu_t,
        identifiable=identifiable,
        stress=stress,
        wall_derivative=float(f[0]),
    )


def _solve_inverted(inversion: EddyViscosityInversion, y: torch.Tensor):
    closure = TabulatedClosure(inversion.y, inversion.nu_t)
    return solve_channel(closure, inversion.Re_tau, y=y)


def replay_error(U: torch.Tensor, inversion: EddyViscosityInversion) -> float:
    """Relative profile error after feeding the inverted closure back to RANS."""
    replay = _solve_inverted(inversion, inversion.y)
    return relative_l2(replay.U, _column(U))


@dataclass
class ErrorDecomposition:
    """Additive split of the baseline velocity error on the coarse grid."""

    Re_tau: float
    y: torch.Tensor
    reference_U: torch.Tensor
    baseline_U: torch.Tensor
    truth_coarse_U: torch.Tensor
    closure_error_U: torch.Tensor
    discretization_error_U: torch.Tensor
    total_error_U: torch.Tensor
    baseline_nu_t: torch.Tensor
    truth_nu_t: torch.Tensor
    inversion: EddyViscosityInversion
    inversion_relative_error: float
    replay_relative_error: float
    closure_relative_l2: float
    discretization_relative_l2: float
    total_relative_l2: float
    closure_share: float
    discretization_share: float


def decompose_errors(
    Re_tau: float,
    *,
    coarse_y: Optional[torch.Tensor] = None,
    fine_y: Optional[torch.Tensor] = None,
) -> ErrorDecomposition:
    """Split baseline error into closure and discretisation components.

    The components are evaluated on the same coarse grid and satisfy the exact
    vector identity ``total = closure + discretisation``.  The truth profile is
    solved on a much finer grid before interpolation, so the fine-grid solution
    is the reference rather than a second coarse solve.
    """
    coarse_y = wall_clustered_grid(DEFAULT_COARSE_N) if coarse_y is None else _column(coarse_y)
    fine_y = wall_clustered_grid(DEFAULT_FINE_N) if fine_y is None else _column(fine_y)
    coarse_y = coarse_y.to(torch.float64)
    fine_y = fine_y.to(torch.float64)

    truth = TruthClosure()
    baseline = BaselineClosure()
    fine_truth = solve_channel(truth, Re_tau, y=fine_y)
    reference_U = _interpolate(
        coarse_y.reshape(-1),
        fine_y.reshape(-1),
        fine_truth.U.reshape(-1),
    ).reshape(-1, 1)
    baseline_solution = solve_channel(baseline, Re_tau, y=coarse_y)
    truth_coarse = solve_channel(truth, Re_tau, y=coarse_y)

    closure_error = baseline_solution.U - truth_coarse.U
    discretization_error = truth_coarse.U - reference_U
    total_error = baseline_solution.U - reference_U
    reference_norm = torch.norm(reference_U)
    closure_norm = torch.norm(closure_error)
    discretization_norm = torch.norm(discretization_error)
    share_denominator = closure_norm + discretization_norm

    inversion = invert_eddy_viscosity(truth_coarse.U, coarse_y, Re_tau)
    truth_nu_t = truth(coarse_y, Re_tau).reshape(-1, 1)
    mask = inversion.identifiable.reshape(-1, 1)
    inversion_error = torch.norm(
        (inversion.nu_t.reshape(-1, 1) - truth_nu_t)[mask]
    ) / torch.norm(truth_nu_t[mask])
    replay = replay_error(truth_coarse.U, inversion)

    return ErrorDecomposition(
        Re_tau=float(Re_tau),
        y=coarse_y,
        reference_U=reference_U,
        baseline_U=baseline_solution.U,
        truth_coarse_U=truth_coarse.U,
        closure_error_U=closure_error,
        discretization_error_U=discretization_error,
        total_error_U=total_error,
        baseline_nu_t=baseline(coarse_y, Re_tau).reshape(-1, 1),
        truth_nu_t=truth_nu_t,
        inversion=inversion,
        inversion_relative_error=float(inversion_error),
        replay_relative_error=float(replay),
        closure_relative_l2=float(closure_norm / reference_norm),
        discretization_relative_l2=float(discretization_norm / reference_norm),
        total_relative_l2=float(torch.norm(total_error) / reference_norm),
        closure_share=float(closure_norm / share_denominator),
        discretization_share=float(discretization_norm / share_denominator),
    )


@dataclass(frozen=True)
class BandSensitivity:
    Re_tau: float
    y_plus_lo: float
    y_plus_hi: float
    sensitivity: float
    normalized_sensitivity: float
    identifiable: bool
    reason: str
    is_centreline: bool = False


@dataclass(frozen=True)
class IdentifiabilityReport:
    Re_tau: float
    floor: float
    bands: Tuple[BandSensitivity, ...]
    centreline_sensitivity: float
    centreline_identifiable: bool
    overall_identifiable: bool


def _band_window(y_plus: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    # The wall factor keeps the probe physically admissible: nu_t(0) remains 0.
    wall = 1.0 - torch.exp(-y_plus / IDENTIFIABILITY_TAPER_PLUS)
    return wall * torch.sigmoid((y_plus - lo) / IDENTIFIABILITY_TAPER_PLUS) * torch.sigmoid(
        (hi - y_plus) / IDENTIFIABILITY_TAPER_PLUS
    )

def identifiability_report(
    Re_tau: float,
    *,
    y: Optional[torch.Tensor] = None,
    perturbation: float = IDENTIFIABILITY_PERTURBATION,
    floor: float = IDENTIFIABILITY_FLOOR,
) -> IdentifiabilityReport:
    """Measure observable sensitivity of each wall-normal band.

    Sensitivity is the relative profile change per relative band perturbation,
    normalised by the strongest band.  It is a local diagnostic around the
    baseline closure, not a claim that every small effect is recoverable from
    noisy data.
    """
    if perturbation <= 0.0 or floor <= 0.0:
        raise ValueError("perturbation and floor must be positive")
    y = wall_clustered_grid(DEFAULT_COARSE_N) if y is None else _column(y)
    y = y.to(torch.float64)
    y_plus = y.reshape(-1) * Re_tau
    baseline = BaselineClosure()
    baseline_nu_t = baseline(y, Re_tau).reshape(-1, 1)
    baseline_solution = solve_channel(TabulatedClosure(y, baseline_nu_t), Re_tau, y=y)
    base_norm = torch.norm(baseline_solution.U)

    raw: List[Tuple[Tuple[float, float], float]] = []
    for lo, hi in _nested_bands(Re_tau):
        window = _band_window(y_plus, lo, hi).reshape(-1, 1)
        perturbed_nu_t = baseline_nu_t * (1.0 + perturbation * window)
        perturbed = solve_channel(TabulatedClosure(y, perturbed_nu_t), Re_tau, y=y)
        sensitivity = float(torch.norm(perturbed.U - baseline_solution.U) / (perturbation * base_norm))
        raw.append(((lo, hi), sensitivity))

    strongest = max(sensitivity for _, sensitivity in raw)
    bands: List[BandSensitivity] = []
    for (lo, hi), sensitivity in raw:
        normalized = sensitivity / strongest if strongest else 0.0
        identifiable = normalized >= floor
        reason = "" if identifiable else "band sensitivity below practical floor"
        bands.append(
            BandSensitivity(
                Re_tau=float(Re_tau),
                y_plus_lo=float(lo),
                y_plus_hi=float(hi),
                sensitivity=sensitivity,
                normalized_sensitivity=normalized,
                identifiable=identifiable,
                reason=reason,
            )
        )

    # Perturb only the final node.  The stress factor is exactly zero there, so
    # this is the explicit rank-deficiency check, independent of band windows.
    centreline_nu_t = baseline_nu_t.clone()
    centreline_nu_t[-1] *= 1.0 + perturbation
    centreline = solve_channel(TabulatedClosure(y, centreline_nu_t), Re_tau, y=y)
    centreline_sensitivity = float(
        torch.norm(centreline.U - baseline_solution.U) / (perturbation * base_norm)
    )
    centreline_identifiable = centreline_sensitivity > 1e-12
    bands_tuple = tuple(bands)
    return IdentifiabilityReport(
        Re_tau=float(Re_tau),
        floor=float(floor),
        bands=bands_tuple,
        centreline_sensitivity=centreline_sensitivity,
        centreline_identifiable=centreline_identifiable,
        overall_identifiable=centreline_identifiable and all(b.identifiable for b in bands_tuple),
    )


@dataclass
class InversionAnalysis:
    decompositions: Tuple[ErrorDecomposition, ...]
    identifiability: Tuple[IdentifiabilityReport, ...]


def analyze_inversion(
    Re_taus: Sequence[float] = DEFAULT_RE_TAUS,
    *,
    coarse_n: int = DEFAULT_COARSE_N,
    fine_n: int = DEFAULT_FINE_N,
) -> InversionAnalysis:
    coarse_y = wall_clustered_grid(coarse_n)
    fine_y = wall_clustered_grid(fine_n)
    decompositions = tuple(
        decompose_errors(Re_tau, coarse_y=coarse_y, fine_y=fine_y) for Re_tau in Re_taus
    )
    reports = tuple(identifiability_report(Re_tau, y=coarse_y) for Re_tau in Re_taus)
    return InversionAnalysis(decompositions=decompositions, identifiability=reports)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_inversion_analysis(analysis: InversionAnalysis, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4))
    ax = axes[0, 0]
    for decomposition in analysis.decompositions:
        y_plus = (decomposition.y.reshape(-1) * decomposition.Re_tau).numpy()
        truth = decomposition.truth_nu_t.reshape(-1).numpy()
        inverted = decomposition.inversion.nu_t.reshape(-1).numpy()
        baseline = decomposition.baseline_nu_t.reshape(-1).numpy()
        visible = torch.isfinite(decomposition.inversion.nu_t).reshape(-1).numpy()
        ax.loglog(y_plus[visible], truth[visible], label=f"truth Re={decomposition.Re_tau:g}")
        ax.loglog(
            y_plus[visible],
            inverted[visible],
            "--",
            label=f"inverted Re={decomposition.Re_tau:g}",
        )
        ax.loglog(
            y_plus[visible],
            baseline[visible],
            ":",
            alpha=0.65,
            label=f"baseline Re={decomposition.Re_tau:g}",
        )
        ax.scatter([decomposition.Re_tau], [truth[-1]], marker="x", s=35, color="black")
    ax.set_xlabel("y+")
    ax.set_ylabel("nu_t")
    ax.set_title("Inversion recovers the reference closure (centreline marked x)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    representative = max(analysis.decompositions, key=lambda item: item.Re_tau)
    y_plus = representative.y.reshape(-1).numpy() * representative.Re_tau
    for name, values in (
        ("closure", representative.closure_error_U),
        ("discretisation", representative.discretization_error_U),
        ("total", representative.total_error_U),
    ):
        ax.semilogy(y_plus, values.reshape(-1).abs().numpy() + 1e-16, label=name)
    ax.set_xlabel("y+")
    ax.set_ylabel("|U+ error|")
    ax.set_title(f"Additive error split at Re={representative.Re_tau:g}")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    labels = [f"Re={item.Re_tau:g}" for item in analysis.decompositions]
    positions = torch.arange(len(labels)).numpy()
    width = 0.34
    ax.bar(
        positions - width / 2,
        [100.0 * item.closure_relative_l2 for item in analysis.decompositions],
        width,
        label="closure",
    )
    ax.bar(
        positions + width / 2,
        [100.0 * item.discretization_relative_l2 for item in analysis.decompositions],
        width,
        label="discretisation",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("relative L2 error (%)")
    ax.set_title("Closure error dominates the coarse baseline")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    for report in analysis.identifiability:
        centres = [(band.y_plus_lo + band.y_plus_hi) / 2.0 for band in report.bands]
        values = [band.normalized_sensitivity for band in report.bands]
        ax.plot(centres, values, marker="o", label=f"Re={report.Re_tau:g}")
        unidentifiable = [
            (centre, value)
            for centre, value, band in zip(centres, values, report.bands)
            if not band.identifiable
        ]
        if unidentifiable:
            ax.scatter(
                [centre for centre, _ in unidentifiable],
                [value for _, value in unidentifiable],
                marker="x",
                s=55,
                color="red",
                zorder=4,
            )
    ax.axhline(
        analysis.identifiability[0].floor if analysis.identifiability else IDENTIFIABILITY_FLOOR,
        color="black",
        linewidth=0.9,
        linestyle="--",
        label="practical floor",
    )
    ax.set_xlabel("y+ band centre")
    ax.set_ylabel("normalised sensitivity")
    ax.set_title("Observable sensitivity; centreline is exactly unidentifiable")
    ax.grid(True, alpha=0.25)
    ax.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(path))


def save_inversion_artifacts(
    analysis: InversionAnalysis,
    out_dir: Path,
    *,
    figure_path: Optional[Path] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    error_rows: List[Dict[str, object]] = []
    sensitivity_rows: List[Dict[str, object]] = []
    summary: List[Dict[str, object]] = []
    for decomposition, report in zip(analysis.decompositions, analysis.identifiability):
        unidentifiable = [
            f"{band.y_plus_lo:g}-{band.y_plus_hi:g}" for band in report.bands if not band.identifiable
        ]
        if not report.centreline_identifiable:
            unidentifiable.append("centreline")
        error_rows.append(
            {
                "Re_tau": decomposition.Re_tau,
                "coarse_points": int(decomposition.y.numel()),
                "total_relative_l2": decomposition.total_relative_l2,
                "closure_relative_l2": decomposition.closure_relative_l2,
                "discretization_relative_l2": decomposition.discretization_relative_l2,
                "closure_share": decomposition.closure_share,
                "discretization_share": decomposition.discretization_share,
                "inversion_relative_error": decomposition.inversion_relative_error,
                "replay_relative_error": decomposition.replay_relative_error,
                "unidentifiable": ";".join(unidentifiable),
            }
        )
        for band in report.bands:
            sensitivity_rows.append(
                {
                    "Re_tau": band.Re_tau,
                    "y_plus_lo": band.y_plus_lo,
                    "y_plus_hi": band.y_plus_hi,
                    "sensitivity": band.sensitivity,
                    "normalized_sensitivity": band.normalized_sensitivity,
                    "identifiable": band.identifiable,
                    "reason": band.reason,
                    "is_centreline": False,
                }
            )
        sensitivity_rows.append(
            {
                "Re_tau": report.Re_tau,
                "y_plus_lo": report.Re_tau,
                "y_plus_hi": report.Re_tau,
                "sensitivity": report.centreline_sensitivity,
                "normalized_sensitivity": 0.0,
                "identifiable": report.centreline_identifiable,
                "reason": "centreline stress factor is zero",
                "is_centreline": True,
            }
        )
        summary.append(
            {
                "Re_tau": decomposition.Re_tau,
                "inversion_relative_error": decomposition.inversion_relative_error,
                "replay_relative_error": decomposition.replay_relative_error,
                "closure_relative_l2": decomposition.closure_relative_l2,
                "discretization_relative_l2": decomposition.discretization_relative_l2,
                "centreline_identifiable": report.centreline_identifiable,
                "overall_identifiable": report.overall_identifiable,
                "unidentifiable_bands": unidentifiable,
                "status": (
                    "partially_identifiable"
                    if not report.overall_identifiable
                    else "identifiable"
                ),
            }
        )

    _write_csv(out_dir / "inversion_error_table.csv", error_rows)
    _write_csv(out_dir / "identifiability.csv", sensitivity_rows)
    with (out_dir / "inversion_report.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    if figure_path is None:
        figure_path = out_dir / "inversion_error_decomposition.png"
    plot_inversion_analysis(analysis, Path(figure_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("RANS_PINN_Demo/results"))
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("RANS_PINN_Demo/figures/inversion_error_decomposition.png"),
    )
    parser.add_argument("--re-taus", type=float, nargs="+", default=list(DEFAULT_RE_TAUS))
    parser.add_argument("--coarse-n", type=int, default=DEFAULT_COARSE_N)
    parser.add_argument("--fine-n", type=int, default=DEFAULT_FINE_N)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analysis = analyze_inversion(tuple(args.re_taus), coarse_n=args.coarse_n, fine_n=args.fine_n)
    save_inversion_artifacts(analysis, args.out_dir, figure_path=args.figure)
    for decomposition in analysis.decompositions:
        print(
            f"Re={decomposition.Re_tau:g}: closure={decomposition.closure_relative_l2:.6g}, "
            f"discretisation={decomposition.discretization_relative_l2:.6g}, "
            f"replay={decomposition.replay_relative_error:.3g}"
        )
    print(f"Wrote inversion artifacts to {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

