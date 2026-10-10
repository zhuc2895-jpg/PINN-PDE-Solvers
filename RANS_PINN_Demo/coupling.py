"""T8: freeze the learned closure and scan under-relaxation factors.

The offline model is trained once per seed with the T6/T7 frozen
hyperparameters, then its parameters are frozen before any coupled solve.  The
same :func:`~RANS_PINN_Demo.solver.solve_channel` used by the experiment is
then asked to converge with ``alpha`` in ``{1.0, 0.5, 0.2}``.

The report is deliberately not a success summary: a run is labelled
``converged``, ``oscillated``, ``diverged``, ``stalled`` or ``non_finite`` from
the solver history, and the raw history is archived for every seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from pinn.evaluate import relative_l2
from pinn.model import MLP
from pinn.plotting import plt, save_figure
from pinn.trainer import TrainConfig, train

from .channel import (
    TRAINING_SOLVE_TOL,
    ChannelClosure,
    ChannelProblem,
)
from .experiment import (
    DEFAULT_EPOCHS,
    DEFAULT_MODEL_DEPTH,
    DEFAULT_MODEL_WIDTH,
    DEFAULT_SEEDS,
    EXTRAPOLATION_BUDGET,
    TEST_RE_TAU,
    Hyperparameters,
)
from .solver import SolveResult, solve_channel

__all__ = [
    "ALPHAS",
    "CouplingRun",
    "CouplingResult",
    "classify_coupling_status",
    "train_frozen_full_model",
    "run_alpha_scan",
    "run_coupling_experiment",
    "save_coupling_artifacts",
    "plot_coupling_history",
]


ALPHAS: Tuple[float, ...] = (1.0, 0.5, 0.2)
DEFAULT_COUPLING_MAX_ITER = 200


@dataclass(frozen=True)
class CouplingRun:
    """One frozen model/seed and one relaxation factor."""

    seed: int
    alpha: float
    status: str
    converged: bool
    iterations: int
    final_residual: float
    final_field_change: float
    residual_monotone: bool
    profile_rel_l2: float
    baseline_profile_rel_l2: float
    wall_shear_plus: float
    truth_wall_shear_plus: float
    wall_shear_rel_error: float
    model_weight_drift: float
    residual_norms: Tuple[float, ...]
    step_norms: Tuple[float, ...]


@dataclass
class CouplingResult:
    """All per-run histories plus their mean/std summary rows."""

    runs: List[CouplingRun]
    summary_rows: List[Dict[str, object]]
    hyperparameters: Hyperparameters
    Re_tau: float
    budget: int
    tol: float
    max_iter: int


def _parameter_snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _parameter_drift(
    snapshot: Mapping[str, torch.Tensor], model: torch.nn.Module
) -> float:
    drifts = []
    for name, parameter in model.named_parameters():
        drifts.append(float((parameter.detach() - snapshot[name]).abs().max()))
    return max(drifts, default=0.0)


def _wall_shear_plus(solved: SolveResult, Re_tau: float) -> float:
    """One-sided wall derivative in wall units, from the solved profile.

    The first cell is intentionally used instead of reading the analytic stress
    relation back out: the latter would make every closure look exact by
    construction and would not test the coupled profile.
    """
    if solved.y.numel() < 2 or float(solved.y[1] - solved.y[0]) == 0.0:
        raise ValueError("wall shear requires at least two distinct grid points")
    derivative = (solved.U[1] - solved.U[0]) / (solved.y[1] - solved.y[0])
    return float(derivative / Re_tau)


def _residual_monotone(residual_norms: Sequence[float]) -> bool:
    """Use the same round-off slack as the solver's convergence decision."""
    return all(
        later <= earlier * (1.0 + 1e-6)
        for earlier, later in zip(residual_norms, residual_norms[1:])
    )


def classify_coupling_status(
    solved: SolveResult,
    *,
    tol: float = TRAINING_SOLVE_TOL,
) -> str:
    """Classify an honest coupling outcome from its residual history."""
    if solved.converged:
        return "converged"
    residual = list(solved.residual_norms)
    if not residual or not all(math.isfinite(value) for value in residual):
        return "non_finite"
    if residual[-1] > residual[0] * (1.0 + 1e-6):
        return "diverged"
    if not _residual_monotone(residual):
        return "oscillated"
    if residual[-1] < tol and solved.step_norms[-1] < tol:
        return "stalled"
    return "stalled"


def train_frozen_full_model(
    seed: int,
    hyperparameters: Hyperparameters,
    *,
    epochs: int = DEFAULT_EPOCHS,
    width: int = DEFAULT_MODEL_WIDTH,
    depth: int = DEFAULT_MODEL_DEPTH,
    budget: int = EXTRAPOLATION_BUDGET,
    verbose: bool = False,
) -> Tuple[ChannelProblem, MLP, Dict[str, List[float]]]:
    """Train the full model exactly as T7 did, then freeze every weight."""
    problem = ChannelProblem(seed=seed)
    model = MLP(3, 1, width=width, depth=depth, seed=seed)
    history = train(
        problem,
        model,
        TrainConfig(
            epochs=epochs,
            lr=hyperparameters.learning_rate,
            n_interior=budget,
            device="cpu",
            seed=seed,
            verbose=verbose,
        ),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return problem, model, history


def _run_one_alpha(
    *,
    seed: int,
    alpha: float,
    problem: ChannelProblem,
    model: torch.nn.Module,
    snapshot: Mapping[str, torch.Tensor],
    truth: SolveResult,
    baseline_error: float,
    Re_tau: float,
    tol: float,
    max_iter: int,
) -> CouplingRun:
    closure = ChannelClosure(model, problem.base, structure=True)
    solved = solve_channel(
        closure,
        Re_tau,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
    )
    corrected_error = relative_l2(solved.U, truth.U)
    wall_shear = _wall_shear_plus(solved, Re_tau)
    truth_wall_shear = _wall_shear_plus(truth, Re_tau)
    wall_shear_error = (
        abs(wall_shear - truth_wall_shear) / abs(truth_wall_shear)
        if truth_wall_shear
        else math.inf
    )
    residual = list(solved.residual_norms)
    steps = list(solved.step_norms)
    return CouplingRun(
        seed=seed,
        alpha=float(alpha),
        status=classify_coupling_status(solved, tol=tol),
        converged=bool(solved.converged),
        iterations=int(solved.iterations),
        final_residual=float(residual[-1]) if residual else math.inf,
        final_field_change=float(steps[-1]) if steps else math.inf,
        residual_monotone=_residual_monotone(residual),
        profile_rel_l2=float(corrected_error),
        baseline_profile_rel_l2=float(baseline_error),
        wall_shear_plus=wall_shear,
        truth_wall_shear_plus=truth_wall_shear,
        wall_shear_rel_error=float(wall_shear_error),
        model_weight_drift=_parameter_drift(snapshot, model),
        residual_norms=tuple(float(value) for value in residual),
        step_norms=tuple(float(value) for value in steps),
    )


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    values = list(values)
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return float(mean), float(std)


def _summarize_coupling(runs: Sequence[CouplingRun]) -> List[Dict[str, object]]:
    grouped: Dict[float, List[CouplingRun]] = {}
    for run in runs:
        grouped.setdefault(run.alpha, []).append(run)

    rows: List[Dict[str, object]] = []
    for alpha, alpha_runs in sorted(grouped.items()):
        row: Dict[str, object] = {
            "alpha": alpha,
            "n_seeds": len(alpha_runs),
            "n_converged": sum(int(run.converged) for run in alpha_runs),
            "n_residual_monotone": sum(
                int(run.residual_monotone) for run in alpha_runs
            ),
            "all_residual_monotone": all(
                run.residual_monotone for run in alpha_runs
            ),
            "statuses": "; ".join(
                f"seed{run.seed}={run.status}" for run in sorted(alpha_runs, key=lambda r: r.seed)
            ),
        }
        for field_name in (
            "iterations",
            "final_residual",
            "final_field_change",
            "profile_rel_l2",
            "baseline_profile_rel_l2",
            "wall_shear_plus",
            "truth_wall_shear_plus",
            "wall_shear_rel_error",
            "model_weight_drift",
        ):
            values = [float(getattr(run, field_name)) for run in alpha_runs]
            mean, std = _mean_std(values)
            row[f"{field_name}_mean"] = mean
            row[f"{field_name}_std"] = std
        rows.append(row)
    return rows


def run_alpha_scan(
    hyperparameters: Hyperparameters,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    alphas: Sequence[float] = ALPHAS,
    epochs: int = DEFAULT_EPOCHS,
    width: int = DEFAULT_MODEL_WIDTH,
    depth: int = DEFAULT_MODEL_DEPTH,
    budget: int = EXTRAPOLATION_BUDGET,
    Re_tau: float = TEST_RE_TAU,
    tol: float = TRAINING_SOLVE_TOL,
    max_iter: int = DEFAULT_COUPLING_MAX_ITER,
    verbose: bool = False,
) -> CouplingResult:
    """Train/freeze one model per seed and run every requested ``alpha``."""
    if not seeds:
        raise ValueError("at least one seed is required")
    if not alphas:
        raise ValueError("at least one alpha is required")
    for alpha in alphas:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    runs: List[CouplingRun] = []
    for seed in seeds:
        if verbose:
            print(f"[T8] training/freezing full model seed={seed}")
        problem, model, _ = train_frozen_full_model(
            seed,
            hyperparameters,
            epochs=epochs,
            width=width,
            depth=depth,
            budget=budget,
            verbose=verbose,
        )
        snapshot = _parameter_snapshot(model)
        truth = problem.solved_truth(Re_tau)
        baseline = solve_channel(
            problem.base,
            Re_tau,
            y=truth.y,
            tol=tol,
            max_iter=max_iter,
        )
        baseline_error = relative_l2(baseline.U, truth.U)
        for alpha in alphas:
            if verbose:
                print(f"[T8] alpha={alpha:g} seed={seed}")
            runs.append(
                _run_one_alpha(
                    seed=seed,
                    alpha=float(alpha),
                    problem=problem,
                    model=model,
                    snapshot=snapshot,
                    truth=truth,
                    baseline_error=baseline_error,
                    Re_tau=Re_tau,
                    tol=tol,
                    max_iter=max_iter,
                )
            )

    return CouplingResult(
        runs=runs,
        summary_rows=_summarize_coupling(runs),
        hyperparameters=hyperparameters,
        Re_tau=float(Re_tau),
        budget=budget,
        tol=float(tol),
        max_iter=int(max_iter),
    )


run_coupling_experiment = run_alpha_scan


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_coupling_history(result: CouplingResult, path: Path) -> None:
    """Plot every residual history; do not average away failures."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = plt.cm.tab10.colors
    alphas = sorted({run.alpha for run in result.runs}, reverse=True)
    for index, alpha in enumerate(alphas):
        color = colors[index % len(colors)]
        alpha_runs = sorted(
            (run for run in result.runs if run.alpha == alpha),
            key=lambda run: run.seed,
        )
        for run in alpha_runs:
            ax.plot(
                range(1, len(run.residual_norms) + 1),
                run.residual_norms,
                color=color,
                alpha=0.55,
                linewidth=1.2,
                label=f"alpha={alpha:g}" if run is alpha_runs[0] else None,
            )

    ax.axhline(
        result.tol,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"tol={result.tol:g}",
    )
    ax.set_yscale("log")
    ax.set_xlabel("fixed-point iteration")
    ax.set_ylabel("max residual")
    ax.set_title(f"Frozen closure coupling at Re_tau={result.Re_tau:g}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(path))


def save_coupling_artifacts(
    result: CouplingResult,
    out_dir: Path,
    *,
    figure_path: Optional[Path] = None,
) -> None:
    """Write summary, raw histories, report JSON, and the residual plot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_rows: List[Dict[str, object]] = []
    for run in result.runs:
        for iteration, (residual, step) in enumerate(
            zip(run.residual_norms, run.step_norms), start=1
        ):
            history_rows.append(
                {
                    "seed": run.seed,
                    "alpha": run.alpha,
                    "iteration": iteration,
                    "residual": residual,
                    "field_change": step,
                }
            )

    _write_csv(out_dir / "coupling_history.csv", history_rows)
    _write_csv(out_dir / "coupling_summary.csv", result.summary_rows)
    report = {
        "Re_tau": result.Re_tau,
        "budget": result.budget,
        "tol": result.tol,
        "max_iter": result.max_iter,
        "hyperparameters": asdict(result.hyperparameters),
        "status_counts": dict(
            Counter(run.status for run in result.runs)
        ),
        "summary": result.summary_rows,
        "runs": [
            {
                **{
                    key: value
                    for key, value in asdict(run).items()
                    if key not in {"residual_norms", "step_norms"}
                },
                "residual_norms": list(run.residual_norms),
                "step_norms": list(run.step_norms),
            }
            for run in result.runs
        ],
    }
    with (out_dir / "coupling_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    if figure_path is None:
        figure_path = out_dir / "coupling_residual_history.png"
    plot_coupling_history(result, Path(figure_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        default=Path("RANS_PINN_Demo/results/frozen_hyperparameters.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("RANS_PINN_Demo/results"))
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("RANS_PINN_Demo/figures/coupling_residual_history.png"),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--width", type=int, default=DEFAULT_MODEL_WIDTH)
    parser.add_argument("--depth", type=int, default=DEFAULT_MODEL_DEPTH)
    parser.add_argument("--budget", type=int, default=EXTRAPOLATION_BUDGET)
    parser.add_argument("--Re-tau", type=float, default=TEST_RE_TAU)
    parser.add_argument("--tol", type=float, default=TRAINING_SOLVE_TOL)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_COUPLING_MAX_ITER)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw = json.loads(Path(args.hyperparameters).read_text(encoding="utf-8"))
    hyperparameters = Hyperparameters(**raw)
    result = run_alpha_scan(
        hyperparameters,
        seeds=tuple(args.seeds),
        alphas=tuple(args.alphas),
        epochs=args.epochs,
        width=args.width,
        depth=args.depth,
        budget=args.budget,
        Re_tau=args.Re_tau,
        tol=args.tol,
        max_iter=args.max_iter,
        verbose=args.verbose,
    )
    save_coupling_artifacts(result, args.out_dir, figure_path=args.figure)
    print("Status counts:", dict(Counter(run.status for run in result.runs)))
    print(f"Wrote {len(result.runs)} coupling runs to {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
