"""T6 experiment harness: validation selection and the data-volume sweep.

This is deliberately a thin orchestration layer.  The physics and the model are
already separate seams; what T6 adds is the experimental discipline around
them:

* one frozen hyperparameter choice, selected on held-out points at Re_tau=395;
* every ablation arm trained on the same nested 8/12/16 budgets and seeds;
* per-seed records plus a mean +/- std summary table;
* a data-volume curve that plots every seed, not only the best one.

The runner is CPU-only because ``ChannelProblem.losses`` solves on the CPU.  A
full default sweep is a few minutes on this machine; tests use the same public
functions with a tiny one-epoch configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from pinn.evaluate import relative_l2
from pinn.model import MLP
from pinn.plotting import plt, save_figure
from pinn.trainer import TrainConfig, train

from .ablations import (
    ABLATIONS,
    AMPLITUDE_LOSS_WEIGHT,
    RANS_LOSS_WEIGHT,
    SMOOTH_LOSS_WEIGHT,
    AblationConfig,
)
from .channel import (
    DATA_BUDGETS,
    ChannelClosure,
    ChannelProblem,
    _band_points,
    _profile_at,
    _strata,
    stratified_points,
)
from .solver import solve_channel, wall_clustered_grid

__all__ = [
    "Hyperparameters",
    "RunSpec",
    "ProfileMetrics",
    "RunResult",
    "ExperimentResult",
    "validation_points",
    "validation_error",
    "profile_metrics",
    "DEFAULT_HYPERPARAMETER_CANDIDATES",
    "DEFAULT_SEEDS",
    "select_hyperparameters",
    "run_experiment",
    "save_experiment",
]

VALIDATION_RE_TAU = 395.0
DEFAULT_SELECTION_BUDGET = 16
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_MODEL_WIDTH = 32
DEFAULT_MODEL_DEPTH = 2
DEFAULT_EPOCHS = 300


@dataclass(frozen=True)
class Hyperparameters:
    """Knobs selected on validation and then frozen for every experiment arm."""

    learning_rate: float = 1e-2
    rans_weight: float = RANS_LOSS_WEIGHT
    smooth_weight: float = SMOOTH_LOSS_WEIGHT
    amplitude_weight: float = AMPLITUDE_LOSS_WEIGHT


#: A small, intentionally discrete candidate grid.  Keeping it finite and
#: publishing it before the run is the point: the validation choice must not
#: become a second, unrecorded experiment.
DEFAULT_HYPERPARAMETER_CANDIDATES: Tuple[Hyperparameters, ...] = (
    Hyperparameters(learning_rate=1e-2, rans_weight=1e-2, smooth_weight=1e-3),
    Hyperparameters(learning_rate=3e-2, rans_weight=1e-2, smooth_weight=1e-3),
    Hyperparameters(learning_rate=1e-2, rans_weight=1e-1, smooth_weight=1e-3),
    Hyperparameters(learning_rate=3e-2, rans_weight=1e-1, smooth_weight=1e-3),
    Hyperparameters(learning_rate=1e-2, rans_weight=1e-2, smooth_weight=1e-2),
    Hyperparameters(learning_rate=3e-2, rans_weight=1e-2, smooth_weight=1e-2),
)


@dataclass(frozen=True)
class RunSpec:
    """One arm/budget/seed cell of the sweep."""

    arm: str
    budget: int
    seed: int


@dataclass(frozen=True)
class ProfileMetrics:
    """Errors on the dense evaluation grid at one training Reynolds number."""

    Re_tau: float
    baseline_rel_l2: float
    corrected_rel_l2: float
    improvement: float
    converged: bool
    iterations: int


@dataclass
class RunResult:
    """One completed training run and the profiles it was scored on."""

    spec: RunSpec
    validation_rel_l2: float
    validation_converged: bool
    profiles: Tuple[ProfileMetrics, ...]
    history: Dict[str, List[float]]


@dataclass
class ExperimentResult:
    """Everything needed to write the T6 tables and figure."""

    selected: Hyperparameters
    selection_records: List[Dict[str, object]]
    runs: List[RunResult]
    summary_rows: List[Dict[str, object]]


def validation_points(Re_tau: float = VALIDATION_RE_TAU) -> torch.Tensor:
    """Held-out validation points for the validation Reynolds number.

    They sit geometrically between the four candidate points of each training
    band.  That makes them disjoint from all nested training budgets while
    keeping the validation distribution stratified in the same way as the
    observations.
    """
    bands = _strata(Re_tau)
    pieces = []
    for index, (lo, hi) in enumerate(bands):
        candidates = _band_points(lo, hi, interior=0 < index < len(bands) - 1)
        # Geometric midpoints: positive by construction, and never equal to an
        # endpoint already used by training.
        pieces.append(torch.sqrt(candidates[:-1] * candidates[1:]))
    return (torch.cat(pieces) / Re_tau).reshape(-1, 1)


def _assert_validation_split_is_clean() -> None:
    """Fail early if a future sampling change leaks validation into training."""
    validation = validation_points().reshape(-1)
    for budget in DATA_BUDGETS:
        training = stratified_points(budget, VALIDATION_RE_TAU).reshape(-1)
        for value in validation:
            if torch.any(torch.isclose(training, value, rtol=0.0, atol=1e-12)):
                raise AssertionError("validation point overlaps a training observation")
    grid = wall_clustered_grid().reshape(-1)
    for value in validation:
        if torch.any(torch.isclose(grid, value, rtol=0.0, atol=1e-12)):
            raise AssertionError("validation point overlaps the evaluation grid")


def _make_problem(
    ablation: AblationConfig,
    seed: int,
    hyperparameters: Hyperparameters,
) -> ChannelProblem:
    return ChannelProblem(
        seed=seed,
        ablation=ablation,
        rans_weight=hyperparameters.rans_weight,
        smooth_weight=hyperparameters.smooth_weight,
        amplitude_weight=hyperparameters.amplitude_weight,
    )


def _make_model(seed: int, width: int, depth: int) -> MLP:
    return MLP(3, 1, width=width, depth=depth, seed=seed)


def _solved_closure(problem: ChannelProblem, model: torch.nn.Module, Re_tau: float):
    closure = ChannelClosure(model, problem.base, structure=problem.ablation.structure)
    return solve_channel(
        closure,
        Re_tau,
        max_iter=problem.solve_max_iter,
        tol=problem.solve_tol,
    )


def _validation_metrics(
    problem: ChannelProblem, model: torch.nn.Module
) -> Tuple[float, bool, int]:
    """Noiseless profile error and solve status on the held-out 395 points."""
    model.eval()
    y = validation_points().to(torch.float64)
    with torch.no_grad():
        solved = _solved_closure(problem, model, VALIDATION_RE_TAU)
        predicted = _profile_at(solved, y)
        target = problem.reference(y, VALIDATION_RE_TAU)
    return relative_l2(predicted, target), bool(solved.converged), int(solved.iterations)


def validation_error(problem: ChannelProblem, model: torch.nn.Module) -> float:
    """Noiseless profile error on the held-out 395 points."""
    return _validation_metrics(problem, model)[0]


def profile_metrics(
    problem: ChannelProblem,
    model: torch.nn.Module,
    Re_tau: float,
) -> ProfileMetrics:
    """Score the learned closure against baseline and truth on a dense grid."""
    model.eval()
    with torch.no_grad():
        solved = _solved_closure(problem, model, Re_tau)
        baseline = solve_channel(
            problem.base,
            Re_tau,
            y=solved.y,
            max_iter=problem.solve_max_iter,
            tol=problem.solve_tol,
        )
        reference = problem.reference(solved.y.reshape(-1), Re_tau)
        baseline_error = relative_l2(baseline.U, reference)
        corrected_error = relative_l2(solved.U, reference)
    improvement = 1.0 - corrected_error / baseline_error if baseline_error else 0.0
    return ProfileMetrics(
        Re_tau=float(Re_tau),
        baseline_rel_l2=float(baseline_error),
        corrected_rel_l2=float(corrected_error),
        improvement=float(improvement),
        converged=bool(solved.converged),
        iterations=int(solved.iterations),
    )


def _train_once(
    problem: ChannelProblem,
    budget: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    width: int,
    depth: int,
    verbose: bool,
) -> Tuple[MLP, Dict[str, List[float]]]:
    model = _make_model(seed, width, depth)
    history = train(
        problem,
        model,
        TrainConfig(
            epochs=epochs,
            lr=learning_rate,
            n_interior=budget,
            device="cpu",
            seed=seed,
            verbose=verbose,
        ),
    )
    return model, history


def select_hyperparameters(
    candidates: Sequence[Hyperparameters] = DEFAULT_HYPERPARAMETER_CANDIDATES,
    *,
    seed: int = 0,
    budget: int = DEFAULT_SELECTION_BUDGET,
    epochs: int = DEFAULT_EPOCHS,
    width: int = DEFAULT_MODEL_WIDTH,
    depth: int = DEFAULT_MODEL_DEPTH,
    verbose: bool = False,
) -> Tuple[Hyperparameters, List[Dict[str, object]]]:
    """Choose on Re_tau=395's held-out points, then return the frozen choice."""
    if not candidates:
        raise ValueError("at least one hyperparameter candidate is required")
    if budget not in DATA_BUDGETS:
        raise ValueError(f"selection budget must be one of {DATA_BUDGETS}, got {budget}")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    records: List[Dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        problem = _make_problem(AblationConfig(), seed, candidate)
        model = _make_model(seed, width, depth)
        history = train(
            problem,
            model,
            TrainConfig(
                epochs=epochs,
                lr=candidate.learning_rate,
                n_interior=budget,
                device="cpu",
                seed=seed,
                verbose=verbose,
            ),
        )
        try:
            score, converged, iterations = _validation_metrics(problem, model)
        except (AssertionError, RuntimeError, FloatingPointError) as exc:
            if verbose:
                print(f"  candidate {index} failed validation: {exc}")
            score, converged, iterations = math.inf, False, 0
        records.append(
            {
                "candidate": asdict(candidate),
                "validation_rel_l2": score,
                "validation_converged": converged,
                "validation_iterations": iterations,
                "final_data_loss": history["data"][-1],
            }
        )

    best_index = min(
        range(len(records)),
        key=lambda i: (float(records[i]["validation_rel_l2"]), i),
    )
    return candidates[best_index], records


def _variant_run(
    spec: RunSpec,
    hyperparameters: Hyperparameters,
    *,
    epochs: int,
    width: int,
    depth: int,
    verbose: bool,
) -> RunResult:
    problem = _make_problem(ABLATIONS[spec.arm], spec.seed, hyperparameters)
    model, history = _train_once(
        problem,
        spec.budget,
        spec.seed,
        epochs,
        hyperparameters.learning_rate,
        width,
        depth,
        verbose,
    )
    profiles = tuple(profile_metrics(problem, model, Re_tau) for Re_tau in problem.Re_taus)
    validation, validation_converged, _ = _validation_metrics(problem, model)
    return RunResult(
        spec=spec,
        validation_rel_l2=validation,
        validation_converged=validation_converged,
        profiles=profiles,
        history=history,
    )


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    values = list(values)
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return float(mean), float(std)


def _summarize(runs: Sequence[RunResult]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, float], List[ProfileMetrics]] = {}
    for run in runs:
        for profile in run.profiles:
            grouped.setdefault(
                (run.spec.arm, run.spec.budget, profile.Re_tau), []
            ).append(profile)

    rows: List[Dict[str, object]] = []
    for (arm, budget, Re_tau), profiles in sorted(grouped.items()):
        baseline_mean, baseline_std = _mean_std([p.baseline_rel_l2 for p in profiles])
        corrected_mean, corrected_std = _mean_std([p.corrected_rel_l2 for p in profiles])
        improvement_mean, improvement_std = _mean_std([p.improvement for p in profiles])
        rows.append(
            {
                "arm": arm,
                "budget": budget,
                "Re_tau": Re_tau,
                "n_seeds": len(profiles),
                "baseline_rel_l2_mean": baseline_mean,
                "baseline_rel_l2_std": baseline_std,
                "corrected_rel_l2_mean": corrected_mean,
                "corrected_rel_l2_std": corrected_std,
                "improvement_mean": improvement_mean,
                "improvement_std": improvement_std,
            }
        )
    return rows


def run_experiment(
    *,
    candidates: Sequence[Hyperparameters] = DEFAULT_HYPERPARAMETER_CANDIDATES,
    arms: Sequence[str] = tuple(ABLATIONS),
    budgets: Sequence[int] = DATA_BUDGETS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    width: int = DEFAULT_MODEL_WIDTH,
    depth: int = DEFAULT_MODEL_DEPTH,
    selection_seed: int = 0,
    selection_budget: int = DEFAULT_SELECTION_BUDGET,
    select: bool = True,
    verbose: bool = False,
) -> ExperimentResult:
    """Run the T6 grid and return in-memory results without writing files."""
    _assert_validation_split_is_clean()
    if not arms:
        raise ValueError("at least one arm is required")
    if not budgets:
        raise ValueError("at least one data budget is required")
    if not seeds:
        raise ValueError("at least one seed is required")
    unknown = [arm for arm in arms if arm not in ABLATIONS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {sorted(ABLATIONS)}")
    for budget in budgets:
        if budget not in DATA_BUDGETS:
            raise ValueError(f"budget must be one of {DATA_BUDGETS}, got {budget}")

    if select:
        selected, selection_records = select_hyperparameters(
            candidates,
            seed=selection_seed,
            budget=selection_budget,
            epochs=epochs,
            width=width,
            depth=depth,
            verbose=verbose,
        )
    else:
        if not candidates:
            raise ValueError("no candidate available when selection is disabled")
        selected = candidates[0]
        selection_records = []

    runs: List[RunResult] = []
    for arm in arms:
        for budget in budgets:
            for seed in seeds:
                spec = RunSpec(arm=arm, budget=budget, seed=seed)
                if verbose:
                    print(f"[T6] {arm} budget={budget} seed={seed}")
                runs.append(
                    _variant_run(
                        spec,
                        selected,
                        epochs=epochs,
                        width=width,
                        depth=depth,
                        verbose=verbose,
                    )
                )

    return ExperimentResult(
        selected=selected,
        selection_records=selection_records,
        runs=runs,
        summary_rows=_summarize(runs),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_data_volume(result: ExperimentResult, path: Path) -> None:
    """Plot mean +/- sample std with every seed shown as a marker."""
    per_seed: Dict[Tuple[str, int], List[float]] = {}
    for run in result.runs:
        values = [p.improvement for p in run.profiles]
        per_seed.setdefault((run.spec.arm, run.spec.budget), []).append(
            sum(values) / len(values)
        )

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    arms = list(dict.fromkeys(run.spec.arm for run in result.runs))
    budgets = sorted({run.spec.budget for run in result.runs})
    colors = plt.cm.tab10.colors

    for arm_index, arm in enumerate(arms):
        color = colors[arm_index % len(colors)]
        means, errors = [], []
        for budget in budgets:
            values = per_seed.get((arm, budget), [])
            mean, std = _mean_std(values)
            means.append(100.0 * mean)
            errors.append(100.0 * std)
            offsets = [
                (i - (len(values) - 1) / 2.0) * 0.07 for i in range(len(values))
            ]
            ax.scatter(
                [budget + offset for offset in offsets],
                [100.0 * value for value in values],
                color=color,
                s=16,
                alpha=0.55,
                zorder=3,
            )
        ax.errorbar(
            budgets,
            means,
            yerr=errors,
            marker="o",
            capsize=3,
            linewidth=1.6,
            color=color,
            label=arm,
            zorder=2,
        )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("observations per training Re_tau")
    ax.set_ylabel("U+ relative L2 improvement vs baseline (%)")
    ax.set_title("T6 data-volume sweep (mean over Re_tau=180,395; all seeds shown)")
    ax.set_xticks(budgets)
    ax.grid(True, alpha=0.25)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(path))


def save_experiment(
    result: ExperimentResult,
    out_dir: Path,
    *,
    figure_path: Optional[Path] = None,
) -> None:
    """Write per-seed archives, summary tables, and the data-volume figure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "frozen_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(result.selected), handle, indent=2)
        handle.write("\n")

    selection_rows = []
    for record in result.selection_records:
        candidate = record["candidate"]
        assert isinstance(candidate, dict)
        selection_rows.append(
            {
                **candidate,
                "validation_rel_l2": record["validation_rel_l2"],
                "validation_converged": record["validation_converged"],
                "validation_iterations": record["validation_iterations"],
                "final_data_loss": record["final_data_loss"],
            }
        )
    if selection_rows:
        _write_csv(out_dir / "validation_selection.csv", selection_rows)

    main_rows: List[Dict[str, object]] = []
    for run in result.runs:
        for profile in run.profiles:
            main_rows.append(
                {
                    "arm": run.spec.arm,
                    "budget": run.spec.budget,
                    "seed": run.spec.seed,
                    "Re_tau": profile.Re_tau,
                    "baseline_rel_l2": profile.baseline_rel_l2,
                    "corrected_rel_l2": profile.corrected_rel_l2,
                    "improvement": profile.improvement,
                    "converged": profile.converged,
                    "iterations": profile.iterations,
                    "validation_rel_l2": run.validation_rel_l2,
                    "validation_converged": run.validation_converged,
                }
            )

        run_path = runs_dir / f"{run.spec.arm}_budget{run.spec.budget}_seed{run.spec.seed}.json"
        with run_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "spec": asdict(run.spec),
                    "validation_rel_l2": run.validation_rel_l2,
                    "validation_converged": run.validation_converged,
                    "profiles": [asdict(profile) for profile in run.profiles],
                    "history": run.history,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
    _write_csv(out_dir / "main_runs.csv", main_rows)
    _write_csv(out_dir / "error_table.csv", result.summary_rows)

    if figure_path is None:
        figure_path = out_dir / "data_volume_curve.png"
    plot_data_volume(result, Path(figure_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("RANS_PINN_Demo/results"))
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("RANS_PINN_Demo/figures/data_volume_curve.png"),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DATA_BUDGETS))
    parser.add_argument("--arms", nargs="+", choices=sorted(ABLATIONS), default=sorted(ABLATIONS))
    parser.add_argument("--width", type=int, default=DEFAULT_MODEL_WIDTH)
    parser.add_argument("--depth", type=int, default=DEFAULT_MODEL_DEPTH)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--selection-budget", type=int, default=DEFAULT_SELECTION_BUDGET)
    parser.add_argument("--no-select", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_experiment(
        arms=tuple(args.arms),
        budgets=tuple(args.budgets),
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        width=args.width,
        depth=args.depth,
        selection_seed=args.selection_seed,
        selection_budget=args.selection_budget,
        select=not args.no_select,
        verbose=args.verbose,
    )
    save_experiment(result, args.out_dir, figure_path=args.figure)
    print("Frozen hyperparameters:", asdict(result.selected))
    print(f"Wrote {len(result.runs)} runs to {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

