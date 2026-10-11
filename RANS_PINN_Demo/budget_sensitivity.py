"""Budget-sensitivity audit for the pre-2D RANS/PINN demo.

The original T7 sweep used three seeds and read its E3/E4 gate at the 12-point
budget.  That made the apparent budget=12 cliff vulnerable to a single noise
realization: the original table answered whether every seed converged, but not
whether 12 is intrinsically worse than 8 or 16.  This module performs the
missing robustness audit and keeps failure counts separate from accuracy means.

It is intentionally a diagnostic, not a tuning pass.  Hyperparameters are the
frozen T7 values supplied by the caller; no candidate selection happens here.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .ablations import EXPERIMENT_ARMS
from .channel import DATA_BUDGETS, NOISE_LEVEL, ChannelProblem
from .experiment import (
    DEFAULT_EPOCHS,
    DEFAULT_MODEL_DEPTH,
    DEFAULT_MODEL_WIDTH,
    TEST_RE_TAU,
    Hyperparameters,
    _train_once,
    profile_metrics,
)

__all__ = [
    "BudgetAuditRun",
    "BudgetAuditResult",
    "run_budget_audit",
    "save_budget_audit",
]

DEFAULT_AUDIT_SEEDS: Tuple[int, ...] = tuple(range(20))
DEFAULT_CONTROL_SEEDS: Tuple[int, ...] = (2, 4, 11)


@dataclass(frozen=True)
class BudgetAuditRun:
    """One arm/budget/seed cell, with an optional noiseless control."""

    arm: str
    budget: int
    seed: int
    noisy_converged: bool
    noisy_corrected_rel_l2: float
    noisy_iterations: int
    noise_free_converged: Optional[bool]
    noise_free_corrected_rel_l2: Optional[float]
    noise_free_iterations: Optional[int]


@dataclass
class BudgetAuditResult:
    """Raw cells and deterministic summaries for the budget-sensitivity audit."""

    seeds: Tuple[int, ...]
    control_seeds: Tuple[int, ...]
    hyperparameters: Hyperparameters
    runs: List[BudgetAuditRun]
    summary_rows: List[Dict[str, object]]


def _problem(
    arm: str,
    seed: int,
    hyperparameters: Hyperparameters,
    *,
    noise: float,
) -> ChannelProblem:
    return ChannelProblem(
        seed=seed,
        ablation=EXPERIMENT_ARMS[arm],
        noise=noise,
        rans_weight=hyperparameters.rans_weight,
        smooth_weight=hyperparameters.smooth_weight,
        amplitude_weight=hyperparameters.amplitude_weight,
    )


def _score_cell(
    arm: str,
    budget: int,
    seed: int,
    hyperparameters: Hyperparameters,
    *,
    noise: float,
    epochs: int,
    width: int,
    depth: int,
    verbose: bool,
):
    problem = _problem(arm, seed, hyperparameters, noise=noise)
    model, _ = _train_once(
        problem,
        budget,
        seed,
        epochs,
        hyperparameters.learning_rate,
        width,
        depth,
        verbose,
    )
    return profile_metrics(problem, model, TEST_RE_TAU)


def _summary(runs: Sequence[BudgetAuditRun]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[BudgetAuditRun]] = {}
    for run in runs:
        grouped.setdefault((run.arm, run.budget), []).append(run)

    rows: List[Dict[str, object]] = []
    for (arm, budget), cell_runs in sorted(grouped.items()):
        converged = [run for run in cell_runs if run.noisy_converged]
        controls = [run for run in cell_runs if run.noise_free_converged is not None]
        conditional = [run.noisy_corrected_rel_l2 for run in converged]
        rows.append(
            {
                "arm": arm,
                "budget": budget,
                "n_seeds": len(cell_runs),
                "n_converged": len(converged),
                "failure_rate": 1.0 - len(converged) / len(cell_runs),
                "conditional_median_rel_l2": (
                    statistics.median(conditional) if conditional else None
                ),
                "conditional_max_rel_l2": max(conditional) if conditional else None,
                "n_noise_free_controls": len(controls),
                "noise_free_converged": sum(
                    int(bool(run.noise_free_converged)) for run in controls
                ),
                "noise_only_failures_among_controls": sum(
                    int(
                        (not run.noisy_converged)
                        and bool(run.noise_free_converged)
                    )
                    for run in controls
                ),
            }
        )
    return rows


def run_budget_audit(
    hyperparameters: Hyperparameters,
    *,
    arms: Sequence[str] = ("all_off", "full"),
    budgets: Sequence[int] = DATA_BUDGETS,
    seeds: Sequence[int] = DEFAULT_AUDIT_SEEDS,
    control_seeds: Sequence[int] = DEFAULT_CONTROL_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    width: int = DEFAULT_MODEL_WIDTH,
    depth: int = DEFAULT_MODEL_DEPTH,
    verbose: bool = False,
) -> BudgetAuditResult:
    """Run the noisy budget sweep plus targeted noiseless controls."""
    if not arms:
        raise ValueError("at least one arm is required")
    if not budgets:
        raise ValueError("at least one budget is required")
    if not seeds:
        raise ValueError("at least one seed is required")
    unknown = [arm for arm in arms if arm not in EXPERIMENT_ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {sorted(EXPERIMENT_ARMS)}")
    for budget in budgets:
        if budget not in DATA_BUDGETS:
            raise ValueError(f"budget must be one of {DATA_BUDGETS}, got {budget}")

    controls = set(control_seeds)
    runs: List[BudgetAuditRun] = []
    for arm in arms:
        for budget in budgets:
            for seed in seeds:
                if verbose:
                    print(f"[audit] {arm} budget={budget} seed={seed}")
                noisy = _score_cell(
                    arm,
                    budget,
                    seed,
                    hyperparameters,
                    noise=NOISE_LEVEL,
                    epochs=epochs,
                    width=width,
                    depth=depth,
                    verbose=verbose,
                )
                control = None
                if seed in controls:
                    control = _score_cell(
                        arm,
                        budget,
                        seed,
                        hyperparameters,
                        noise=0.0,
                        epochs=epochs,
                        width=width,
                        depth=depth,
                        verbose=verbose,
                    )
                runs.append(
                    BudgetAuditRun(
                        arm=arm,
                        budget=budget,
                        seed=seed,
                        noisy_converged=noisy.converged,
                        noisy_corrected_rel_l2=noisy.corrected_rel_l2,
                        noisy_iterations=noisy.iterations,
                        noise_free_converged=(
                            None if control is None else control.converged
                        ),
                        noise_free_corrected_rel_l2=(
                            None if control is None else control.corrected_rel_l2
                        ),
                        noise_free_iterations=(
                            None if control is None else control.iterations
                        ),
                    )
                )

    return BudgetAuditResult(
        seeds=tuple(seeds),
        control_seeds=tuple(control_seeds),
        hyperparameters=hyperparameters,
        runs=runs,
        summary_rows=_summary(runs),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_budget_audit(result: BudgetAuditResult, out_dir: Path) -> None:
    """Write per-cell and summary artifacts."""
    out_dir = Path(out_dir)
    _write_csv(out_dir / "budget_sensitivity_runs.csv", [asdict(run) for run in result.runs])
    _write_csv(out_dir / "budget_sensitivity.csv", result.summary_rows)
    report = {
        "description": (
            "Noisy 8/12/16 budget audit with targeted noiseless controls. "
            "Failure counts are reported separately from conditional errors."
        ),
        "seeds": list(result.seeds),
        "control_seeds": list(result.control_seeds),
        "hyperparameters": asdict(result.hyperparameters),
        "summary": result.summary_rows,
        "runs": [asdict(run) for run in result.runs],
    }
    (out_dir / "budget_sensitivity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("RANS_PINN_Demo/results"))
    parser.add_argument("--hyperparameters", type=Path, default=Path("RANS_PINN_Demo/results/frozen_hyperparameters.json"))
    parser.add_argument("--arms", nargs="+", default=["all_off", "full"])
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DATA_BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_AUDIT_SEEDS))
    parser.add_argument("--control-seeds", type=int, nargs="+", default=list(DEFAULT_CONTROL_SEEDS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--width", type=int, default=DEFAULT_MODEL_WIDTH)
    parser.add_argument("--depth", type=int, default=DEFAULT_MODEL_DEPTH)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    hyperparameters = Hyperparameters(
        **json.loads(args.hyperparameters.read_text(encoding="utf-8"))
    )
    result = run_budget_audit(
        hyperparameters,
        arms=tuple(args.arms),
        budgets=tuple(args.budgets),
        seeds=tuple(args.seeds),
        control_seeds=tuple(args.control_seeds),
        epochs=args.epochs,
        width=args.width,
        depth=args.depth,
        verbose=args.verbose,
    )
    save_budget_audit(result, args.out_dir)
    print(json.dumps(result.summary_rows, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()