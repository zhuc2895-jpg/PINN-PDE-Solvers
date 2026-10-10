"""T8: frozen-model coupling and the under-relaxation scan."""

from __future__ import annotations

import json

import torch

from RANS_PINN_Demo.coupling import (
    ALPHAS,
    classify_coupling_status,
    run_alpha_scan,
    save_coupling_artifacts,
)
from RANS_PINN_Demo.experiment import Hyperparameters
from RANS_PINN_Demo.solver import SolveResult


def _solved(residuals, *, converged=False, steps=None) -> SolveResult:
    steps = steps if steps is not None else [value * 0.5 for value in residuals]
    return SolveResult(
        y=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        U=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        nu_t=torch.tensor([[0.0], [0.1]], dtype=torch.float64),
        iterations=len(residuals),
        converged=converged,
        residual_norms=list(residuals),
        step_norms=list(steps),
    )


def test_coupling_status_keeps_oscillation_and_failure_visible():
    assert classify_coupling_status(_solved([1.0, 0.5], converged=True)) == "converged"
    assert classify_coupling_status(_solved([1.0, 0.5, 0.8])) == "oscillated"
    assert classify_coupling_status(_solved([1.0, 2.0])) == "diverged"
    assert classify_coupling_status(_solved([1.0, 0.1], steps=[0.5, 0.05]), tol=1e-3) == "stalled"
    assert classify_coupling_status(_solved([1.0, float("nan")])) == "non_finite"


def test_alpha_scan_freezes_weights_and_archives_every_history(tmp_path):
    hyperparameters = Hyperparameters(learning_rate=1e-2)
    result = run_alpha_scan(
        hyperparameters,
        seeds=(0,),
        alphas=ALPHAS,
        epochs=1,
        width=8,
        depth=1,
        max_iter=50,
    )

    assert len(result.runs) == len(ALPHAS)
    assert {run.alpha for run in result.runs} == set(ALPHAS)
    assert all(run.model_weight_drift == 0.0 for run in result.runs)
    assert all(run.iterations == len(run.residual_norms) for run in result.runs)
    assert all(run.iterations == len(run.step_norms) for run in result.runs)
    assert all(run.final_residual == run.residual_norms[-1] for run in result.runs)
    assert all(run.final_field_change == run.step_norms[-1] for run in result.runs)
    assert all(run.profile_rel_l2 >= 0.0 for run in result.runs)
    assert all(run.wall_shear_rel_error >= 0.0 for run in result.runs)

    out = tmp_path / "results"
    figure = tmp_path / "figures" / "coupling_residual_history.png"
    save_coupling_artifacts(result, out, figure_path=figure)

    assert (out / "coupling_history.csv").is_file()
    assert (out / "coupling_summary.csv").is_file()
    assert (out / "coupling_report.json").is_file()
    assert figure.is_file() and figure.stat().st_size > 0

    report = json.loads((out / "coupling_report.json").read_text(encoding="utf-8"))
    assert report["budget"] == 12
    assert report["Re_tau"] == 590.0
    assert len(report["runs"]) == len(ALPHAS)
    assert all(run["model_weight_drift"] == 0.0 for run in report["runs"])
