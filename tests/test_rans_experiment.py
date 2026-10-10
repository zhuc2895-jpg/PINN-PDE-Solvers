"""T6: validation split, equal-budget sweep, and per-seed artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from RANS_PINN_Demo.channel import DATA_BUDGETS, stratified_points
from RANS_PINN_Demo.experiment import (
    TEST_RE_TAU,
    Hyperparameters,
    run_experiment,
    save_experiment,
    validation_points,
)
from RANS_PINN_Demo.solver import wall_clustered_grid


def test_validation_points_are_stratified_and_held_out():
    validation = validation_points().reshape(-1)
    assert validation.numel() == 12
    assert torch.all(validation > 0.0)
    assert torch.all(validation < 1.0)

    for budget in DATA_BUDGETS:
        training = stratified_points(budget, 395.0).reshape(-1)
        for value in validation:
            assert not torch.any(torch.isclose(training, value, rtol=0.0, atol=1e-12))

    grid = wall_clustered_grid().reshape(-1)
    for value in validation:
        assert not torch.any(torch.isclose(grid, value, rtol=0.0, atol=1e-12))


def test_the_sweep_archives_every_seed_and_summarizes_equal_cells(tmp_path):
    hp = Hyperparameters(learning_rate=1e-2)
    result = run_experiment(
        candidates=(hp,),
        arms=("all_off", "A2"),
        budgets=(8,),
        seeds=(0, 1),
        epochs=1,
        width=8,
        depth=1,
        select=True,
        selection_budget=8,
    )

    assert result.selected == hp
    assert len(result.runs) == 4
    assert all(run.test_profile.Re_tau == TEST_RE_TAU for run in result.runs)
    assert len(result.summary_rows) == 4  # two arms x Re_tau=180,395
    assert all(row["n_seeds"] == 2 for row in result.summary_rows)
    assert len(result.extrapolation_rows) == 2  # two arms x unseen Re_tau=590
    assert all(row["n_seeds"] == 2 for row in result.extrapolation_rows)

    out = tmp_path / "results"
    figure = tmp_path / "figures" / "data_volume_curve.png"
    save_experiment(result, out, figure_path=figure)

    assert (out / "main_runs.csv").is_file()
    assert (out / "error_table.csv").is_file()
    assert (out / "extrapolation_runs.csv").is_file()
    assert (out / "extrapolation_table.csv").is_file()
    assert (out / "frozen_hyperparameters.json").is_file()
    assert (out / "validation_selection.csv").is_file()
    assert figure.is_file() and figure.stat().st_size > 0

    archived = sorted((out / "runs").glob("*.json"))
    assert len(archived) == 4
    assert {path.name for path in archived} == {
        "A2_budget8_seed0.json",
        "A2_budget8_seed1.json",
        "all_off_budget8_seed0.json",
        "all_off_budget8_seed1.json",
    }

    with (out / "main_runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8  # four runs x two training Re_tau
    assert {row["arm"] for row in rows} == {"A2", "all_off"}
    assert {row["budget"] for row in rows} == {"8"}
    assert {row["seed"] for row in rows} == {"0", "1"}


def test_unseen_reynolds_is_evaluated_once_per_frozen_model(monkeypatch):
    from RANS_PINN_Demo import experiment as experiment_module

    original = experiment_module.profile_metrics
    calls = []

    def counted(problem, model, Re_tau):
        if Re_tau == TEST_RE_TAU:
            calls.append((problem, model))
        return original(problem, model, Re_tau)

    monkeypatch.setattr(experiment_module, "profile_metrics", counted)
    hp = Hyperparameters(learning_rate=1e-2)
    result = run_experiment(
        candidates=(hp,),
        arms=("full", "A1"),
        budgets=(8,),
        seeds=(0, 1),
        epochs=1,
        width=8,
        depth=1,
        select=True,
        selection_budget=8,
    )

    assert len(result.runs) == 4
    assert len(calls) == len(result.runs)
    assert len({id(model) for _, model in calls}) == len(result.runs)
    assert all(run.test_profile.Re_tau == TEST_RE_TAU for run in result.runs)


def test_axis_attribution_reports_full_vs_each_ablation(tmp_path):
    hp = Hyperparameters(learning_rate=1e-2)
    result = run_experiment(
        candidates=(hp,),
        arms=("full", "A1", "A2", "A3"),
        budgets=(12,),
        seeds=(0, 1),
        epochs=1,
        width=8,
        depth=1,
        select=False,
    )

    assert {row["axis"] for row in result.ablation_rows} == {
        "structure",
        "rans_residual",
        "smoothness",
    }
    assert all(row["budget"] == 12 and row["Re_tau"] == TEST_RE_TAU for row in result.ablation_rows)
    assert len(result.criteria_rows) == 4
    assert {row["scope"] for row in result.criteria_rows} == {
        "full",
        "structure",
        "rans_residual",
        "smoothness",
    }

    out = tmp_path / "results"
    figure = tmp_path / "figures" / "physics_ablation.png"
    save_experiment(
        result,
        out,
        figure_path=tmp_path / "figures" / "data_volume_curve.png",
        ablation_figure_path=figure,
    )

    assert (out / "ablation_table.csv").is_file()
    assert (out / "criteria.csv").is_file()
    assert figure.is_file() and figure.stat().st_size > 0

    with (out / "ablation_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["ablated_arm"] for row in rows} == {"A1", "A2", "A3"}
