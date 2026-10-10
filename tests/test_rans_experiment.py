"""T6: validation split, equal-budget sweep, and per-seed artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from RANS_PINN_Demo.channel import DATA_BUDGETS, stratified_points
from RANS_PINN_Demo.experiment import (
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
    assert len(result.summary_rows) == 4  # two arms x Re_tau=180,395
    assert all(row["n_seeds"] == 2 for row in result.summary_rows)

    out = tmp_path / "results"
    figure = tmp_path / "figures" / "data_volume_curve.png"
    save_experiment(result, out, figure_path=figure)

    assert (out / "main_runs.csv").is_file()
    assert (out / "error_table.csv").is_file()
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
