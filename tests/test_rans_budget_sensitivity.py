"""Pre-2D audit: budget sensitivity must separate failures from accuracy."""

from __future__ import annotations

import csv
import json

from RANS_PINN_Demo.budget_sensitivity import run_budget_audit, save_budget_audit
from RANS_PINN_Demo.experiment import Hyperparameters


def test_budget_audit_keeps_failure_counts_and_controls(tmp_path):
    result = run_budget_audit(
        Hyperparameters(learning_rate=1e-2),
        arms=("all_off", "full"),
        budgets=(8,),
        seeds=(0, 1),
        control_seeds=(0,),
        epochs=1,
        width=8,
        depth=1,
    )

    assert len(result.runs) == 4
    assert len(result.summary_rows) == 2
    assert all(row["n_seeds"] == 2 for row in result.summary_rows)
    assert all(row["n_noise_free_controls"] == 1 for row in result.summary_rows)
    assert all(0 <= row["noise_free_converged"] <= 1 for row in result.summary_rows)

    save_budget_audit(result, tmp_path)
    assert (tmp_path / "budget_sensitivity_runs.csv").is_file()
    assert (tmp_path / "budget_sensitivity.csv").is_file()
    assert (tmp_path / "budget_sensitivity.json").is_file()

    report = json.loads(
        (tmp_path / "budget_sensitivity.json").read_text(encoding="utf-8")
    )
    assert report["control_seeds"] == [0]
    assert len(report["runs"]) == 4
    assert len(report["summary"]) == 2

    with (tmp_path / "budget_sensitivity.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["arm"] for row in rows} == {"all_off", "full"}