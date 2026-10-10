"""T3: exact profile inversion, error split, and identifiability reporting."""

from __future__ import annotations

import csv

import torch

from RANS_PINN_Demo.closure import TruthClosure
from RANS_PINN_Demo.inversion import (
    TabulatedClosure,
    analyze_inversion,
    decompose_errors,
    identifiability_report,
    invert_eddy_viscosity,
    replay_error,
    save_inversion_artifacts,
)
from RANS_PINN_Demo.solver import solve_channel, wall_clustered_grid


def test_inversion_recovers_a_known_closure_and_replays_the_profile():
    Re_tau = 395.0
    y = wall_clustered_grid(n=201)
    solution = solve_channel(TruthClosure(), Re_tau, y=y)

    inversion = invert_eddy_viscosity(solution.U, y, Re_tau)
    truth = TruthClosure()(y, Re_tau).reshape(-1)
    mask = inversion.identifiable.reshape(-1)
    nu_t = inversion.nu_t.reshape(-1)

    assert mask.sum().item() == y.numel() - 1
    assert bool(torch.isnan(inversion.nu_t[-1]))
    recovered_error = torch.norm(nu_t[mask] - truth[mask]) / torch.norm(truth[mask])
    assert recovered_error < 1e-8
    assert replay_error(solution.U, inversion) < 1e-12


def test_the_error_split_is_additive_and_closure_dominated():
    decomposition = decompose_errors(
        395.0,
        coarse_y=wall_clustered_grid(n=201),
        fine_y=wall_clustered_grid(n=2001),
    )

    assert torch.allclose(
        decomposition.total_error_U,
        decomposition.closure_error_U + decomposition.discretization_error_U,
        rtol=0.0,
        atol=1e-14,
    )
    assert decomposition.closure_relative_l2 > decomposition.discretization_relative_l2
    assert decomposition.inversion_relative_error < 1e-8
    assert decomposition.replay_relative_error < 1e-12
    assert abs(decomposition.closure_share + decomposition.discretization_share - 1.0) < 1e-12


def test_identifiability_flags_the_zero_stress_centreline():
    report = identifiability_report(395.0, y=wall_clustered_grid(n=401))

    assert not report.centreline_identifiable
    assert report.centreline_sensitivity < 1e-12
    assert len(report.bands) == 6
    assert any(not band.identifiable for band in report.bands)
    assert not report.overall_identifiable


def test_artifacts_include_tables_figure_and_explicit_unidentifiable_rows(tmp_path):
    analysis = analyze_inversion((180.0, 395.0), coarse_n=201, fine_n=2001)
    out = tmp_path / "results"
    figure = tmp_path / "figures" / "inversion_error_decomposition.png"
    save_inversion_artifacts(analysis, out, figure_path=figure)

    assert (out / "inversion_error_table.csv").is_file()
    assert (out / "identifiability.csv").is_file()
    assert (out / "inversion_report.json").is_file()
    assert figure.is_file() and figure.stat().st_size > 0

    with (out / "inversion_error_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all("centreline" in row["unidentifiable"] for row in rows)

    with (out / "identifiability.csv").open(newline="", encoding="utf-8") as handle:
        sensitivity_rows = list(csv.DictReader(handle))
    centreline = [row for row in sensitivity_rows if row["is_centreline"] == "True"]
    assert len(centreline) == 2
    assert all(row["identifiable"] == "False" for row in centreline)


def test_tabulated_closure_interpolates_a_prescribed_profile():
    y = wall_clustered_grid(n=101)
    closure = TabulatedClosure(y, y)
    query = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    values = closure(query, 395.0).reshape(-1)
    assert torch.allclose(
        values,
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        atol=1e-12,
    )
