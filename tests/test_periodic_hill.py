"""Geometry, independent-reference ingestion, and the coarse 2-D baseline."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from RANS_PINN_Demo.periodic_hill import (
    CHANNEL_HEIGHT,
    KINEMATIC_VISCOSITY,
    PERIOD,
    REFERENCE_ARCHIVE_SHA256,
    REFERENCE_STATIONS,
    download_reference,
    hill_height,
    load_reference,
    parse_profile,
    reference_wall_shear,
    wall_normal_coordinate,
)
from RANS_PINN_Demo.periodic_hill_solver import (
    PeriodicHillSolver,
    separation_reattachment,
    solve_periodic_hill,
    wall_shear_profile,
)


def _write_reference_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index, station in enumerate(REFERENCE_STATIONS, start=1):
        x_header = f"{station:.10E}" if station == 0.05 else f"{station}"
        columns = [
            "0.0 0.0 0.0 0.0 0.0 0.0 0.0",
            "0.5 0.5 0.01 0.01 0.001 0.001 0.005",
            "1.0 1.0 0.02 0.02 0.002 0.002 0.010",
        ]
        text = (
            f"# fixture\n# Data at location X/h = {x_header}\n"
            + "\n".join(columns)
            + "\n"
        )
        (root / f"data-{index:03d}.dat").write_text(text, encoding="utf-8")


def test_hill_geometry_matches_published_knots():
    x = np.array(
        [
            0.0,
            9.0 / 28.0,
            14.0 / 28.0,
            20.0 / 28.0,
            30.0 / 28.0,
            40.0 / 28.0,
            54.0 / 28.0,
            2.0,
        ]
    )
    expected = np.array([28.0, 27.0, 24.0, 19.0, 11.0, 4.0, 0.0, 0.0]) / 28.0
    assert np.allclose(hill_height(x), expected, atol=1e-6)
    assert hill_height(np.array([PERIOD + 0.0]))[0] == pytest.approx(1.0)


def test_wall_normal_coordinate_uses_local_hill_height():
    y = np.array([1.0, 1.5, 2.0])
    x = np.array([0.0, 0.5, 2.0])
    shift = wall_normal_coordinate(y, x)
    assert shift[0] == pytest.approx(0.0)
    assert shift[1] == pytest.approx(1.5 - hill_height(np.array([0.5]))[0])
    assert shift[2] == pytest.approx(2.0)


def test_reference_parser_and_loader_snap_fortran_header(tmp_path):
    _write_reference_fixture(tmp_path)
    profiles = load_reference(tmp_path)
    assert sorted(profiles) == list(REFERENCE_STATIONS)
    profile = profiles[0.05]
    assert profile.x_over_h == pytest.approx(0.05)
    assert profile.y_over_h.shape == (3,)
    assert profile.u_over_ub[1] == pytest.approx(0.5)
    # At x/h=0.05 the first fixture sample (y=0) lies inside the h=1 hill,
    # so the wall-normal coordinate is negative; the shift is the point.
    assert profile.wall_normal[0] == pytest.approx(-1.0)
    assert profiles[5.0].wall_normal[0] == pytest.approx(0.0)


def test_reference_wall_shear_uses_molecular_viscosity(tmp_path):
    _write_reference_fixture(tmp_path)
    profile = load_reference(tmp_path)[5.0]
    expected_slope = (profile.u_over_ub[1] - profile.u_over_ub[0]) / (
        profile.y_over_h[1] - profile.y_over_h[0]
    )
    assert reference_wall_shear(profile) == pytest.approx(
        KINEMATIC_VISCOSITY * expected_slope
    )


def test_download_reference_verifies_and_extracts(tmp_path):
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "data-001.dat",
            "# X/h = 0.05\n0.0 0.0 0.0 0.0 0.0 0.0 0.0\n",
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    extracted = download_reference(
        tmp_path / "download", url=source.as_uri(), sha256=digest
    )
    assert (extracted / "data-001.dat").is_file()


def test_download_reference_rejects_bad_checksum(tmp_path):
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("data-001.dat", "ignored")
    with pytest.raises(ValueError):
        download_reference(
            tmp_path / "download",
            url=source.as_uri(),
            sha256="0" * 64,
        )


def test_flat_channel_limit_matches_parabolic_profile():
    solver = PeriodicHillSolver(
        nx=16,
        ny=10,
        re_h=1.0,
        mixing_length=False,
        dt=0.005,
        f0=1.0,
        controller=False,
    )
    solver.b_u[:] = 0.0
    solver.b_v[:] = 0.0
    solver.b_c[:] = 0.0
    solver.chi_u[:] = 0.0
    solver.chi_v[:] = 0.0
    solver.chi_c[:] = 0.0
    solver.u[:] = 0.0
    solution = solver.run(max_steps=2000, tol=1e-12, patience=1000)
    exact = solver.f0 / (2.0 * solver.nu) * (
        solver.y_c * (solver.height - solver.y_c)
    )
    profile = solution.u.mean(axis=0)
    assert np.max(np.abs(profile - exact)) < 0.02 * exact.max()
    assert np.max(np.abs(solution.v)) == pytest.approx(0.0, abs=1e-12)


def test_periodic_hill_smoke_solution_and_wall_diagnostics():
    solver = PeriodicHillSolver(
        nx=16,
        ny=10,
        dt=0.005,
        f0=0.02,
        controller=False,
    )
    solver.u[:] = 1.0
    solution = solver.run(max_steps=20, tol=0.0, patience=1000)
    assert np.all(np.isfinite(solution.u))
    assert np.all(np.isfinite(solution.v))
    assert np.allclose(solution.u[solver.chi_u > 0.5], 0.0)
    assert np.allclose(solution.v[solver.chi_v > 0.5], 0.0)
    x, tau = wall_shear_profile(solution)
    assert x.shape == tau.shape == (solver.nx,)
    # The short smoke solve need not produce a converged separation bubble,
    # but the diagnostic must return either a float or None without raising.
    separation, reattachment = separation_reattachment(solution)
    assert separation is None or np.isfinite(separation)
    assert reattachment is None or np.isfinite(reattachment)


def test_2d_plan_records_independent_reference_and_solver_gate():
    root = Path(__file__).resolve().parents[1]
    plan = (root / "docs" / "2d-periodic-hill-plan.md").read_text(encoding="utf-8")
    for required in (
        "ERCOFTAC",
        "case 081",
        "61ca8053a187c67d914f1ab951a003136bd92c4b61e7de7b359d18002978030b",
        "独立",
        "tau_w",
        "力平衡",
        "20 个种子",
        "T10.2",
    ):
        assert required in plan
