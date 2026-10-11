"""T9: report, ADR, and the two generated profile figures."""

from __future__ import annotations

from pathlib import Path

from RANS_PINN_Demo.deliverables import (
    build_adr_0003_markdown,
    build_adr_markdown,
    build_report_markdown,
    collect_deliverable_data,
    plot_eddy_viscosity_correction,
    plot_velocity_profiles,
)
from RANS_PINN_Demo.experiment import Hyperparameters


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_one_page_report_uses_archived_pre_registered_results():
    root = _repo_root() / "RANS_PINN_Demo"
    report = build_report_markdown(
        results_dir=root / "results",
        figures_dir=root / "figures",
    )

    assert "H1" in report and "H2" in report and "H3" in report
    assert "88.1%" in report
    assert "9/9" in report
    assert "A2/A3 的单轴归因不通过" in report
    assert "TODO" not in report


def test_adr_records_the_2d_decision_and_caveat():
    adr = build_adr_markdown()
    assert "ADR-0002" in adr
    assert "Accepted" in adr
    assert "二维周期性山丘" in adr
    assert "A2/A3" in adr


def test_adr_0003_removes_the_constructed_wall_shear_criterion():
    adr = build_adr_0003_markdown()
    assert "ADR-0003" in adr
    assert "构造恒等式" in adr
    assert "20-seed" in adr
    assert "u_tau" in adr
    assert "力平衡" in adr


def test_t9_profile_figures_can_be_generated(tmp_path):
    data = collect_deliverable_data(
        Hyperparameters(learning_rate=1e-2),
        seeds=(0,),
        epochs=1,
        width=8,
        depth=1,
        Re_taus=(590.0,),
    )
    velocity = tmp_path / "velocity.png"
    correction = tmp_path / "correction.png"

    plot_velocity_profiles(data, velocity, seeds=(0,), Re_taus=(590.0,))
    plot_eddy_viscosity_correction(
        data, correction, seeds=(0,), Re_taus=(590.0,)
    )

    assert velocity.is_file() and velocity.stat().st_size > 0
    assert correction.is_file() and correction.stat().st_size > 0
